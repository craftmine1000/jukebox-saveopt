import jukebox
import torch as t
import librosa
import os
from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
import torch.distributed as dist

if False: # False for windows
	rank, local_rank, device = setup_dist_from_mpi()
else:
	rank, local_rank, device = (0, 0, t.device('cuda') if t.cuda.is_available() else t.device('cpu'))
	print(device)
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "29500"
	dist.init_process_group("gloo", rank=0, world_size=1)


from torch.multiprocessing import get_sharing_strategy
print(get_sharing_strategy())

model = "1b_lyrics" # or "1b_lyrics"
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 1 if model=='5b_lyrics' else 1
hps.name = 'samples'
chunk_size = 16 if model=="5b_lyrics" else 32
max_batch_size = hps.n_samples#1 if model=="5b_lyrics" else 8
hps.levels = 3
hps.hop_fraction = [1,4,.125]

# Specify an audio file here.
audio_file = 'samples/primer.wav'
# Specify how many seconds of audio to prime on.
prompt_length_in_seconds=10

restore_prior = f'models/{model}/prior_level_2.pth.tar'
restore_vqvae = 'models/5b/vqvae.pth.tar'

vqvae, *priors = MODELS[model]
print('creating vqvae...')
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576, restore_vqvae = restore_vqvae)), device)
print('done.')

if True:
	print('encoding primer...')
	audio_files = audio_file.split(',')
	duration = (int(prompt_length_in_seconds*hps.sr)//128)*128
	x = load_prompts(audio_files, duration, hps)
	zs = vqvae.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
	print('done')
else:
	zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cpu') for _ in range(len(priors))]

print('creating prior...')
top_prior = make_prior(setup_hparams(priors[-1], dict(restore_prior = restore_prior)), vqvae, device)
print('done.')
sample_length_in_seconds = 60          # Full length of musical sample to generate - we find songs in the 1 to 4 minute
                                       # range work well, with generation time proportional to sample length.  
                                       # This total length affects how quickly the model 
                                       # progresses through lyrics (model also generates differently
                                       # depending on if it thinks it's in the beginning, middle, or end of sample)

hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

metas = [dict(artist = "unknown",
            genre = "unknown",
            total_length = hps.sample_length,
            offset = 0,
            lyrics = """""",
            ),
          ] * hps.n_samples
labels = [None, None, top_prior.labeller.get_batch_labels(metas, device)]

sampling_temperature = .985

lower_batch_size = 8
lower_level_chunk_size = 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True, 
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]

zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)

# Set this False if you are on a local machine that has enough memory (this allows you to do the
# lyrics alignment visualization during the upsampling stage). For a hosted runtime, 
# we'll need to go ahead and delete the top_prior if you are using the 5b_lyrics model.
if True:
	del top_prior
	empty_cache()
	top_prior=None
upsamplers = [make_prior(setup_hparams(prior, dict(restore_prior=f'models/5b/prior_level_{level}.pth.tar')), vqvae, 'cpu') for level, prior in enumerate(priors[:-1])]
labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]

zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

del upsamplers
empty_cache()