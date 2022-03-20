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
from jukebox.utils.audio_utils import save_wav, load_audio

import torch.distributed as dist

if False: # False for windows
	rank, local_rank, device = setup_dist_from_mpi()
else:
	rank, local_rank, device = (0, 0, t.device('cuda') if t.cuda.is_available() else t.device('cpu'))
	print(device)
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "29500"
	dist.init_process_group("gloo", rank=0, world_size=1)

hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 1

restore_vqvae = 'models/5b/vqvae.pth.tar'
audio_file = 'samples/primer.wav'

vqvae, *priors = MODELS['5b_lyrics']
print('creating vqvae...')
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576, restore_vqvae = restore_vqvae)), device)
print('done.')

prompt_length_in_seconds = 1000000 # yes we go extreme here

audio_files = audio_file.split(',')
duration = (int(prompt_length_in_seconds*hps.sr)//128)*128
x = load_prompts(audio_files, duration, hps)

#encode it all
zs = vqvae.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])

#decode each level to seperate files
for level in range(len(zs)):
	x = vqvae.decode(zs[level:], start_level=level)
	pth = f'samples/vqvae_test_level_{level}'
	if not os.path.exists(pth):
		os.makedirs(pth)
	save_wav(pth, x, hps.sr)