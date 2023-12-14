import numpy as np
import torch
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from seq_dataset import Raw_Data5k_Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/plp_ini.ckpt'
#batch_size = 4
#logger_freq = 300
#learning_rate = 1e-5
#sd_locked = True
#only_mid_control = False

print('qwq')

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/plp_model.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))

def get_batch(seq_t, jpg, mask, caption):
    txt_t = np.random.randint(0, len(caption))
    hint = mask * jpg

    return {
            'seq_t': seq_t,
            'jpg': jpg, #from last step
            'txt': caption[txt_t],
            'hint': hint,
            'mask': mask, #from last step
        }


def inference(img_path, prompt, sample_timestep, N=1, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                use_ema_scope=True,
                **kwargs):
    use_ddim = ddim_steps is not None
    
    img_shape = [512, 512, 3]
    mask_shape = [512, 512, 1]
    img = np.zeros(img_shape)
    mask = np.ones(mask_shape)

    z, c = model.get_input(get_batch(0, img, mask, prompt), bs=N)
    c_cat, c, seq_t, mask = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c['seq_t'][:N], c['mask'][:N]
    N = min(z.shape[0], N)
    n_row = min(z.shape[0], n_row)
    #log["reconstruction"] = model.decode_first_stage(z)
    #log["control"] = c_cat * 2.0 - 1.0
    #log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

    samples_cfg, _ = model.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "seq_t": seq_t, "mask": mask},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=None,
                                             )
    x_samples_cfg = model.decode_first_stage(samples_cfg)
    #log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg