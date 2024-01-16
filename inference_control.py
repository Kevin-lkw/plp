from share import *
import config
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
#from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image

mask_shape = [1, 512, 512,]

def create_seg(seq_t : int, mask_array : np.ndarray)->np.ndarray:
        '''
        mask_array is gt_mask[0:seq_t]
        '''

        mask_array = mask_array.astype(np.int32)
        print('mask_array: ', mask_array.shape)
        h, w = mask_array.shape[1], mask_array.shape[2]

        seg = np.zeros((h, w))
        prev_mask = seg
        for i in range(seq_t):
            curr_mask = mask_array[i] - prev_mask
            prev_mask = mask_array[i]
            seg[curr_mask==1] = i+1
        
        return seg # (h, w)


def inference_step(model, prompt, batch_size, seq_t, inference_mask, ddim_steps=50):
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    a_prompt = '' 
    n_prompt = ''
    
    seg = create_seg(seq_t, inference_mask)
    seg = np.tile(seg, (3, 1, 1)).transpose(1,2,0) # 512x512x3
    inference_mask = inference_mask.astype(np.float32)
    # Normalize hint images to [0, 1].
    seg = seg.astype(np.float32) / (seg.max() + 1e-6)
            
    hint = torch.from_numpy(seg.copy()).cuda()
    hint = torch.stack([hint for _ in range(batch_size)], dim=0)
    hint = einops.rearrange(hint, 'b h w c -> b c h w').clone()
            
    seq_t = torch.stack([torch.tensor(seq_t) for _ in range(batch_size)], dim=0).cuda()
    mask_t = torch.from_numpy(inference_mask.copy())
    mask_t = torch.stack([mask_t for _ in range(batch_size)], dim=0).cuda()
            
    cond = {"c_concat": [hint], 
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt]*batch_size)],
            "seq_t": seq_t, 
            "mask": mask_t}
    un_cond = {"c_concat": [hint], 
           "c_crossattn": [model.get_learned_conditioning([n_prompt]*batch_size)],
           "seq_t": seq_t, 
           "mask": mask_t}           
            
    shape = (4, 512 // 8, 512 // 8)
    strength = 1.0
    threshold = 0.6

    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    model.control_scales = ([strength] * 13)  
            
    samples, intermediates, mask_pred = ddim_sampler.sample(ddim_steps, batch_size,
                                                 shape, cond, verbose=False, eta=0.0,
                                                 unconditional_guidance_scale=9.0,
                                                 unconditional_conditioning=un_cond, 
                                                 inference=True)

    x_samples = model.decode_first_stage(samples)
    x_samples = einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
    x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)
            
    # where x > threshold clip to 1, else 0
    mask_pred = mask_pred.cpu().numpy()
    mask_pred = np.where(mask_pred - np.floor(mask_pred) >= threshold, \
                        np.ceil(mask_pred), np.floor(mask_pred))

    #results = [x_samples[i] for i in range(batch_size)]
    inference_mask = np.concatenate((inference_mask, mask_pred[0][np.newaxis, ...]), axis=0)
    print('infer_mask: ', inference_mask.shape)
    return mask_pred, inference_mask, x_samples # expected 512*512

def inference(paint_timestep, model, prompt, a_prompt, n_prompt, seed, batch_size=4, 
              ddim_steps=50, guess_mode=False, strength=1.0, scale=9.0, eta=0.0):

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    img_shape = [512, 512, 3]
    threshold = 0.6
    segement = []
    results = []

    with torch.no_grad():
        H, W, C = img_shape

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        
        for seq_t in range(paint_timestep):
            seg = create_seg(seq_t, mask)
            seg = np.tile(seg, (3, 1, 1)).transpose(1,2,0) # 512x512x3
            segement.append(seg)
            mask = mask.astype(np.float32)
            # Normalize hint images to [0, 1].
            seg = seg.astype(np.float32) / (seg.max() + 1e-6)
            
            hint = torch.from_numpy(seg.copy()).cuda()
            hint = torch.stack([hint for _ in range(batch_size)], dim=0)
            hint = einops.rearrange(hint, 'b h w c -> b c h w').clone()
            
            seq_t = torch.stack([torch.tensor(seq_t) for _ in range(batch_size)], dim=0).cuda()
            mask_t = torch.from_numpy(mask.copy())
            mask_t = torch.stack([mask_t for _ in range(batch_size)], dim=0).cuda()
            
            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)
            
            cond = {"c_concat": [hint], 
                    "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt]*batch_size)],
                    "seq_t": seq_t, 
                    "mask": mask_t}
            un_cond = {"c_concat": None if guess_mode else [hint], 
                   "c_crossattn": [model.get_learned_conditioning([n_prompt]*batch_size)],
                   "seq_t": seq_t, 
                   "mask": mask_t}           
            
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] \
                                    if guess_mode else ([strength] * 13)  
            
            samples, intermediates, mask_pred = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, eta=eta,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond, 
                                                         inference=True)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
            x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)
            
            # where x > threshold clip to 1, else 0
            mask_pred = mask_pred.cpu().numpy()
            mask_pred = np.where(mask_pred - np.floor(mask_pred) >= threshold, \
                                np.ceil(mask_pred), np.floor(mask_pred))

            #results = [x_samples[i] for i in range(batch_size)]
            results.append(x_samples[0])
            mask = np.concatenate((mask, mask_pred[0][np.newaxis, ...]), axis=0)
    
    mask = mask.astype(np.int32)
    
    return results, mask, segement


if __name__ == "__main__":
    paint_timestep = 4
    prompt = "A couple of people sitting at a table while using smart phones"
    a_prompt = ""
    n_prompt = ""
    seed = 43
    output_path = 'Inference/'
    syn_img_seq, syn_img_mask, syn_img_seg = inference(paint_timestep, prompt,
                                                        a_prompt, n_prompt, seed)
    
    image_height = syn_img_seq[0].shape[0] * 2
    image_width = syn_img_seq[0].shape[1] * len(syn_img_seq)
    output_image = Image.new("RGB", (image_width, image_height))
    
    x_offset = 0
    y_offset = syn_img_seq[0].shape[0]
    for i, syn_img in enumerate(syn_img_seq):
        syn_imgwith_mask = syn_img * syn_img_mask[i+1][..., np.newaxis]
        image = Image.fromarray(syn_img.astype('uint8'))
        image_mask = Image.fromarray(syn_imgwith_mask.astype('uint8'))
        output_image.paste(image, (x_offset, 0, x_offset + syn_img.shape[1], syn_img.shape[0]))
        output_image.paste(image_mask, (x_offset, y_offset, \
                                        x_offset + syn_img.shape[1], y_offset + syn_img.shape[0]))
        x_offset += syn_img.shape[1]
        
    output_image.save(output_path +'out.png')