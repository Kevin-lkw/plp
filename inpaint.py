from typing import Tuple

from PIL import Image
import os, sys
import numpy as np
import torch as th
import torch.nn.functional as F
from cldm.model import create_model, load_state_dict
import pdb
# from torchvision.utils import save_images

from inference import inference, inference_mask, inference_step

sys.path.append('glide-text2im')

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
save_path = 'output/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
#     197
# (Pdb) len(mask_model.state_dict().keys())
# 1638
# (Pdb) 


# resume_path = './models/plp_ini.ckpt'
resume_path = 'lightning_logs/epoch=20-step=6573.ckpt'
mask_model = create_model('./models/plp_model.yaml').cpu()
pdb.set_trace()
mask_model.load_state_dict(load_state_dict(resume_path, location='cpu'))
pdb.set_trace()

threshold = 0.5

def save_images(batch: th.Tensor, path: str):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    img = Image.fromarray(reshaped.numpy())
    img.save(path)
    
    # save image


def read_image(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


def main():
    # Create base model.
    options = model_and_diffusion_defaults()
    options['inpaint'] = True
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base-inpaint', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['inpaint'] = True
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample-inpaint', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

    # Sampling parameters
    prompt = "a cat in a field"
    batch_size = 1
    guidance_scale = 5.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    # Source image we are inpainting
    source_image_256 = read_image('glide-text2im/notebooks/grass.png', size=256)
    source_image_64 = read_image('glide-text2im/notebooks/grass.png', size=64)

    # The mask should always be a boolean 64x64 mask, and then we
    # can upsample it for the second stage.
    source_mask_64 = th.zeros_like(source_image_64)[:, :1]
    # source_mask_64[:, :, 20:] = 0
    source_mask_256 = F.interpolate(source_mask_64, (256, 256), mode='nearest')

    # Visualize the image we are inpainting
    save_images(source_image_256 * source_mask_256, save_path+'img_mask.png')

    ##############################
    # Sample from the base model #
    ##############################

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    for t in range(10):
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
    
            # Masked inpainting image
            # change here to allow auto regressive
            inpaint_image=(source_image_64 * source_mask_64).repeat(full_batch_size, 1, 1, 1).to(device),
            inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
        )
    
        # Create an classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return th.cat([eps, rest], dim=1)
    
        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_start * (1 - model_kwargs['inpaint_mask'])
                + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
            )
    
        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            denoised_fn=denoised_fn,
        )[:batch_size]
        model.del_cache()
    
        # Show the output
        save_images(samples, save_path+f'out_64_00{t}.png')
    
        ##############################
        # Upsample the 64x64 samples #
        ##############################
    
        # tokens = model_up.tokenizer.encode(prompt)
        # tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        #     tokens, options_up['text_ctx']
        # )
    
        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,
    
            # Text tokens
            tokens=th.tensor(
                [tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
    
            # Masked inpainting image.
            inpaint_image=(source_image_256 * source_mask_256).repeat(batch_size, 1, 1, 1).to(device),
            inpaint_mask=source_mask_256.repeat(batch_size, 1, 1, 1).to(device),
        )
    
        # Sample from the base model.
        model_up.del_cache()
        up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
        up_samples = diffusion_up.p_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            denoised_fn=denoised_fn,
        )[:batch_size]
        model_up.del_cache()
        
        ###
        # predict new mask 
        mask_pred = inference_step(model=mask_model,
                              prompt=prompt, batch_size=batch_size, seq_t=t,
                              inference_mask=inference_mask
                              )
        # new_img = up_samples
        pdb.set_trace()
        
        assert mask_pred.shape == (1, 512, 512)
        mask_pred = mask_pred[0]
        
        # update mask and image
        source_mask_64 = mask_pred.reshape(64, 64, 8, 8).mean(axis=(2, 3))
        source_mask_256 = mask_pred.reshape(256, 256, 2, 2).mean(axis=(2, 3))
        
        source_mask_64[source_mask_64>threshold] = 1
        source_mask_64[source_mask_64<=threshold] = 0
        source_mask_256[source_mask_256>threshold] = 1
        source_mask_256[source_mask_256<=threshold] = 0
        
        source_image_256 = up_samples
        assert source_image_256.shape == (256, 256, 3)
        source_image_64 = source_image_256.reshape(64, 64, 4, 4, 3).mean(axis=(2, 3))
    
        # Show the output
        save_images(up_samples, save_path+f'out_256_00{t}.png')

if __name__ == '__main__':
    main()