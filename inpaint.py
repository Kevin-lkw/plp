from typing import Tuple

from PIL import Image
from tqdm.auto import tqdm
import os, sys
import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from cldm.model import create_model, load_state_dict
import pdb
# from torchvision.utils import save_images

from inference import inference, inference_step

from seq_dataset import Raw_Data5k_Dataset, Segment_Hint5k_Dataset
from datetime import datetime

sys.path.append('glide-text2im')

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

current_time = datetime.now()
formatted_time = current_time.strftime("%H:%M:%S")
save_dir = f'output/{formatted_time}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = None



# resume_path = './models/plp_ini.ckpt'
resume_path = 'lightning_logs/version_24881/checkpoints/epoch=20-step=6573.ckpt'
mask_model = create_model('./models/plp_model.yaml').cpu()
mask_model.load_state_dict(load_state_dict(resume_path, location='cpu'))

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
    
    # load dataset
    dataset = Segment_Hint5k_Dataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)

    for idx, batch in tqdm(enumerate(dataloader)):
        jpg, txt = batch['jpg'], batch['txt']
        # denormalize jpg
        jpg = (jpg + 1) * 127.5
        jpg = jpg / 255.0
        # pdb.set_trace()
        image_path = dataset.data_path_list[idx]
        global save_path
        save_path = os.path.join(save_dir, image_path)
        os.makedirs(save_path, exist_ok=True)
        generate_single_image(model=model, model_up=model_up, prompt=txt[0], image_gt=jpg, 
                            options=options, options_up=options_up, diffusion=diffusion, diffusion_up=diffusion_up)


def generate_single_image(model, model_up, prompt, image_gt, options, options_up, diffusion, diffusion_up):

    # save img_gt
    image_gt_pil = TF.to_pil_image(image_gt.squeeze(0).permute(2, 0, 1))
    image_gt_pil.save(f'{save_path}/image_gt_512.png')

    # save text
    with open(f'{save_path}/prompt.txt', 'w') as f:
        f.write(prompt)

    # Sampling parameters
    batch_size = 1
    guidance_scale = 5.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    # Source image we are inpainting
    source_image_256 = th.zeros((1, 3, 256, 256))
    source_image_64 = th.zeros((1, 3, 64, 64))

    # The mask should always be a boolean 64x64 mask, and then we
    # can upsample it for the second stage.
    source_mask_64 = th.zeros_like(source_image_64)[:, :1]
    # source_mask_64[:, :, 20:] = 0
    source_mask_256 = F.interpolate(source_mask_64, (256, 256), mode='nearest')

    # Visualize the image we are inpainting
    save_images(source_image_256 * source_mask_256, f'{save_path}/img_mask.png')

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

    inference_mask = np.zeros([1, 512, 512]).astype(np.float32)

    # Pack the tokens together into model kwargs.
    for t in range(5):
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
            inpaint_image=(source_image_64 * source_mask_64).repeat(full_batch_size, 1, 1, 1).to(device), # (2, 3, 64, 64)
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
        save_images(samples, f'{save_path}/out_64_00{t}.png')
    
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
        mask_pred, inference_mask = inference_step(model=mask_model,
                              prompt=prompt, batch_size=batch_size, seq_t=t,
                              inference_mask=inference_mask
                              )

        mask_pred = th.from_numpy(mask_pred)
        # new_img = up_samples
        
        # assert mask_pred.shape == (1, 512, 512)
        
        # update mask and image
        source_mask_64 = mask_pred.unsqueeze(0)
        source_mask_256 = mask_pred.unsqueeze(0)

        source_mask_64 = F.interpolate(source_mask_64, size=(64, 64), mode='bilinear', align_corners=False).to(device)
        source_mask_256 = F.interpolate(source_mask_256, size=(256, 256), mode='bilinear', align_corners=False).to(device)
        
        source_mask_64[source_mask_64>threshold] = 1
        source_mask_64[source_mask_64<=threshold] = 0
        source_mask_256[source_mask_256>threshold] = 1
        source_mask_256[source_mask_256<=threshold] = 0
        
        source_image_256 = up_samples.to(device)
        assert source_image_256.shape == (1, 3, 256, 256)
        source_image_64 = F.interpolate(source_image_256, size=(64, 64), mode='bilinear', align_corners=False).to(device)
    
        # Show the output
        save_images(up_samples, f'{save_path}/out_256_00{t}.png')
        
        mask_pil = TF.to_pil_image(source_mask_256.squeeze(0))
        mask_pil.save(f'{save_path}/mask_256_00{t}.png')
        
        

if __name__ == '__main__':
    main()