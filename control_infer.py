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

from inference_control import inference, inference_step, create_seg

from seq_dataset import Raw_Data5k_Dataset, Segment_Hint5k_Dataset
from datetime import datetime

'''
    This module using finetuned PLP_controlnet to inference masks and images.
'''


def save_images(batch: th.Tensor, path: str):
    """ Display a batch of images inline. """
    scaled = (batch).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    img = Image.fromarray(reshaped.numpy())
    img.save(path)
    

def generate_single_image(model,
                        prompt, 
                        image_gt, 
                        save_path, 
                        ):

    # save img_gt
    image_gt_pil = TF.to_pil_image(image_gt.squeeze(0).permute(2, 0, 1))
    image_gt_pil.save(f'{save_path}/image_gt_512.png')

    # save text
    with open(f'{save_path}/prompt.txt', 'w') as f:
        f.write(prompt)

    inference_mask = np.zeros([1, 512, 512]).astype(np.float32)                    
    for t in range(5):
        # generate new image and image
        mask_pred, inference_mask, samples = inference_step(model=model,
                            prompt=prompt, batch_size=1, seq_t=t,
                            inference_mask=inference_mask
                            )

        mask_pred = th.from_numpy(mask_pred)
        mask_pred[mask_pred>threshold] = 1
        mask_pred[mask_pred<=threshold] = 0

        # pdb.set_trace()

        # get segmentation
        seg = create_seg(t, inference_mask) # 512x512
        seg = (seg - seg.min()) / (seg.max()-seg.min())
        seg = th.from_numpy(seg)

        # save results
        samples = np.transpose(samples, (0, 3, 1, 2))
        samples = th.from_numpy(samples)
        save_images(samples, f'{save_path}/out_512_00{t}.png')
        
        mask_pil = TF.to_pil_image(mask_pred)
        mask_pil.save(f'{save_path}/mask_512_00{t}.png')

        seg_pil = TF.to_pil_image(seg)
        seg_pil.save(f'{save_path}/seg_512_00{t}.png')


if __name__ == "__main__":

    current_time = datetime.now()
    formatted_time = current_time.strftime(r"%d_%H:%M")
    save_dir = f'output_control/{formatted_time}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = None


    # # load image generater
    # img_model_path = 'lightning_logs/version_22032_seg/checkpoints/epoch=123-step=155000.ckpt'
    # model = create_model('./models/plp_model.yaml').cpu()
    # model.load_state_dict(load_state_dict(img_model_path), location='cpu'))

    # load mask predicter
    # resume_path = './models/plp_ini.ckpt'
    resume_path = 'lightning_logs/version_24881/checkpoints/epoch=20-step=6573.ckpt'
    mask_model = create_model('./models/plp_model.yaml').cpu()
    mask_model.load_state_dict(load_state_dict(resume_path, location='cpu'))

    threshold = 0.5

    # load dataset
    dataset = Segment_Hint5k_Dataset()
    # dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)

    for idx in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(idx)
        jpg, txt = batch['jpg'][np.newaxis, ...], batch['txt']
        jpg = th.from_numpy(jpg)
        # denormalize jpg
        jpg = (jpg + 1) * 127.5
        jpg = jpg / 255.0
        # pdb.set_trace()
        image_path = dataset.data_path_list[idx]
        save_path = os.path.join(save_dir, image_path)
        os.makedirs(save_path, exist_ok=True)
        generate_single_image(model=mask_model, 
                            prompt=txt, 
                            image_gt=jpg, 
                            save_path=save_path, 
                            )
