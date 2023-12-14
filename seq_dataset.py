import json
import cv2
import numpy as np
import pdb
import pickle as pkl
import os
from PIL import Image
import random

from torch.utils.data import Dataset


class Raw_Data5k_Dataset(Dataset):
    '''
    hint = masked image
    jpg = raw image
    '''
    def __init__(self, 
                data_dir = './dataset/raw_data5k',
    ):
        self.data_dir = data_dir
        self.data_path_list = os.listdir(data_dir)
        

    def __len__(self):
        return len(self.data_path_list)


    def resize_image(self, image):
        assert image.shape[2] == 3

        H, W, _ = image.shape
        min_HW = min(H, W)

        target_size = int(H/min_HW * 512), int(W/min_HW * 512)
        target_size = max(512, target_size[0]), max(512, target_size[1])

        # 将NumPy数组转换为PIL图像
        pil_image = Image.fromarray(image.astype('uint8'))

        # 调整图像大小
        resized_image = pil_image.resize(target_size)

        # 将PIL图像转换回NumPy数组
        resized_np_image = np.array(resized_image)

        assert resized_np_image.shape[0] == 512 or resized_np_image.shape[1] == 512

        return resized_np_image

    def crop_img(self, array, left, top):
        # 截取512x512的图片区域
        cropped_array = array[top:top+512, left:left+512, ...]
        return cropped_array

    def __getitem__(self, idx):
        '''
        return a dict with keys: 
        'jpg': the target image, i.e., x_M
        'txt': txt description
        'hint': the hint image, i.e., x_{t-1}
        'seq_t': image seq timestep t
        'mask': image mask of t, i.e. x_t = x_M * mask
        '''
        data_path = os.path.join(self.data_dir, self.data_path_list[idx])
        with open(data_path, 'rb') as f:
            data = pkl.load(f) # dict, with 'captions', 'image' and 'mask'

        seq_len = data['mask'].shape[0]
        # sample seq_t from [0, seq_len]
        seq_t = np.random.randint(0, seq_len)

        jpg = data['image']
        
        mask = data['mask'][seq_t]

        # resize the jpg and mask to 512x512
        jpg = self.resize_image(jpg) # 512x512x3
        mask = self.resize_image(np.tile(mask, (3, 1, 1)).transpose(1,2,0))[..., [0]] # 512x512x1
        mask = mask.astype(np.float32)

        # compute hint, which is jpg * mask_{t-1}
        prev_mask = np.zeros_like(data['mask'][0])
        if seq_t > 0:
            prev_mask = data['mask'][seq_t-1]
        prev_mask = self.resize_image(np.tile(prev_mask, (3, 1, 1)).transpose(1,2,0))[..., [0]] # 512x512x1
        prev_mask = prev_mask.astype(np.float32)

        hint = prev_mask * jpg # 512x512x3

        # Normalize hint images to [0, 1].
        hint = hint.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        jpg = (jpg.astype(np.float32) / 127.5) - 1.0


        # crop the jpg, hint and mask to 512*512
        # 获取原始图片的宽度和高度
        height, width, _ = jpg.shape

        # 计算可截取的最大起始位置
        max_left = width - 512
        max_top = height - 512

        # 生成随机的截取起始位置
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)

        jpg = self.crop_img(jpg, left, top)
        hint = self.crop_img(hint, left, top)
        mask = self.crop_img(mask, left, top)

        # random pick a caption
        txt_t = np.random.randint(0, len(data['captions']))
        
        return {
            'seq_t': seq_t,
            'jpg': jpg,
            'txt': data['captions'][txt_t],
            'hint': hint,
            'mask': mask,
        }
    
    # check dataset img
    def show_img(self, idx):
        data = self.__getitem__(idx)
        
        ## TODO


class Segment_Hint5k_Dataset(Dataset):
    '''
    hint = mask segment from 0 to t
    jpg = raw image
    '''
    def __init__(self, 
                data_dir = './dataset/raw_data5k',
    ):
        self.data_dir = data_dir
        self.data_path_list = os.listdir(data_dir)
        

    def __len__(self):
        return len(self.data_path_list)


    def resize_image(self, image):
        assert image.shape[2] == 3

        H, W, _ = image.shape
        min_HW = min(H, W)

        target_size = int(H/min_HW * 512), int(W/min_HW * 512)
        target_size = max(512, target_size[0]), max(512, target_size[1])

        # 将NumPy数组转换为PIL图像
        pil_image = Image.fromarray(image.astype('uint8'))

        # 调整图像大小
        resized_image = pil_image.resize(target_size)

        # 将PIL图像转换回NumPy数组
        resized_np_image = np.array(resized_image)

        assert resized_np_image.shape[0] == 512 or resized_np_image.shape[1] == 512

        return resized_np_image

    def crop_img(self, array, left, top):
        # 截取512x512的图片区域
        cropped_array = array[top:top+512, left:left+512, ...]
        return cropped_array

    def create_seg(self, seq_t : int, mask_array : np.ndarray)->np.ndarray:
        '''
        mask_array is gt_mask[0:seq_t]
        '''

        mask_array = mask_array.astype(np.int32)
        h, w = mask_array.shape[1], mask_array.shape[2]

        seg = np.zeros((h, w))
        prev_mask = seg
        for i in range(seq_t):
            curr_mask = mask_array[i] - prev_mask
            prev_mask = mask_array[i]
            seg[curr_mask==1] = i+1
        
        return seg # (h, w)


    def __getitem__(self, idx):
        '''
        return a dict with keys: 
        'jpg': the target image, i.e., x_M
        'txt': txt description
        'hint': the hint image, i.e., x_{t-1}
        'seq_t': image seq timestep t
        'mask': image mask of t, i.e. x_t = x_M * mask
        '''
        data_path = os.path.join(self.data_dir, self.data_path_list[idx])
        with open(data_path, 'rb') as f:
            data = pkl.load(f) # dict, with 'captions', 'image' and 'mask'

        seq_len = data['mask'].shape[0]
        # sample seq_t from [0, seq_len]
        seq_t = np.random.randint(0, seq_len)

        jpg = data['image']
        
        mask = data['mask'][seq_t]

        seg = self.create_seg(seq_t, data['mask'][:seq_t])

        # resize the jpg and mask to 512x512
        jpg = self.resize_image(jpg) # 512x512x3
        mask = self.resize_image(np.tile(mask, (3, 1, 1)).transpose(1,2,0))[..., [0]] # 512x512x1
        mask = mask.astype(np.float32)

        seg = self.resize_image(np.tile(seg, (3, 1, 1)).transpose(1,2,0)) # 512x512x3

        hint = seg

        # Normalize hint images to [0, 1].
        hint = hint.astype(np.float32) / hint.max()
        # Normalize target images to [-1, 1].
        jpg = (jpg.astype(np.float32) / 127.5) - 1.0


        # crop the jpg, hint and mask to 512*512
        # 获取原始图片的宽度和高度
        height, width, _ = jpg.shape

        # 计算可截取的最大起始位置
        max_left = width - 512
        max_top = height - 512

        # 生成随机的截取起始位置
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)

        jpg = self.crop_img(jpg, left, top)
        hint = self.crop_img(hint, left, top)
        mask = self.crop_img(mask, left, top)

        # random pick a caption
        txt_t = np.random.randint(0, len(data['captions']))
        
        return {
            'seq_t': seq_t,
            'jpg': jpg,
            'txt': data['captions'][txt_t],
            'hint': hint,
            'mask': mask,
        }
    
    # check dataset img
    def show_img(self, idx):
        data = self.__getitem__(idx)
        
        ## TODO


if __name__ == "__main__":
    dataset = Segment_Hint5k_Dataset()
    dataset[0]