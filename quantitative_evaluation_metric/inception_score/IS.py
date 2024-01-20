from inception_score import inception_score
import os
import pdb
from PIL import Image
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='../output_control_ablation/18_18:03')
args = parser.parse_args()
img_path = args.img_path
imgs = []
for root, dirs, files in os.walk(img_path):
    for file in files:
        # transfer png to array
        # imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]]
        if file.endswith('out_512_000.png'):
            img = Image.open(os.path.join(root, file))
            img_arr=np.array(img)
            img_arr = img_arr.transpose(2,0,1)
            img_arr = img_arr/255.0
            img_arr = img_arr*2-1
            imgs.append(img_arr) 
mean,std=inception_score(imgs, cuda=False, batch_size=32, resize=True, splits=10)
print(mean,std)