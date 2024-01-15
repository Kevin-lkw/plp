import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import re
import numpy as np

'''
This script is used to put the training log images together and show them, 
including text prompt, hint images, ground truth images, sampled images.
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", type=str, required=True, help="global step")
    args = parser.parse_args()
    image_log_dir = 'image_log/train'
    global_step = args.gs


    filename_list = os.listdir(image_log_dir)
    filename_list = list(filter(lambda x: re.search(f'conditioning_gs-{global_step}', x) is not None, filename_list))
    assert len(filename_list) == 1

    text_filename = filename_list[0]
    hint_filename = text_filename.replace('conditioning', 'control')
    gt_filename = text_filename.replace('conditioning', 'reconstruction')
    sample_filename = text_filename.replace('conditioning', 'samples_cfg_scale_9.00')
    mask_pred_filename = text_filename.replace('conditioning', 'mask_pred')
    mask_gt_filename = text_filename.replace('conditioning', 'mask_gt')

    text = Image.open(os.path.join(image_log_dir, text_filename))
    hint = Image.open(os.path.join(image_log_dir, hint_filename))
    gt = Image.open(os.path.join(image_log_dir, gt_filename))
    sample = Image.open(os.path.join(image_log_dir, sample_filename))
    mask_pred = Image.open(os.path.join(image_log_dir, mask_pred_filename))
    mask_gt = Image.open(os.path.join(image_log_dir, mask_gt_filename))

    array1 = np.array(text)
    array2 = np.array(gt)
    array3 = np.array(hint)
    array4 = np.array(sample)
    array5 = np.array(mask_pred)[..., :3]
    array6 = np.array(mask_gt)[..., :3]
    # breakpoint()

    # 创建一个新的空图像，用于拼接图像
    height = 3096
    width = 2058
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # 将图像拼接到新图像中
    result[:516, :2058] = array1
    result[516:1032, :2058] = array2
    result[1032:1548, :2058] = array3
    result[1548:2064, :2058] = array4
    result[2068:2580, :512] = array5
    result[2584:, :512] = array6

    # 将NumPy数组转换为PIL图像
    result_image = Image.fromarray(result)

    # 保存拼接后的图像为PNG格式
    result_image.save(f'./validation_utils/mask_gs-{global_step}.png')
    print(f"image saved in './validation_utils/mask_gs-{global_step}.png'")

