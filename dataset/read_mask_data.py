import os
import argparse
import joblib
import numpy as np
import pickle as pkl
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, default='val2017')
    parser.add_argument('-m', '--mask_path', type=str, default='panoptic_val2017')
    args = parser.parse_args()
    return args


def read_mask_data(
        image_path,
        mask_path,
        threshold=2048,
):
    image_list = sorted(os.listdir(image_path))

    mask_all = {}

    for image_name in tqdm(image_list):
        # if int(image_name[:-4]) != 11615:
        #     continue

        mask_name = image_name.replace('.jpg', '.png')

        image = Image.open(os.path.join(image_path, image_name))
        mask = Image.open(os.path.join(mask_path, mask_name))
        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        # if len(image.shape) == 2:
        #     image = np.stack([image, image, image], axis=-1)
        #     print(image_name)
        #     image = Image.fromarray(image)
        #     image.save(os.path.join(image_path, image_name))

        all_mask_colors = [tuple(c) for c in mask.reshape((-1, 3)).tolist()]
        all_mask_colors = list(set(all_mask_colors))

        mask_color_masks = []
        mask_color_areas = []
        for mask_color in all_mask_colors:
            mask_color_mask = (mask[..., 0] == mask_color[0]) * (mask[..., 1] == mask_color[1]) * (mask[..., 2] == mask_color[2])
            mask_color_mask_ = np.zeros_like(mask_color_mask, dtype=np.uint8)
            mask_color_mask_[mask_color_mask] = 1
            mask_color_area = mask_color_mask_.sum()

            mask_color_masks.append(mask_color_mask_)
            mask_color_areas.append(mask_color_area)

        # assert sum(mask_color_areas) == mask.shape[0] * mask.shape[1]

        all_mask_colors = np.array(all_mask_colors)
        mask_color_masks = np.array(mask_color_masks)
        mask_color_areas = np.array(mask_color_areas)

        order = np.argsort(-mask_color_areas)
        all_mask_colors = all_mask_colors[order]
        mask_color_masks = mask_color_masks[order]
        mask_color_areas = mask_color_areas[order]

        final_mask = []
        early_stop = False
        for i in range(all_mask_colors.shape[0]):
            if i > 0:
                mask_color_masks[i] = mask_color_masks[i - 1] + mask_color_masks[i]
            if mask_color_areas[i] <= threshold:
                mask_color_masks[i] = 1
                early_stop = True
            final_mask.append(mask_color_masks[i])
            if early_stop:
                break

        final_mask = [np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)] + final_mask
        final_mask = np.array(final_mask).astype(np.bool_)

        # _image = np.zeros_like(image, dtype=np.uint8)
        # for i in range(1, final_mask.shape[0]):
        #     _image += np.expand_dims(final_mask[i].astype(np.uint8) - final_mask[i - 1].astype(np.uint8), axis=-1) * all_mask_colors[i - 1].astype(np.uint8)
        #     image_i = _image.copy()
        #     image_i = Image.fromarray(image_i)
        #     image_i.show()

        image_id = int(image_name[:-4])
        mask_all[image_id] = {
            'image': image,
            'mask': final_mask,
            'mask_colors': all_mask_colors,
        }

    output_path = 'masks.pkl'
    with open(output_path, 'wb') as f:
        joblib.dump(mask_all, f)


if __name__ == '__main__':
    args = parse_args()
    read_mask_data(
        image_path=args.image_path,
        mask_path=args.mask_path
    )
