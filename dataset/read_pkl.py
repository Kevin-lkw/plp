import os
import joblib
import numpy as np
import pickle as pkl
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=1145)
    args = parser.parse_args()
    return args


def read_data(
        data_path,
        index=0,
):
    with open(data_path, 'rb') as f:
        data = joblib.load(f)

    image_ids = list(data.keys())

    image_id = image_ids[index]

    captions = data[image_id]['captions']
    image = data[image_id]['image'].astype(np.uint8)
    mask = data[image_id]['mask']
    mask_colors = data[image_id]['mask_colors']

    for caption in captions:
        print(caption)
    print('')

    os.makedirs('tmp', exist_ok=True)

    # for i in range(mask.shape[0]):
    #     image_i = image * np.expand_dims(mask[i], axis=-1)
    #     image_i = Image.fromarray(image_i)
    #     image_i.save(f'tmp/{image_id}_{i}.png')

    _image = np.zeros_like(image, dtype=np.uint8)
    for i in range(1, mask.shape[0]):
        _image += np.expand_dims(mask[i].astype(np.uint8) - mask[i - 1].astype(np.uint8), axis=-1) * mask_colors[i - 1].astype(np.uint8)
        image_i = _image.copy()
        image_i = Image.fromarray(image_i)
        image_i.save(f'tmp/{image_id}_{i}.png')

    image = Image.fromarray(image)
    image.save(f'tmp/{image_id}.png')


if __name__ == '__main__':
    args = parse_args()
    read_data(
        data_path='data.pkl',
        index=args.idx
    )
