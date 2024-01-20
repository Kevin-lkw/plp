import argparse
import joblib
import pickle as pkl
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--caption_path', type=str, default='captions_val2017.pkl')
    parser.add_argument('-i', '--image_mask_path', type=str, default='images_masks_val2017.pkl')
    args = parser.parse_args()
    return args


def merge_data(
        caption_path,
        image_mask_path
):
    with open(caption_path, 'rb') as f:
        captions = joblib.load(f)

    with open(image_mask_path, 'rb') as f:
        image_masks = joblib.load(f)

    image_ids = list(image_masks.keys())

    data = {}
    for image_id in tqdm(image_ids):
        data[image_id] = {}
        data[image_id]['captions'] = captions[image_id]
        data[image_id]['image'] = image_masks[image_id]['image']
        data[image_id]['mask'] = image_masks[image_id]['mask']
        data[image_id]['mask_colors'] = image_masks[image_id]['mask_colors']

    output_path = 'data.pkl'

    with open(output_path, 'wb') as f:
        joblib.dump(data, f)


if __name__ == '__main__':
    args = parse_args()
    merge_data(
        caption_path=args.caption_path,
        image_mask_path=args.image_mask_path
    )
