import os
import json
import argparse
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--caption_path', type=str, default='annotations/captions_val2017.json')
    args = parser.parse_args()
    return args


def read_caption_data(
        caption_path
):
    with open(caption_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']

    captions = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        caption = annotation['caption']
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)

    output_path = os.path.splitext(caption_path)[0] + '.pkl'

    with open(output_path, 'wb') as f:
        pkl.dump(captions, f)


if __name__ == '__main__':
    args = parse_args()
    read_caption_data(
        caption_path=args.caption_path
    )
