'''

Transform the data.pkl which contains 5k images and corresponding captions and masks.

Each item in data.pkl is a dict with keys: 'captions' : list[str], 'images' : Tensor(h, w, c), 'mask' : Tensor(seq_len, h, w)

Run the following codes to get preprocessed dataset in ./dataset/raw_data5k/
```
cd plp
python pre_processing/dataset_process.py
```

'''

import os
import pickle as pkl
from tqdm.auto import tqdm
import pdb


def preprocess_raw_data5k():
    '''
    preprocess data.pkl to ./dataset/raw_data5k/ directory, where each dir in it contains 'captions', 'images', 'mask'

    '''
    save_dir = './dataset/raw_data5k'
    load_path = './dataset/data.pkl'

    os.makedirs(save_dir, exist_ok=True)

    # load data.pkl
    with open(load_path, 'rb') as f:
        raw_data = pkl.load(f)
    # pdb.set_trace()

    # save each item to save_dir
    for k in tqdm(raw_data.keys()):
        assert isinstance(k, int)
        save_path = os.path.join(save_dir, str(k))

        with open(save_path, 'wb') as f:
            pkl.dump(raw_data[k], f)




if __name__ == "__main__":
    preprocess_raw_data5k()



