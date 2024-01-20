# Quantitative Evaluation

We use FID, IS, and Clip-Scoring to evaluate the quality of the generated images.

## Result File Format

We suppose the generated images are organized in the following format:

```bash
path/to/generated_images
├── 000000.png
├── 000001.png
├── 000002.png
path/to/gound_truth_images
├── 000000.png
├── 000001.png
├── 000002.png
path/to/prompt
├── 000000.txt
├── 000001.txt
├── 000002.txt
```

Notice that the file name of the images and the corresponding prompt should be the same, otherwise the evaluation of Clip Score will fail.

However, when calulating the FID score, we only need the generated images and the ground truth images, and the file name of the images can be different.

## FID

We use the [pytorch-fid](https://github.com/mseitzer/pytorch-fid/tree/0a754fb8e66021700478fd365b79c2eaa316e31b) to calculate the FID score.

For more details, please refer to the [README](./fid/README.md) in the `fid` folder.

### environment setup

```bash
pip install pytorch-fid
```

### Usage

Calculate the FID score between two datasets, where images of each dataset are contained in an individual folder:

```bash
python -m pytorch_fid path/to/dataset1 path/to/dataset2
```

## Inception Score

We use the [inception-score-pytorch](https://github.com/sbarratt/inception-score-pytorch) to calculate the Inception Score.

For more details, please refer to the [README](./inception_score/README.md) in the `inception_score` folder.

### environment setup

```bash
pip install torch torchvision numpy 
```
### usage

```bash
python inception_score/IS.py --img_path path/to/generated_images
```

## Clip Score

We use the [clip-score-pytorch](https://github.com/Taited/clip-score) to calculate the Clip Score.

For more details, please refer to the [README](./clip_score/README.md) in the `clip_score` folder.

### environment setup

Requirements:
- Install PyTorch:
  ```
  pip install torch  # Choose a version that suits your GPU
  ```
- Install CLIP:
  ```
  pip install git+https://github.com/openai/CLIP.git
  ```
- Install clip-score from [PyPI](https://pypi.org/project/clip-score/):
  ```
  pip install clip-score
  ```

### Usage

To compute the CLIP score between images and texts, make sure that the image and text data are contained in two separate folders, and each sample has the same name in both modalities. Run the following command:

```
python -m clip_score path/to/image path/to/text
```