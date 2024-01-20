# PLP

# Installation

## ControlNet

We use python=3.8, pytorch=2.1.0 and pytorch-lightning=1.9.0

Please install those packages first, then install the rest packages in `environment.yaml`.

You can also refer to the [installation guidance](https://github.com/lllyasviel/ControlNet) of official ControlNet repo to set up the environment.

# Inference with PLP framework

To inference using PLP+ControlNet, run
```bash
python control_infer.py --model_path path/to/your/ckpt --output_path path/to/output/folder
```

To inference using PLP+GLIDE, run
```bash
python inpaint.py --model_path path/to/your/ckpt --output_path path/to/output/folder
```

`path/to/your/ckpt` is the path where you save your trained ControlNet weights. We have provided our ckpts in `./ckpts` folder.

Note that when inference with GLIGE, the ckpt is used only for mask prediction.


# Training ControlNet

1、Download SDv1.5 weights ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and place it under `./models`.

2、If you have trouble in getting access to huggingfaceHub, download the openai/CLIP model weights manually and place it under `./openai/clip-vit-large-patch14`. The downloading link is [here](https://huggingface.co/openai/clip-vit-large-patch14/tree/main).

3、Download the dataset we preprocessed from PKU Drive. Place it under `./dataset`. Then run `python pre_processing/dataset_process.py` to get the preprocessed dataset in `./dataset/raw_data5k/`.

4、Run `python seq_add_control.py ./models/v1-5-pruned.ckpt ./models/plp_ini.ckpt` to get initialized PLP Model from SD weights.

5、Run `python train_plp.py` to start training.

The training logs including inferenced images and loss curves should be in `./image_log` and `./lightning_logs`.

You may refer to the [Official Finetuning Guidance](https://civitai.com/articles/2078/play-in-control-controlnet-training-setup-guide)

# Quantitative Evaluation
See [quantitative_evaluation_metric/README.md](quantitative_evaluation_metric/README.md) for more details.
