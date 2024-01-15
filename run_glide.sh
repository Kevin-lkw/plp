#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=glide
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 3-00:00:00

python inpaint.py
