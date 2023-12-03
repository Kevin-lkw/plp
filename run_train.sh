#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=control
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 2-00:00:00

python train_plp.py
