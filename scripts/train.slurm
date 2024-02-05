#!/bin/sh
#SBATCH -J embedding
#SBATCH --error=log/slurm-%A_%a.err
#SBATCH --output=log/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

export OMP_NUM_THREADS=8
accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml run.py train