#!/bin/sh
#SBATCH -J embedding
#SBATCH --error=log/slurm-%A_%a.err
#SBATCH --output=log/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --partition=llm_t
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

python eval.py