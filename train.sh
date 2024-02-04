#!/bin/sh

export OMP_NUM_THREADS=8
accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml train.py