#!/bin/sh
MODEL_NAME=mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp1_lr24_bs600_ml512_1500
CKPT_DIR=mistral_embedder_20240205124927
CKPT_NAME=mistral_embedder_1500.pt
CUDA_RANK=0

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME