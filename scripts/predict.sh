#!/bin/sh
MODEL_NAME=mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs300_ml512_1500
CUDA_RANK=1

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py predict --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$MODEL_NAME.pt --embedder_name=$MODEL_NAME