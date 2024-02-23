# !/bin/sh
MODEL_NAME=mistral_embedder33_filter_eos2_prompt_ins_matryoshka9_temp2_lr24_bs300_ml512_2000
CKPT_DIR=mistral_filter33_20240221040715
CKPT_NAME=mistral_filter33_2000.pt
CUDA_RANK=3

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --pool_type=eos --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME

# MODEL_NAME=mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs300_ml512_3000
# CKPT_NAME=mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs300_ml512_3000.pt
# CUDA_RANK=2

# CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_NAME --embedder_name=$MODEL_NAME
