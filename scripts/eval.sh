# !/bin/sh
MODEL_NAME=mistral_embedder33_triple_eos1_noprompt_ins_matryoshka1_temp1_lr14_bs150_ml512_1000
CKPT_DIR=mistral_bge_triple33_limit100k_20240308135601
CKPT_NAME=mistral_bge_triple33_limit100k_1000.pt
CUDA_RANK=3

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --pool_type=eos --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME

# MODEL_NAME=mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs300_ml512_3000
# CKPT_NAME=mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs300_ml512_3000.pt
# CUDA_RANK=2

# CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_NAME --embedder_name=$MODEL_NAME
