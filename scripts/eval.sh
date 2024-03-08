# !/bin/sh
MODEL_NAME=mistral_embedder8_triple_eos1_prompt_ins_matryoshka1_temp1_lr14_bs150_ml512_226
CKPT_DIR=mistral_bge_triple8_20240307161437
CKPT_NAME=mistral_bge_triple8_226.pt
CUDA_RANK=0

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --pool_type=position_weight --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME

# MODEL_NAME=mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs300_ml512_3000
# CKPT_NAME=mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs300_ml512_3000.pt
# CUDA_RANK=2

# CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_NAME --embedder_name=$MODEL_NAME
