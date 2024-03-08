# !/bin/sh
MODEL_NAME=mistral_16lay
CKPT_DIR=/fs-computility/llm/shared/yangyf/Embedding_pruning/PrLM
CKPT_NAME=mistral_16lay
CUDA_RANK=3

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --pool_type=position_weight --init_backbone=$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME
