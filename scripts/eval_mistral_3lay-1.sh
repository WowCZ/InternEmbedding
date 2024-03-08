# !/bin/sh
MODEL_NAME=mistral_3lay-1
CKPT_DIR=/fs-computility/llm/shared/yangyf/Embedding_pruning/PrLM
CKPT_NAME=mistral_3lay-1
CUDA_RANK=1

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --pool_type=position_weight --init_backbone=$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME
