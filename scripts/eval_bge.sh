# !/bin/sh
MODEL_NAME=bge-base-en-v1.5
CKPT_DIR=/fs-computility/llm/shared/yangyf/Embedding_pruning/PrLM
CKPT_NAME=bge-base-en-v1.5
CUDA_RANK=0
# export TORCH_USE_CUDA_DSA=1
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --pool_type=position_weight --backbone_type=Phi --init_backbone=$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME
CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --pool_type=position_weight --mytryoshka_size=512 --backbone_type=Phi --init_backbone=$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME