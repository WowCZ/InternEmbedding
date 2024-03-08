# !/bin/sh
MODEL_NAME=mistral_test
CKPT_DIR=/fs-computility/llm/chenzhi/huggingface_cache/
CKPT_NAME=models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e
CUDA_RANK=0

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py evaluate --pool_type=position_weight --init_backbone=$CKPT_DIR/$CKPT_NAME --embedder_name=$MODEL_NAME
