#!/bin/sh

# For BGE Model
MODEL_NAME=bge_embedder48_example
CKPT_DIR=bge_indataset48_adaptive_paired_prompt_20240327185931
CKPT_NAME=bge_indataset48_adaptive_paired_prompt_1000.pt

CUDA_VISIBLE_DEVICES=0 python /fs-computility/llm/chenzhi/AwesomeICL/InternEmbedding/run.py predict \
                                        --backbone_type=BGE \
                                        --init_backbone=BAAI/bge-base-en-v1.5 \
                                        --pool_type=cls \
                                        --mytryoshka_size=768 \
                                        --task_prompt \
                                        --embedder_ckpt_path=/fs-computility/llm/shared/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME
