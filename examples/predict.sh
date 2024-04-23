#!/bin/sh

source activate /root/miniconda3/envs/embedding

export http_proxy=100.66.27.20:3128
export https_proxy=100.66.27.20:3128

# For BGE Model
MODEL_NAME=bge_embedder48_example
CKPT_DIR=bge_indataset48_adaptive_paired_prompt_20240327185931
CKPT_NAME=bge_indataset48_adaptive_paired_prompt_1000.pt

CUDA_VISIBLE_DEVICES=0 python /fs-computility/llm/chenzhi/InternEmbedding/run.py predict \
                                        --backbone_type=BGE \
                                        --init_backbone=BAAI/bge-base-en-v1.5 \
                                        --pool_type=cls \
                                        --mytryoshka_size=768 \
                                        --task_prompt \
                                        --embedder_ckpt_path=/fs-computility/llm/shared/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME &

MODEL_NAME=mistral_embedder18_example
CKPT_DIR=mistral_filter18_20240303141730
CKPT_NAME=mistral_filter18_2000.pt

CUDA_VISIBLE_DEVICES=1 python /fs-computility/llm/chenzhi/InternEmbedding/run.py predict \
                                        --backbone_type=Mistral \
                                        --pool_type=eos \
                                        --which_layer=-1 \
                                        --peft_lora \
                                        --task_prompt \
                                        --mytryoshka_size=4096 \
                                        --embedder_ckpt_path=/fs-computility/llm/shared/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME &

wait