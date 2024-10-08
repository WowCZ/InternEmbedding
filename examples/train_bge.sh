#!/bin/sh
source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding

export http_proxy=100.66.27.20:3128
export https_proxy=100.66.27.20:3128

accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/bge_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --init_backbone=BAAI/bge-base-zh-v1.5 \
                                           --pool_type=cls \
                                           --backbone_type=BGE \
                                           --embedding_norm \
                                           --sampler=random \
                                           --warmup_rate=0.1 \
                                           --clip_gradient \
                                           --checkpoint_batch_size=200 \
                                           --gradcache_chunk_size=200 \
                                           --temperature=0.01 \
                                           --learning_rate=1e-5 \
                                           --matryoshka_adaptive_dims=768 \
                                           --mytryoshka_size=768 \
                                           --batch_size_per_gpu=600 \
                                           --wandb_project_name=BGEEmbedder \
                                           --dataset_config=/fs-computility/llm/chenzhi/InternEmbedding/configs/dataset_configs/gaokao_xes.yaml \
                                           --embedder_name=bge_gaokao_xes_kp &
wait