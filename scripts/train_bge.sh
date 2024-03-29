#!/bin/sh
source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding

accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/bge_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --init_backbone=BAAI/bge-base-en-v1.5 \
                                           --pool_type=cls \
                                           --backbone_type=BGE \
                                           --task_prompt \
                                           --embedding_norm \
                                           --task_adaptation \
                                           --sampler=indataset \
                                           --hard_negative_sampling \
                                           --hard_negative_num=1 \
                                           --warmup_rate=0.1 \
                                           --clip_gradient \
                                           --checkpoint_batch_size=200 \
                                           --gradcache_chunk_size=200 \
                                           --temperature=0.015 \
                                           --learning_rate=1e-5 \
                                           --matryoshka_adaptive_dims=768 \
                                           --mytryoshka_size=768 \
                                           --batch_size_per_gpu=1600 \
                                           --dataset_config=/fs-computility/llm/chenzhi/InternEmbedding/configs/dataset_configs/datasets_test.yaml \
                                           --embedder_name=bge_indataset48_adaptive_paired_prompt &
wait