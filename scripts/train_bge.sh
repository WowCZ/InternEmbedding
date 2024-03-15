#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1 /root/miniconda3/envs/embedding/bin/accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/bge_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --init_backbone=BAAI/bge-base-zh-v1.5 \
                                           --pool_type=cls \
                                           --backbone_type=BGE \
                                           --checkpoint_batch_size=128 \
                                           --gradcache_chunk_size=-1 \
                                           --temperature=0.01 \
                                           --learning_rate=1e-4 \
                                           --hard_negative_sampling \
                                           --hard_negative_num=2 \
                                           --matryoshka_adaptive_dims=768 \
                                           --mytryoshka_size=768 \
                                           --batch_size_per_gpu=256 \
                                           --embedder_name=bge_keypoint_triple5