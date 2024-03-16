#!/bin/sh

/root/miniconda3/envs/embedding/bin/accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/bge_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --init_backbone=BAAI/bge-base-en-v1.5 \
                                           --pool_type=cls \
                                           --backbone_type=BGE \
                                           --task_prompt \
                                           --warmup_rate=0.1 \
                                           --checkpoint_batch_size=156 \
                                           --gradcache_chunk_size=156 \
                                           --temperature=0.01 \
                                           --learning_rate=1e-5 \
                                           --matryoshka_adaptive_dims=768 \
                                           --mytryoshka_size=768 \
                                           --batch_size_per_gpu=1560 \
                                           --embedder_name=bge_sfr49