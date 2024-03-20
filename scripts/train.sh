#!/bin/sh

/root/miniconda3/envs/embedding/bin/accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/mistral_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --pool_type=eos \
                                           --backbone_type=Mistral \
                                           --task_prompt \
                                           --peft_lora \
                                           --embedding_norm \
                                           --sampler=random \
                                           --warmup_rate=0.1 \
                                           --checkpoint_batch_size=10 \
                                           --gradcache_chunk_size=10 \
                                           --temperature=0.01 \
                                           --learning_rate=3e-5 \
                                           --batch_size_per_gpu=600 \
                                           --embedder_name=mistral_indataset_sfr35