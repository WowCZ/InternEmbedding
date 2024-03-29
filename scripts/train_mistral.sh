#!/bin/sh
source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding

accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/mistral_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --pool_type=eos \
                                           --backbone_type=Mistral \
                                           --which_layer=-12 \
                                           --task_prompt \
                                           --peft_lora \
                                           --embedding_norm \
                                           --task_adaptation \
                                           --sampler=indataset \
                                           --hard_negative_sampling \
                                           --hard_negative_num=1 \
                                           --warmup_rate=0.1 \
                                           --checkpoint_batch_size=10 \
                                           --gradcache_chunk_size=10 \
                                           --temperature=0.01 \
                                           --learning_rate=1e-5 \
                                           --save_ckpt_steps=200 \
                                           --batch_size_per_gpu=600 \
                                           --dataset_config=/fs-computility/llm/chenzhi/InternEmbedding/configs/dataset_configs/datasets_test.yaml \
                                           --embedder_name=mistral_indataset48