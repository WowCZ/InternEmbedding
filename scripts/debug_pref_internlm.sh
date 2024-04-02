#!/bin/sh
source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding
# CODE=/fs-computility/llm/wangyikun/workspace/TrainPrefModel/run.py
CODE=/fs-computility/llm/shared/wangyikun/code/TrainPrefModel/run.py

accelerate launch --config_file /fs-computility/llm/shared/wangyikun/code/TrainPrefModel/scripts/debug_pref_internlm.yaml \
                                           $CODE train \
                                           --pool_type=eos \
                                           --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
                                           --backbone_type=InternLM \
                                           --peft_lora \
                                           --embedding_norm \
                                           --task_adaptation \
                                           --sampler=random \
                                           --warmup_rate=0.1 \
                                           --num_epochs 5 \
                                           --checkpoint_batch_size=10 \
                                           --gradcache_chunk_size=10 \
                                           --temperature=1. \
                                           --learning_rate=1e-5 \
                                           --save_ckpt_steps=200 \
                                           --batch_size_per_gpu=60 \
                                           --dataset_config=/fs-computility/llm/shared/wangyikun/code/TrainPrefModel/configs/dataset_configs/pref_datasets.yaml \
                                           --embedder_name=pref_internlm

# accelerate launch --config_file /fs-computility/llm/wangyikun/workspace/TrainPrefModel/scripts/debug_pref_internlm.yaml \
#                                            /fs-computility/llm/wangyikun/workspace/TrainPrefModel/run.py train \
#                                            --pool_type=eos \
#                                            --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
#                                            --backbone_type=InternLM \
#                                            --peft_lora \
#                                            --embedding_norm \
#                                            --task_adaptation \
#                                            --sampler=random \
#                                            --warmup_rate=0.1 \
#                                            --checkpoint_batch_size=10 \
#                                            --gradcache_chunk_size=10 \
#                                            --temperature=1. \
#                                            --learning_rate=1e-5 \
#                                            --save_ckpt_steps=200 \
#                                            --batch_size_per_gpu=600 \
#                                            --dataset_config=/fs-computility/llm/chenzhi/InternEmbedding/configs/dataset_configs/datasets_test.yaml \
#                                            --embedder_name=pref_internlm
