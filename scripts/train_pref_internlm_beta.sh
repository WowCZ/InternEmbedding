#!/bin/sh
source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding
CODE=/fs-computility/llm/shared/wangyikun/code/TrainPrefModel/run.py

accelerate launch --config_file /fs-computility/llm/shared/wangyikun/code/TrainPrefModel/scripts/train_pref_internlm.yaml \
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
                                           --max_length 2048 \
                                           --checkpoint_batch_size=10 \
                                           --gradcache_chunk_size=10 \
                                           --temperature=1. \
                                           --learning_rate=1e-5 \
                                           --save_ckpt_steps=200 \
                                           --training_loss=logit_margin_loss \
                                           --batch_size_per_gpu=4 \
                                           --dataset_config=configs/dataset_configs/pref_datasets_prompt.yaml \
                                           --embedder_name=pref_internlm_prompt
