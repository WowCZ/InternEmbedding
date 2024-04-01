#!/bin/sh

/fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding/bin/accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/internlm_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
                                           --pool_type=eos \
                                           --backbone_type=InternLM \
                                           --which_layer=-8 \
                                           --task_prompt \
                                           --peft_lora \
                                           --embedding_norm \
                                           --sampler=random \
                                           --warmup_rate=0.1 \
                                           --clip_gradient \
                                           --checkpoint_batch_size=50 \
                                           --gradcache_chunk_size=50 \
                                           --temperature=0.01 \
                                           --learning_rate=1e-5 \
                                           --save_ckpt_steps=200 \
                                           --matryoshka_adaptive_dims=2048 \
                                           --mytryoshka_size=2048 \
                                           --batch_size_per_gpu=1500 \
                                           --wandb_project_name=InternEmbedder \
                                           --dataset_config=/fs-computility/llm/chenzhi/InternEmbedding/configs/dataset_configs/datasets_test.yaml \
                                           --embedder_name=internlm_random48_adaptive_paired_prompt &
wait