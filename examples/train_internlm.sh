#!/bin/sh
source activate /root/miniconda3/envs/embedding

export http_proxy=100.66.27.20:3128
export https_proxy=100.66.27.20:3128

pip install flash-attn==2.0.0.post1

accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/internlm_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --init_backbone=/fs-computility/llm/shared/yangyf/share/PrLM/internlm2-chat-7b \
                                           --pool_type=eos \
                                           --backbone_type=InternLM \
                                           --which_layer=-1 \
                                           --task_prompt \
                                           --peft_lora \
                                           --embedding_norm \
                                           --task_adaptation \
                                           --sampler=indataset \
                                           --hard_negative_sampling \
                                           --hard_negative_num=1 \
                                           --warmup_rate=0.1 \
                                           --clip_gradient \
                                           --checkpoint_batch_size=14 \
                                           --gradcache_chunk_size=14 \
                                           --temperature=0.01 \
                                           --learning_rate=1e-5 \
                                           --save_ckpt_steps=200 \
                                           --matryoshka_adaptive_dims=4096 \
                                           --mytryoshka_size=4096 \
                                           --batch_size_per_gpu=1400 \
                                           --wandb_project_name=InternEmbedder \
                                           --dataset_config=/fs-computility/llm/chenzhi/InternEmbedding/configs/dataset_configs/datasets_test.yaml \
                                           --embedder_name=internlm_random48_adaptive_paired_prompt &
wait