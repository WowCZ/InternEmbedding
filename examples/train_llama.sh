#!/bin/sh
source activate /root/miniconda3/envs/embedding

export http_proxy=100.66.27.20:3128
export https_proxy=100.66.27.20:3128

pip install flash-attn==2.0.0.post1

accelerate launch --config_file /fs-computility/llm/chenzhi/InternEmbedding/configs/llama_accelerate_config.yaml \
                                           /fs-computility/llm/chenzhi/InternEmbedding/run.py train \
                                           --init_backbone=/fs-computility/llm/shared/llm_llama3_hf/Meta-Llama-3-8B-Instruct \
                                           --pool_type=eos \
                                           --backbone_type=Llama \
                                           --which_layer=-1 \
                                           --task_prompt \
                                           --peft_lora \
                                           --sampler=random \
                                           --warmup_rate=0.1 \
                                           --clip_gradient \
                                           --checkpoint_batch_size=8 \
                                           --gradcache_chunk_size=-1 \
                                           --temperature=0.01 \
                                           --learning_rate=5e-6 \
                                           --save_ckpt_steps=200 \
                                           --matryoshka_adaptive_dims=4096 \
                                           --mytryoshka_size=4096 \
                                           --batch_size_per_gpu=800 \
                                           --wandb_project_name=LlamaEmbedder \
                                           --dataset_config=/fs-computility/llm/chenzhi/InternEmbedding/configs/dataset_configs/datasets_test.yaml \
                                           --embedder_name=llama_random48_adaptive_paired_prompt &
wait