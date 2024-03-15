# !/bin/sh
MODEL_NAME=mistral_embedder33_triple_eos1_prompt_hns2_matryoshka1_temp1_lr15_bs1000_ml512_500_norm
CKPT_DIR=mistral_filter18_20240311143314
CKPT_NAME=mistral_filter18_500.pt
CUDA_RANK=0

CUDA_VISIBLE_DEVICES=$CUDA_RANK /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
                                        --backbone_type=Mistral \
                                        --pool_type=eos \
                                        --peft_lora \
                                        --task_prompt \
                                        --embedding_norm \
                                        --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME