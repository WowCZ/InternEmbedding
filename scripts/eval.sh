# !/bin/sh

# # For Mistral Model
# MODEL_NAME=mistral_embedder28_sfr_eos1_prompt_ins_matryoshka1_temp1_lr15_bs1000_ml512_600
# CKPT_DIR=mistral_sfr28_20240315035956
# CKPT_NAME=mistral_sfr28_600.pt

# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=eos \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

# MODEL_NAME=mistral_embedder28_sfr_eos1_prompt_ins_matryoshka1_temp1_lr15_bs600_ml512_800
# CKPT_DIR=mistral_sfr28_20240315051830
# CKPT_NAME=mistral_sfr28_800.pt
# CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=eos \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

# For BGE Model
MODEL_NAME=bge_embedder28_sfr_eos1_prompt_ins_matryoshka1_temp1_lr15_bs1024_ml512_1000_norm
CKPT_DIR=bge_sfr28_20240315063027
CKPT_NAME=bge_sfr28_1000.pt

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
                                        --backbone_type=BGE \
                                        --init_backbone=BAAI/bge-base-en-v1.5 \
                                        --pool_type=cls \
                                        --task_prompt \
                                        --embedding_norm \
                                        --mytryoshka_size=768 \
                                        --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME &

MODEL_NAME=bge_embedder28_sfr_eos1_prompt_ins_matryoshka1_temp1_lr15_bs1024_ml512_1447_norm
CKPT_DIR=bge_sfr28_20240315063027
CKPT_NAME=bge_sfr28_1447.pt

CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
                                        --backbone_type=BGE \
                                        --init_backbone=BAAI/bge-base-en-v1.5 \
                                        --pool_type=cls \
                                        --task_prompt \
                                        --embedding_norm \
                                        --mytryoshka_size=768 \
                                        --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME &

wait