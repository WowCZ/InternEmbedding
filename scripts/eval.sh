# !/bin/sh

# MODEL_NAME=mistral_embedder33_triple_eos1_prompt_hns2_matryoshka1_temp1_lr15_bs1000_ml512_713
# CKPT_DIR=mistral_filter18_20240311143314
# CKPT_NAME=mistral_filter18_713.pt
# CUDA_RANK=0

# CUDA_VISIBLE_DEVICES=$CUDA_RANK /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=eos \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME

# MODEL_NAME=bge_embedder28_sfr_eos1_prompt_ins_matryoshka1_temp1_lr15_bs1000_ml512_400
# CKPT_DIR=bge_sfr28_20240315060351
MODEL_NAME=bge_base_v15_baseline
CKPT_DIR=test
CKPT_NAME=bge_sfr28_500.pt
CUDA_RANK=0

CUDA_VISIBLE_DEVICES=$CUDA_RANK /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
                                        --backbone_type=BGE \
                                        --init_backbone=BAAI/bge-base-en-v1.5 \
                                        --pool_type=cls \
                                        --task_prompt \
                                        --mytryoshka_size=768 \
                                        --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME