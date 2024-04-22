# !/bin/sh

source activate /root/miniconda3/envs/embedding

export http_proxy=100.66.27.20:3128
export https_proxy=100.66.27.20:3128

# # For Mistral Model
# MODEL_NAME=mistral_embedder48_sfr_eos1_prompt_indataset_matryoshka1_temp1_lr15_bs1500_ml512_step600_wl15
# CKPT_DIR=mistral_indataset48_20240402041835
# CKPT_NAME=mistral_indataset48_600.pt

# CUDA_VISIBLE_DEVICES=0 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=eos \
#                                         --which_layer=-15 \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

# MODEL_NAME=mistral_embedder48_sfr_eos1_prompt_indataset_matryoshka1_temp1_lr15_bs1500_ml512_step400_wl15
# CKPT_DIR=mistral_indataset48_20240402041835
# CKPT_NAME=mistral_indataset48_400.pt

# CUDA_VISIBLE_DEVICES=1 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=eos \
#                                         --which_layer=-15 \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &

# For BGE Model
MODEL_NAME=bge_embedder48_test_razor_1000
CKPT_DIR=bge_indataset48_adaptive_paired_prompt_20240327185931
CKPT_NAME=bge_indataset48_adaptive_paired_prompt_1000.pt

CUDA_VISIBLE_DEVICES=0 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
                                        --backbone_type=BGE \
                                        --init_backbone=BAAI/bge-base-en-v1.5 \
                                        --pool_type=cls \
                                        --mytryoshka_size=768 \
                                        --task_prompt \
                                        --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME &

MODEL_NAME=bge_embedder48_test_razor_1062
CKPT_DIR=bge_indataset48_adaptive_paired_prompt_20240327185931
CKPT_NAME=bge_indataset48_adaptive_paired_prompt_1062.pt

CUDA_VISIBLE_DEVICES=1 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
                                        --backbone_type=BGE \
                                        --init_backbone=BAAI/bge-base-en-v1.5 \
                                        --pool_type=cls \
                                        --mytryoshka_size=768 \
                                        --task_prompt \
                                        --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
                                        --embedder_name=$MODEL_NAME &


# # For InternLM Model

# pip install flash-attn==2.0.0.post1

# MODEL_NAME=internlm_embedder48_sfr_pw1_prompt_random_matryoshka1_temp1_lr15_bs1500_ml512_step1133_wl8_test
# CKPT_DIR=internlm_random48_adaptive_paired_prompt_20240402035923
# CKPT_NAME=internlm_random48_adaptive_paired_prompt_1133.pt

# CUDA_VISIBLE_DEVICES=0 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=InternLM \
#                                         --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
#                                         --pool_type=eos \
#                                         --which_layer=-8 \
#                                         --mytryoshka_size=2048 \
#                                         --flashatt \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &


# MODEL_NAME=internlm_embedder48_sfr_pw1_prompt_random_matryoshka1_temp1_lr15_bs1500_ml512_step800_wl8_test
# CKPT_DIR=internlm_random48_adaptive_paired_prompt_20240402035923
# CKPT_NAME=internlm_random48_adaptive_paired_prompt_800.pt

# CUDA_VISIBLE_DEVICES=1 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=InternLM \
#                                         --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
#                                         --pool_type=eos \
#                                         --which_layer=-8 \
#                                         --mytryoshka_size=2048 \
#                                         --flashatt \
#                                         --peft_lora \
#                                         --task_prompt \
#                                         --embedder_ckpt_path=/fs-computility/llm/chenzhi/ckpts/$CKPT_DIR/$CKPT_NAME \
#                                         --embedder_name=$MODEL_NAME &



# MODEL_NAME=internlm_layer_8

# CUDA_VISIBLE_DEVICES=0 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=InternLM \
#                                         --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
#                                         --pool_type=position_weight \
#                                         --which_layer=-8 \
#                                         --mytryoshka_size=2048 \
#                                         --task_prompt \
#                                         --embedder_name=$MODEL_NAME &

# MODEL_NAME=internlm_layer_vallina_8

# CUDA_VISIBLE_DEVICES=1 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=InternLM \
#                                         --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-1_8b \
#                                         --pool_type=position_weight \
#                                         --which_layer=-8 \
#                                         --mytryoshka_size=2048 \
#                                         --task_prompt \
#                                         --embedder_name=$MODEL_NAME &

# MODEL_NAME=mistral_layer_12

# CUDA_VISIBLE_DEVICES=0 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=position_weight \
#                                         --which_layer=-12 \
#                                         --task_prompt \
#                                         --embedder_name=$MODEL_NAME &

# MODEL_NAME=mistral_layer_1

# CUDA_VISIBLE_DEVICES=1 python /fs-computility/llm/chenzhi/InternEmbedding/run.py evaluate \
#                                         --backbone_type=Mistral \
#                                         --pool_type=position_weight \
#                                         --which_layer=-1 \
#                                         --task_prompt \
#                                         --embedder_name=$MODEL_NAME &

wait