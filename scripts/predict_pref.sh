#!/bin/sh
INIT_MODEL=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft
MODEL_NAME=pref_internlm
CKPT_PATH=/fs-computility/llm/wangyikun/workspace/ckpts/pref_internlm_v1.0/pref_internlm_600.pt
DATASET_PATH=/fs-computility/llm/shared/wangyikun/dump/ad_preference_0326_segmented/AD_Preference/test.jsonl
CUDA_RANK=1

CUDA_VISIBLE_DEVICES=$CUDA_RANK python run.py predict \
    --init_backbone=${INIT_MODEL} \
    --embedder_name=$MODEL_NAME \
    --embedder_ckpt_path=${CKPT_PATH} \
    --dataset_path=${DATASET_PATH} --backbone_type InternLM --peft_lora

# Path: TrainPrefModel/scripts/train_pref.sh