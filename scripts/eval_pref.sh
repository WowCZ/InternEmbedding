# !/bin/sh

source activate /fs-computility/llm/shared/chenzhi/miniconda3/envs/embedding

# For Mistral Model
MODEL_NAME=pref_internlm
CKPT_DIR=/fs-computility/llm/wangyikun/workspace/ckpts/pref_internlm_20240401081259

# for step in $1 # 200 400 600 800 1000 1200 1600 2200 
# do
#     CKPT_NAME=pref_internlm_${step}.pt
#     CUDA_VISIBLE_DEVICES=$2 python /fs-computility/llm/shared/wangyikun/code/TrainPrefModel/run.py evaluate \
#                                             --backbone_type=InternLM \
#                                             --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
#                                             --pool_type=eos \
#                                             --dataset_config=/fs-computility/llm/shared/wangyikun/code/TrainPrefModel/configs/dataset_configs/pref_datasets.yaml \
#                                             --result_dir /fs-computility/llm/shared/wangyikun/dump/ad_preference_0326_segmented/AD_Preference/results \
#                                             --which_layer=-1 \
#                                             --batch_size_per_gpu 20 \
#                                             --peft_lora \
#                                             --task_prompt \
#                                             --embedder_ckpt_path=$CKPT_DIR/$CKPT_NAME \
#                                             --dev_mode 1 \
#                                             --embedder_name=$MODEL_NAME &> run_eval_step_${step}.log
# done

step=600
CKPT_NAME=pref_internlm_${step}.pt
CODEBASE=/fs-computility/llm/shared/wangyikun/code/TrainPrefModel
CUDA_VISIBLE_DEVICES=1 python ${CODEBASE}/run.py evaluate \
                                        --backbone_type=InternLM \
                                        --init_backbone=/fs-computility/llm/shared/yangyf/share/internlm2-chat-1_8b-sft \
                                        --pool_type=eos \
                                        --dataset_config=${CODEBASE}/configs/dataset_configs/pref_datasets.yaml \
                                        --result_dir /fs-computility/llm/shared/wangyikun/dump/ad_preference_0326_segmented/AD_Preference/results \
                                        --which_layer=-1 \
                                        --batch_size_per_gpu 20 \
                                        --peft_lora \
                                        --task_prompt \
                                        --embedder_ckpt_path=$CKPT_DIR/$CKPT_NAME \
                                        --dev_mode 0 \
                                        --extract_pseduolabel_0326 0 \
                                        --embedder_name=$MODEL_NAME &> ${CODEBASE}/logs/run_eval_step_${step}_gpt_test.log
wait
