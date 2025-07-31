#!/bin/bash

# 设置日志文件路径
LOG_FILE="TCM4_addthinkprunedata_v8_global_step_108_answer_prompt_SFT/train_log.txt"


# 函数：记录命令执行
log_command() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Executing: $1" | tee -a "$LOG_FILE"
}

# 函数：记录命令执行结果
log_result() {
    if [ $? -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Success: $1" | tee -a "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed: $1" | tee -a "$LOG_FILE"
        exit 1
    fi
}


# 开始日志记录
echo "Training log started at $(date '+%Y-%m-%d %H:%M:%S')" | tee "$LOG_FILE"

# 设置CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
export special_token_loss=F




###############! long_short-gsm_long_short


# # 设置PE_MODE
log_command "export PE_MODE=default"
export PE_MODE=default
log_result "export PE_MODE=default"


log_command "train default"


llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /mnt/lyc/wuxinrui/R1_training/training/TCM4_addthinkprunedata_v8/TCM4_addthinkprunedata_v8_global_step_108/models \
    --preprocessing_num_workers 4 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset TCMv4_format_random2000_answer_prompt_generate_below2000_TCMv4 \
    --cutoff_len 5000 \
    --learning_rate 1.0e-05 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 200 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir TCM4_addthinkprunedata_v8_global_step_108_answer_prompt_SFT/outputs/ \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0 \
    --gradient_checkpointing True \
    --lora_target all \

log_result "llamafactory-cli train"


log_command "merge default"


llamafactory-cli export TCM4_addthinkprunedata_v8_global_step_108_answer_prompt_SFT/merge.yaml


log_result "llamafactory-cli export"
    # --include_num_input_tokens_seen True \
