#!/bin/bash

# 设置日志文件路径
LOG_FILE="/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4_8ratio_FULL_1_5B/train_2stage.txt"


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



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# log_command "train 1"

# deepspeed --num_gpus 8 src/train.py \
# --deepspeed examples/deepspeed/ds_z0_config.json \
# --stage sft \
# --model_name_or_path TCMv4_250token/1_5B_TCMv4_250token_models  \
# --do_train \
# --dataset TCMv4_format_below10000_TCMv4_250token \
# --template deepseek3 \
# --finetuning_type full \
# --output_dir  TCMv4_250token_FULL_1_5B/outputs_full_1 \
# --overwrite_cache \
# --per_device_train_batch_size 2 \
# --gradient_accumulation_steps 8 \
# --lr_scheduler_type cosine \
# --logging_steps 10 \
# --save_steps 2000 \
# --learning_rate 2e-5 \
# --num_train_epochs 2.0 \
# --plot_loss \
# --bf16

# log_result "train 1"



# log_command "train 2"

# deepspeed --num_gpus 8 src/train.py \
# --deepspeed examples/deepspeed/ds_z0_config.json \
# --stage sft \
# --model_name_or_path TCMv4_250token_FULL_1_5B/outputs_full_1  \
# --do_train \
# --dataset TCMv4_format_below10000_TCMv4_250token \
# --template deepseek3 \
# --finetuning_type full \
# --output_dir  TCMv4_250token_FULL_1_5B/outputs_full_2 \
# --overwrite_cache \
# --per_device_train_batch_size 2 \
# --gradient_accumulation_steps 8 \
# --lr_scheduler_type cosine \
# --logging_steps 10 \
# --save_steps 2000 \
# --learning_rate 2e-5 \
# --num_train_epochs 4.0 \
# --plot_loss \
# --bf16

# log_result "train 2"



export special_token_loss=T



log_command "train 3"

deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path /mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4_8ratio/1_5B_TCMv4_8ratio_models/models  \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_FULL_1_5B/outputs_full_1 \
--overwrite_cache \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 2000 \
--learning_rate 2e-5 \
--num_train_epochs 2.0 \
--plot_loss \
--bf16

log_result "train 3"


log_command "special token loss"

export special_token_loss=F

log_result "special token loss = F"



log_command "train 4"

deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path TCMv4_8ratio_FULL_1_5B/outputs_full_1  \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_FULL_1_5B/outputs_full_2 \
--overwrite_cache \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 2000 \
--learning_rate 2e-5 \
--num_train_epochs 4.0 \
--plot_loss \
--bf16

log_result "train 4"

export special_token_loss=T







'''
四组训练参数：
1. 1.5B 250token 2e-5 2.0
2. 1.5B 250token 1e-5 4.0
1、2均使用TCMv4_format_below10000_TCMv4_250token数据训练，主要目的是观察2epoch与6epoch的区别

3. 1.5B 8ratio 2e-5 2.0
4. 1.5B 8ratio 1e-5 4.0
3，4均使用TCMv4_format_below10000_TCMv4_8ratio数据训练，前三个epoch充分学习7个新token，再mask掉特殊token的loss，训练6epoch，让模型学会不能输出特殊token

'''