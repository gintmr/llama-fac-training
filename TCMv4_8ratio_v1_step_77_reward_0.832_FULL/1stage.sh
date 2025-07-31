


export special_token_loss=T



deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path /mnt/lyc/wuxinrui/R1_training/training/TCMv4_8ratio_v1/TCMv4_8ratio_v1_step_77_reward_0.832/models \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio_below2k \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_v1_step_77_reward_0.832_FULL/outputs_full_1 \
--overwrite_cache \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 2000 \
--learning_rate 1e-6 \
--num_train_epochs 2.0 \
--plot_loss \
--bf16




export special_token_loss=T



deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path TCMv4_8ratio_v1_step_77_reward_0.832_FULL/outputs_full_1 \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio_below2k \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_v1_step_77_reward_0.832_FULL/outputs_full_2 \
--overwrite_cache \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 2000 \
--learning_rate 1e-6 \
--num_train_epochs 2.0 \
--plot_loss \
--bf16


