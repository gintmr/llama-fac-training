deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path /mnt/lyc/wuxinrui/DS_Huggingface/DS_QW_1_5B \
--do_train \
--dataset TCMv4_format_without_remaining \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_FULL_1_5B_withoutremaining_v2/outputs_full_1 \
--overwrite_cache \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 1000 \
--learning_rate 2e-5 \
--num_train_epochs 2.0 \
--plot_loss \
--bf16



deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path TCMv4_FULL_1_5B_withoutremaining_v2/outputs_full_1 \
--do_train \
--dataset TCMv4_format_without_remaining \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_FULL_1_5B_withoutremaining_v2/outputs_full_2 \
--overwrite_cache \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 1000 \
--learning_rate 2e-5 \
--num_train_epochs 4.0 \
--plot_loss \
--bf16
