export special_token_loss=T


deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z3_config.json \
--stage sft \
--model_name_or_path /mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4_8ratio/7B_TCMv4_8ratio_models \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_FULL_7B/outputs_full_v1 \
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


deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z3_config.json \
--stage sft \
--model_name_or_path TCMv4_8ratio_FULL_7B/outputs_full_v1 \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_FULL_7B/outputs_full_v2 \
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