


export special_token_loss=T10



deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path /mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4_8ratio/1_5B_TCMv4_8ratio_models/models \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_FULL_1_5B/outputs_full_lossx10_1 \
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




export special_token_loss=T10



deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path TCMv4_8ratio_FULL_1_5B/outputs_full_lossx10_1 \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_FULL_1_5B/outputs_full_lossx10_2 \
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





export special_token_loss=T



deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path TCMv4_8ratio_FULL_1_5B/outputs_full_lossx10_2 \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_FULL_1_5B/outputs_full_lossx10_3 \
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
--deepspeed examples/deepspeed/ds_z0_config.json \
--stage sft \
--model_name_or_path TCMv4_8ratio_FULL_1_5B/outputs_full_lossx10_3 \
--do_train \
--dataset TCMv4_format_below10000_TCMv4_8ratio \
--template deepseek3 \
--finetuning_type full \
--output_dir  TCMv4_8ratio_FULL_1_5B/outputs_full_lossx10_4 \
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



