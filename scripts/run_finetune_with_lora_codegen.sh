#!/bin/bash
# Please run this script under ${project_id} in project directory of
export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=finetune_with_lora_codegen_1
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/alpaca/train

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ./models/codegen-350M-multi \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 0.01 \
    --learning_rate 1e-4 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --use_lora 1 \
    --lora_r 8 \
    --save_aggregated_lora 1\
    --deepspeed configs/ds_config_zero2.json \
    --bf16 \
    --run_name finetune_with_lora \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 50 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
