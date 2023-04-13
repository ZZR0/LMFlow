#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

# export WANDB_MODE=offline

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=finetune_llama
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/koala/train
eval_dataset_path=${project_dir}/data/koala/test/test.json

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${project_dir}/models/llama-7b-hf \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --block_size 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name finetune_llama7b \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 100 \
    --dataloader_num_workers 4 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
