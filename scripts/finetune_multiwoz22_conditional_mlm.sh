#!/bin/bash
# USAGE: <script> [debug]
#
#
# based on https://aclanthology.org/D18-1547.pdf Table 1
# avg turns per dialogues 13.68
# avg tokens per turn     13.18
# --> avg dialogue has ~ 200 words
set -euo pipefail
source ./scripts/setup_env.sh


if [[ $1 = "debug" ]] ; then
  printf "\n\nWARNING: You are in a debugging mode for testing the script. Using a small model, few steps, etc.\n\n\n"
  max_steps=4
  gradient_accumulation_steps=2
  logging_steps=2
  save_steps=2
  dataloader_num_workers=0
  max_eval_samples=3
  model_name_or_path="EleutherAI/pythia-70m"
else
  # These args make the training to take long.
  # For debugging / starting with the script it make sense to try faster & less performant settings.
  max_steps=1875
  gradient_accumulation_steps=16
  logging_steps=10
  save_steps=500
  dataloader_num_workers=4
  max_eval_samples=1000
  model_name_or_path="huggyllama/llama-7b"
  # model_name_or_path="$1"  # uncomment if you want to provide your model on command line
fi

$PYTHON \
  qlora.py \
    --dataset multi_woz_v22 \
    --dataset_format multi_woz_v22_turns \
    --do_train \
    --do_eval \
    --max_steps $max_steps \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --logging_steps $logging_steps \
    --save_steps $save_steps \
    --dataloader_num_workers $dataloader_num_workers \
    --max_eval_samples $max_eval_samples \
    --model_name_or_path $model_name_or_path \
    --source_max_len 256 \
    --target_max_len 288 \
    --max_new_tokens 32 \
    --output_dir "./output/${model_name_or_path}_$(date +'%s')_$$" \
    --report_to wandb \
    --save_strategy steps \
    --data_seed 42 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --group_by_length false

    # Commented out options needs to at the end
    # without group_by_length it will be less efficient
    # but 1) I do not get how it affects the training if batch_size is 1
    #     2) without it learns first, second, third, etc turns  together  
    # --group_by_length \
    # --do_mmlu_eval \

