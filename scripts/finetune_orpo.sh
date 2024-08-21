#!/bin/bash

deepspeed --master_port=25640 --include=localhost:0 --module llava.train.train_mem \
    --deepspeed /home/svani/LVLM/temp/LLaVA/scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-13b \
    --version plain \
    --data_path /scratch/svani/data/densely_captioned_images/orpo-dci.json \
    --image_folder /scratch/svani/data/densely_captioned_images/photos \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /scratch/svani/experiments/dci_experiments/llava-v1.6-vicuna-13b-lora-params \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --orpo True