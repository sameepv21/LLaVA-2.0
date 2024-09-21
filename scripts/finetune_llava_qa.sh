#!/bin/bash

DEEPSPEED=/home/svani/LVLM/LLaVA-2.0/scripts/zero2.json
DATA_PATH=/scratch/svani/data/playground/data/llava_instruct_5k.json
IMAGE_FOLDER=/scratch/svani/data/playground/data/coco/train2017
OUTPUT_DIR=/scratch/svani/experiments/dci_experiments/llava-v1.6-vicuna-13b-lora-params

deepspeed --master_port=25640 --include=localhost:0 --module llava.train.train_mem \
    --deepspeed $DEEPSPEED \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-13b \
    --version plain \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 25 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb