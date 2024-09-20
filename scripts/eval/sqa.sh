#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.6-vicuna-13b \
    --load-peft /scratch/svani/experiments/dci_experiments/llava-v1.6-vicuna-13b-lora-params \
    --question-file /scratch/svani/data/playground/data/ScienceQA_DATA/llava_test_QCM-LEA.json \
    --image-folder /scratch/svani/data/playground/data/ScienceQA_DATA/test \
    --answers-file /scratch/svani/evaluation/sqa/dci/llava-v1.6-vicuna-13b-lora.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /scratch/svani/data/playground/data/ScienceQA_DATA \
    --result-file /scratch/svani/evaluation/sqa/dci/llava-v1.6-vicuna-13b-lora.jsonl \
    --output-file /scratch/svani/evaluation/sqa/dci/llava-v1.6-vicuna-13b-output.jsonl \
    --output-result /scratch/svani/evaluation/sqa/dci/llava-v1.6-vicuna-13b-result.json
