#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_${TIMESTAMP}.log"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

MODEL_PATH="Qwen/Qwen3-VL-7B-Instruct"
OUTPUT_DIR="./checkpoints/qwen3vl_caption"
CACHE_DIR="./cache"

DATASETS="your_caption_dataset%100"
DATA_FLATTEN=True

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

args="
    --model_name_or_path \"${MODEL_PATH}\" \
    --cache_dir ${CACHE_DIR} \
    --dataset_use ${DATASETS} \
    --data_flatten ${DATA_FLATTEN} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_pixels $((448*448*3)) \
    --min_pixels $((448*448*3)) \
    --eval_strategy \"no\" \
    --save_strategy \"steps\" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-6 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type \"cosine\" \
    --logging_steps 10 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name \"qwen3vl_caption_finetune\" \
    --report_to none \
    --optim adamw_torch"

echo "Starting Qwen3-VL fine-tuning..."
echo "Log file: ${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         qwenvl/train/train_qwen.py ${args} 2>&1 | tee ${LOG_FILE}
