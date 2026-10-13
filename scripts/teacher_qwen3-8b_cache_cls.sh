#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="teacher_cache_qwen3.py"

# SUBSETS=(
#   "VOC2007"
#   "OK-VQA"
# )

SUBSETS=(
  "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
)

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE \
    $TRAIN_SCRIPT \
    --model_name Qwen/Qwen3-VL-Embedding-8B \
    --model_backbone "qwen3_vl" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir "caching/Qwen3VL_8B_cls" \
    --per_device_train_batch_size 4 \
    --image_resolution "mid" \
    --seed 42 \
    --normalize False \
    --report_to "none" 