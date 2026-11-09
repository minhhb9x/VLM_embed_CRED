#!/bin/bash
set -e

# Run from the repository root. Reuse saved .pt files with SKIP_INFER=1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
ER_DEVICE="${ER_DEVICE:-cuda}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

INFER_SCRIPT="$SCRIPT_DIR/infer_eval_hidden_attention.py"

INFER_SUBSETS=(
    "ImageNet-1K"
    # "N24News" "HatefulMemes" "VOC2007" "SUN397"
    # "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211"
)

MODELS=(
    # "raghavlite/B3_Qwen2_2B"
    # "apple/FastVLM-0.5B"
    # training1/FastVLM-0.5B_base_b16_cls/checkpoint-epoch-0
    # "meta_train/rkd_meta_cls/checkpoint-epoch-0"
    # "meta_train/ckd_meta_cls/checkpoint-epoch-0"
    # "meta_train/norm_meta_cls/checkpoint-epoch-0"
    # meta_train/rebuttal_hierd_cls_5layers/checkpoint-epoch-0
    "meta_train/sft_meta_llavaov_cls/checkpoint-epoch-0"
    "meta_train/rkd_meta_llavaov_cls/checkpoint-epoch-0"
    "meta_train/ckd_meta_llavaov_cls/checkpoint-epoch-0"
    # "meta_train/norm_meta_llavaov_cls/checkpoint-epoch-0"
    "meta_train/span_propose_llava_ov_cls_v3_final/checkpoint-epoch-0"
)

BACKBONES=(
    # "qwen2_vl"
    # "llava_qwen2_old"
    # "llava_qwen2_old"
    # "llava_qwen2_old"
    # "llava_qwen2_old"
    # "llava_qwen2_old"
    # "llava_qwen2_old"
    "llava_onevision_old"
    "llava_onevision_old"
    "llava_onevision_old"
    # "llava_onevision_old"
    "llava_onevision_old"
)

# Kiểm tra MODELS và BACKBONES có cùng số phần tử không
if [ "${#MODELS[@]}" -ne "${#BACKBONES[@]}" ]; then
    echo "Error: MODELS and BACKBONES must have the same length."
    exit 1
fi

mkdir -p infer analyze

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    BACKBONE="${BACKBONES[$i]}"
    # Same inference directory convention as the original launcher.
    EXP_NAME=$(basename "$(dirname "$MODEL")")
    EXTRA_ARGS=()
    OUTPUT_SUFFIX=""
    if [ "$BACKBONE" = "llava_onevision" ] || [ "$BACKBONE" = "llava_onevision_old" ]; then
        EXTRA_ARGS+=(--image_resolution tiny)
        OUTPUT_SUFFIX="_tiny"
    fi
    INFER_OUTPUT="infer/${EXP_NAME}${OUTPUT_SUFFIX}"
    echo "Model: $MODEL; backbone: $BACKBONE; infer: $INFER_OUTPUT"

    if [ "$SKIP_INFER" != "1" ]; then
        if ! python "$INFER_SCRIPT" \
            --model_name "$MODEL" \
            --lora True \
            --lora_r 64 \
            --lora_alpha 64 \
            --pooling eos \
            --model_backbone "$BACKBONE" \
            --normalize True \
            --bf16 \
            --dataset_name "TIGER-Lab/MMEB-eval" \
            --subset_name "${INFER_SUBSETS[@]}" \
            --dataset_split test \
            --image_dir eval_images/ \
            --tgt_prefix_mod \
            --encode_output_path "$INFER_OUTPUT" \
            --per_device_eval_batch_size 1 \
            --report_to None \
            "${EXTRA_ARGS[@]}"; then
            echo "ERROR: Infer failed for $EXP_NAME" >&2
            continue
        fi
    fi

    # Analyze exactly the same saved samples with both definitions.
    # --normalize_by_min_dim: min(n,d) for uncentered; min(n-1,d) for centered.
    # This is different from --normalize, which L2-normalizes individual tokens.
    for SUBSET in "${INFER_SUBSETS[@]}"; do
        PT_DIR="${INFER_OUTPUT}/${SUBSET}/query"
        ANALYZE_BASE="analyze/${EXP_NAME}${OUTPUT_SUFFIX}_${SUBSET}"
        python "$SCRIPT_DIR/er_statistic.py" \
            --pt_dir "$PT_DIR" \
            --num_samples 0 \
            --device "$ER_DEVICE" \
            --normalize_by_min_dim \
            --output_file "${ANALYZE_BASE}_uncentered.txt"

        python "$SCRIPT_DIR/er_statistic_centered.py" \
            --pt_dir "$PT_DIR" \
            --num_samples 0 \
            --device "$ER_DEVICE" \
            --normalize_by_min_dim \
            --output_file "${ANALYZE_BASE}_centered.txt"
    done
    rm -rf "$INFER_OUTPUT"
    echo "Finished: ${EXP_NAME}${OUTPUT_SUFFIX}"
done
