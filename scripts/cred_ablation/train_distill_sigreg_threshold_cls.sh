#!/bin/bash

set -e

# Run from the repository root.
# Usage:
# bash run_ablation_cred_cls.sh [distill] [cse] [vision] [sigreg] \
#     [kd_weight] [sigreg_weight] [num_projections] [d_tau] \
#     [cpr_layers] [cpr_reference_layer] [cpr_threshold] [cpr_statistic]
#
# Defaults: True True True True 1.0 0.05 256 0.02 "" 0 none pr
# cpr_layers: "" uses the criterion defaults; otherwise "12 19" or "12,19".
# cpr_threshold: none uses the reference layer; a number uses a fixed threshold.
# cpr_statistic: pr or erank.

USE_DISTILL_LOSS=${1:-True}
USE_DISTILL_CSE_LOSS=${2:-True}
USE_DISTILL_VISON_LOSS=${3:-True}
USE_SIGREG_LOSS=${4:-True}
KD_WEIGHT=${5:-1.0}
SIGREG_WEIGHT=${6:-0.05}
NUM_LAYER=${7:-256}
D_TAU=${8:-0.02}
CPR_LAYERS=${9:-}
CPR_REFERENCE_LAYER=${10:-0}
CPR_THRESHOLD=${11:-none}
CPR_STATISTIC=${12:-pr}

bool_to_python() {
    case "$1" in
        1|true|True|TRUE) echo "True" ;;
        0|false|False|FALSE) echo "False" ;;
        *)
            echo "ERROR: Boolean argument must be 0/1 or True/False, got '$1'" >&2
            exit 1
            ;;
    esac
}

bool_to_int() {
    case "$1" in
        1|true|True|TRUE) echo "1" ;;
        0|false|False|FALSE) echo "0" ;;
        *)
            echo "ERROR: Boolean argument must be 0/1 or True/False, got '$1'" >&2
            exit 1
            ;;
    esac
}

DISTILL_LOSS_BOOL=$(bool_to_python "$USE_DISTILL_LOSS")
DISTILL_CSE_BOOL=$(bool_to_python "$USE_DISTILL_CSE_LOSS")
DISTILL_VISION_BOOL=$(bool_to_python "$USE_DISTILL_VISON_LOSS")
SIGREG_BOOL=$(bool_to_python "$USE_SIGREG_LOSS")

D_DISTILL=$(bool_to_int "$USE_DISTILL_LOSS")
D_CSE=$(bool_to_int "$USE_DISTILL_CSE_LOSS")
D_VISION=$(bool_to_int "$USE_DISTILL_VISON_LOSS")
D_SIGREG=$(bool_to_int "$USE_SIGREG_LOSS")

# Always pass the reference layer and statistic. Only pass optional values
# when supplied, so Python keeps cpr_layers=[] and cpr_threshold=None.
CPR_ARGS=(
    --cpr_reference_layer "$CPR_REFERENCE_LAYER"
    --cpr_statistic "$CPR_STATISTIC"
)

CPR_LAYERS_TAG=auto
CPR_LAYERS_DISPLAY=auto
if [[ -n "$CPR_LAYERS" ]]; then
    read -r -a CPR_LAYER_LIST <<< "${CPR_LAYERS//,/ }"
    if [[ ${#CPR_LAYER_LIST[@]} -eq 0 ]]; then
        echo "ERROR: cpr_layers must be empty or a list of layer indices." >&2
        exit 1
    fi
    for layer in "${CPR_LAYER_LIST[@]}"; do
        if [[ ! "$layer" =~ ^[0-9]+$ ]]; then
            echo "ERROR: Invalid CPR layer index: '$layer'" >&2
            exit 1
        fi
    done
    CPR_ARGS+=(--cpr_layers "${CPR_LAYER_LIST[@]}")
    CPR_LAYERS_TAG=$(IFS=-; echo "${CPR_LAYER_LIST[*]}")
    CPR_LAYERS_DISPLAY="${CPR_LAYER_LIST[*]}"
fi

case "$CPR_THRESHOLD" in
    none|None|NONE) CPR_THRESHOLD=none ;;
    *)
        if [[ ! "$CPR_THRESHOLD" =~ ^[+]?[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$ &&
              ! "$CPR_THRESHOLD" =~ ^[+]?[.][0-9]+([eE][+-]?[0-9]+)?$ ]]; then
            echo "ERROR: cpr_threshold must be none or a nonnegative number." >&2
            exit 1
        fi
        CPR_ARGS+=(--cpr_threshold "$CPR_THRESHOLD")
        ;;
esac

if [[ ! "$CPR_REFERENCE_LAYER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: cpr_reference_layer must be a nonnegative integer." >&2
    exit 1
fi
if [[ "$CPR_STATISTIC" != pr && "$CPR_STATISTIC" != erank ]]; then
    echo "ERROR: cpr_statistic must be pr or erank." >&2
    exit 1
fi

EXP_NAME="ablation_cred_d${D_DISTILL}_cse${D_CSE}_vis${D_VISION}_sig${D_SIGREG}_kd${KD_WEIGHT}_sw${SIGREG_WEIGHT}_m${NUM_LAYER}_dt${D_TAU}"
EXP_NAME+="_layers${CPR_LAYERS_TAG}_ref${CPR_REFERENCE_LAYER}_thr${CPR_THRESHOLD}_${CPR_STATISTIC}"

PREFIX=FastVLM-0.5B_cls
OUTPUT_DIR="training/${PREFIX}_${EXP_NAME}"
CACHE_DIR="caching/B3_Qwen2_2B_cls"
EVAL_OUTPUT="./MMEB-eval_outputs/${PREFIX}_${EXP_NAME}/"

echo "============================================================"
echo "Experiment: $EXP_NAME"
echo "  USE_DISTILL_LOSS       = $DISTILL_LOSS_BOOL"
echo "  USE_DISTILL_CSE_LOSS   = $DISTILL_CSE_BOOL"
echo "  USE_DISTILL_VISON_LOSS = $DISTILL_VISION_BOOL"
echo "  USE_SIGREG_LOSS        = $SIGREG_BOOL"
echo "  KD_WEIGHT              = $KD_WEIGHT"
echo "  SIGREG_WEIGHT          = $SIGREG_WEIGHT"
echo "  NUM_PROJECTIONS        = $NUM_LAYER"
echo "  D_TAU                  = $D_TAU"
echo "  CPR_LAYERS             = $CPR_LAYERS_DISPLAY"
echo "  CPR_REFERENCE_LAYER    = $CPR_REFERENCE_LAYER"
echo "  CPR_THRESHOLD          = $CPR_THRESHOLD"
echo "  CPR_STATISTIC          = $CPR_STATISTIC"
echo "Train: $OUTPUT_DIR"
echo "Eval:  $EVAL_OUTPUT"
echo "============================================================"

NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_ddp.py"

torchrun --standalone \
    --nproc_per_node="$NUM_GPUS_PER_NODE" "$TRAIN_SCRIPT" \
    --model_name apple/FastVLM-0.5B \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.05 \
    --caching_dir "$CACHE_DIR" \
    --kd_loss_type "ablation_cred" \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config_emo.json" \
    --projector_lr 5e-4 \
    --report_to None \
    --use_distill_loss "$DISTILL_LOSS_BOOL" \
    --use_distill_cse_loss "$DISTILL_CSE_BOOL" \
    --use_distill_vison_loss "$DISTILL_VISION_BOOL" \
    --use_sigreg_loss "$SIGREG_BOOL" \
    --kd_weight "$KD_WEIGHT" \
    --sigreg_weight "$SIGREG_WEIGHT" \
    --num_layers "$NUM_LAYER" \
    --d_cse_temperature "$D_TAU" \
    "${CPR_ARGS[@]}"

MODEL="$OUTPUT_DIR/checkpoint-epoch-0"
if [ ! -d "$MODEL" ]; then
    echo "ERROR: Checkpoint not found: $MODEL" >&2
    exit 1
fi

echo "Training finished. Checkpoint: $MODEL"

SUBSETS=(
    "ImageNet-1K"
    "N24News"
    "HatefulMemes"
    "VOC2007"
    "SUN397"
    "Place365"
    "ImageNet-A"
    "ImageNet-R"
    "ObjectNet"
    "Country211"
)

python eval_mmeb.py \
    --model_name "$MODEL" \
    --encode_output_path "$EVAL_OUTPUT" \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 8 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none

echo "============================================================"
echo "DONE: $EXP_NAME"
echo "Train: $OUTPUT_DIR"
echo "Eval:  $EVAL_OUTPUT"
echo "============================================================"
