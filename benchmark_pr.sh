HIDDEN_DIM=896
CUDA_VISIBLE_DEVICES=0 python benchmark_pr.py \
    --device cuda \
    --hidden-dim "${HIDDEN_DIM}" \
    --batch-size 16 \
    --projections 256 \
    --tokens 144 256 512 576 1024 2048 4096 \
    --output "ER_outputs/pr_forward_D${HIDDEN_DIM}.csv"

# CUDA_VISIBLE_DEVICES=0 python benchmark_pr.py \
#     --device cuda \
#     --hidden-dim 896 \
#     --batch-size 1 \
#     --projections 256 \
#     --mode backward \
#     --output results/pr_backward_D896.csvq