CUDA_VISIBLE_DEVICES=0 python benchmark_pr.py \
    --device cuda \
    --hidden-dim 896 \
    --batch-size 16 \
    --projections 256 \
    --tokens 144 256 576 1024 2048 4096 \
    --mode forward \
    --output results/pr_forward_D896.csv

# CUDA_VISIBLE_DEVICES=0 python benchmark_pr.py \
#     --device cuda \
#     --hidden-dim 896 \
#     --batch-size 1 \
#     --projections 256 \
#     --mode backward \
#     --output results/pr_backward_D896.csv