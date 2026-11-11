CUDA_VISIBLE_DEVICES=0 bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh  1 1 1 1 1.0 0.5 256 0.02 "" 5 none pr # 20% layer for reference

CUDA_VISIBLE_DEVICES=1 bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh 1 1 1 1 1.0 0.5 256 0.02 "" 0 12 pr # use threshold=12 for participation

CUDA_VISIBLE_DEVICES=0 bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh 1 1 1 1 1.0 0.5 256 0.02 "" 0 none erank # use erank

CUDA_VISIBLE_DEVICES=1 bash scripts/cred_ablation/run_ablation_cred_cls.sh 1 1 1 1 1.0 0.1 256 0.02 "" 0 none var_cov 1.0 1.0 1.0 # VicRegL

bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh 1 1 1 1 1.0 0.5 256 0.02 "12 19" 4 none pr # pr reference từ 4


CUDA_VISIBLE_DEVICES=1 bash scripts/cred_ablation/train_single_old_ret_VisDial_CIRR.sh # train on subset of retrieval task 

CUDA_VISIBLE_DEVICES=0 bash scripts/cred_ablation/train_distill_sigreg_ret_VisDial_CIRR.sh # train on subset of retrieval task 

CUDA_VISIBLE_DEVICES=1 bash scripts/cred_ablation/train_distill_sigreg_ret_CIRR.sh # train on subset of retrieval task

CUDA_VISIBLE_DEVICES=0 bash scripts/cred_ablation/train_distill_sigreg_ret_CIRR.sh 1 1 1 0 1.0 0.5 256 0.02 # train on subset of retrieval task CPR w/o sigreg loss

CUDA_VISIBLE_DEVICES=0 bash scripts/cred_ablation/train_single_old_ret_CIRR.sh # train on subset of retrieval task