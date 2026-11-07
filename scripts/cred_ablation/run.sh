bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh  1 1 1 1 1.0 0.05 256 0.02 "" 5 none pr # 20% layer for reference

bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh 1 1 1 1 1.0 0.05 256 0.02 "" 0 12 pr # use threshold=12 for participation

CUDA_VISIBLE_DEVICES=0 bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh 1 1 1 1 1.0 0.05 256 0.02 "" 0 none erank # use erank

bash scripts/cred_ablation/train_distill_sigreg_threshold_cls.sh 1 1 1 1 1.0 0.05 256 0.02 "12 19" 4 none pr # pr reference từ 4


CUDA_VISIBLE_DEVICES=1 bash scripts/cred_ablation/train_single_old_ret_VisDial_CIRR.sh # train on subset of retrieval task 