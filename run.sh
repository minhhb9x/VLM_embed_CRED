
CUDA_VISIBLE_DEVICES=0 bash \
    script_full/Cred_teacher_QwenVL3-8B/train_distill_cred_QwenVL3-8B_FastVLM-0.5B_cls.sh 1 1 1 1 1.0 0.5 256 0.02 &

CUDA_VISIBLE_DEVICES=0 bash \
    script_full/Cred_teacher_QwenVL3-8B/train_distill_cred_QwenVL3-8B_FastVLM-0.5B_vqa.sh 1 1 1 1 1.0 0.5 256 0.02 &

CUDA_VISIBLE_DEVICES=0 bash \
    script_full/Cred_teacher_QwenVL3-8B/train_distill_cred_QwenVL3-8B_FastVLM-0.5B_grounding.sh 1 1 1 1 1.0 0.5 256 0.02 &

CUDA_VISIBLE_DEVICES=0 bash \
    script_full/Cred_teacher_QwenVL3-8B/train_distill_cred_QwenVL3-8B_OV-0.5B_cls.sh 1 1 1 1 1.0 0.5 256 0.02 &

CUDA_VISIBLE_DEVICES=0 bash \
    script_full/Cred_teacher_QwenVL3-8B/train_distill_cred_QwenVL3-8B_OV-0.5B_vqa.sh 1 1 1 1 1.0 0.5 256 0.02 &

CUDA_VISIBLE_DEVICES=0 bash \
    script_full/Cred_teacher_QwenVL3-8B/train_distill_cred_QwenVL3-8B_OV-0.5B_grounding.sh 1 1 1 1 1.0 0.5 256 0.02 &


wait