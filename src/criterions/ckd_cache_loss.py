import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from src.criterions.utils import count_clean_text_tokens, get_hidden_text_vision, pooling


class CKDCacheLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kd_loss_weight = args.kd_weight

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

    def _dist_gather_tensor(self, t: Tensor):
        if self.world_size == 1:
            return t

        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t

        return torch.cat(all_tensors, dim=0)


    def ckd_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        student_diff = student.unsqueeze(1) - student.unsqueeze(0)
        teacher_diff = teacher.unsqueeze(1) - teacher.unsqueeze(0)

        return F.mse_loss(student_diff, teacher_diff)

    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer 
        projectors = model_wrapper.projectors

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device
        dtype = student_qry_reps.dtype

        teacher_qry, teacher_pos = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"]

        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry], dim=0).to(device, dtype=dtype)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos], dim=0).to(device, dtype=dtype)

        if getattr(self, 'world_size', 1) > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
            all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
            all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            all_teacher_qry_reps = teacher_qry_reps
            all_teacher_pos_reps = teacher_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / model_wrapper.temperature, target)

        # =====================================================
        # Project teacher to student embedding dimension
        # =====================================================

        projected_student_qry_reps = projectors["s2t"](all_student_qry_reps)
        projected_student_pos_reps = projectors["s2t"](all_student_pos_reps)


        all_projected_student_qry_reps = self._dist_gather_tensor(projected_student_qry_reps)
        all_projected_student_pos_reps = self._dist_gather_tensor(projected_student_pos_reps)

        # =====================================================
        # Comparative Knowledge Distillation
        #
        # z = [query embeddings, positive embeddings]
        #
        # L_CKD = MSE(
        #     z_s[i] - z_s[j],
        #     z_t[i] - z_t[j]
        # )
        # =====================================================
        student_ckd_reps = torch.cat([all_projected_student_qry_reps, all_projected_student_pos_reps], dim=0)
        teacher_ckd_reps = torch.cat([all_teacher_qry_reps, all_teacher_pos_reps], dim=0)

        kd_loss = self.ckd_loss(student_ckd_reps, teacher_ckd_reps)


        loss = contrastive_loss + self.kd_loss_weight * kd_loss

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }