import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text_vision, pooling
import random
import math
from torch.nn.utils.rnn import pad_sequence


class AblationCred(nn.Module):
    def __init__(self, args):
        super(AblationCred, self).__init__()
        self.args = args
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        self.kd_weight = args.kd_weight

        self.counter = 0
        self.warm_up_sigreg = 0

        self.cpr_statistic = getattr(args, "cpr_statistic", "pr")
        self.cpr_threshold = getattr(args, "cpr_threshold", None)
        if self.cpr_statistic not in ("pr", "erank"):
            raise ValueError("cpr_statistic must be pr or erank.")
        if self.cpr_threshold is not None and (
            not math.isfinite(self.cpr_threshold) or self.cpr_threshold < 0
        ):
            raise ValueError("cpr_threshold must be finite and nonnegative.")

    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def cosine_loss(self, student_embeddings, teacher_embeddings):
        cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
        cos_sim_loss = 1 - cos_sim
        return cos_sim_loss.mean()

    def structure_loss(self, student_embeddings, teacher_embeddings):
        student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)

        student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)
        teacher_similarity = teacher_embeddings @ teacher_embeddings.transpose(-1, -2)

        loss = F.mse_loss(student_similarity, teacher_similarity)

        return loss

    def distillcse_kd_loss(self, S1, S2, T1, T2, tau=0.02):
        """
        Distill teacher similarity distribution over in-batch negatives.

        Student and teacher dimensions do not need to match because
        distillation is applied to pairwise similarity matrices.
        """
        S1 = F.normalize(S1.float(), p=2, dim=-1)

        S2 = F.normalize(S2.float(), p=2, dim=-1)

        T1 = F.normalize(T1.float(), p=2, dim=-1)

        T2 = F.normalize(T2.float(), p=2, dim=-1,)

        s_logits = (S1 @ S2.transpose(0, 1)) / tau

        t_logits = (T1 @ T2.transpose(0, 1)) / tau

        # Positive query-passage pairs are on the diagonal.
        # DistillCSE KD here focuses on the negative distribution.
        mask = torch.eye(s_logits.size(0), device=s_logits.device, dtype=torch.bool,)

        s_logits = s_logits.masked_fill(mask, torch.finfo(s_logits.dtype).min,)

        t_logits = t_logits.masked_fill( mask,torch.finfo(t_logits.dtype).min,)

        teacher_probs = F.softmax(t_logits,dim=1,).detach()

        student_log_probs = F.log_softmax(s_logits,dim=1,)

        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean",)

    def sketched_participation_ratio_erank(self, z_list_first: list[torch.Tensor],
                                           z_list_last: list[torch.Tensor],
                                           num_slices: int = 256,
                                           min_valid_tokens: int = 4, eps: float = 1e-8):

        B = len(z_list_last)
        if B == 0:
            return 0.0

        device, dtype = z_list_last[0].device, z_list_last[0].dtype
        D = z_list_last[0].shape[-1]

        def _pad_and_mask(z_list):
            lengths = torch.tensor([x.size(0) for x in z_list], device=device)
            z_padded = pad_sequence(z_list, batch_first=True, padding_value=0.0)
            N_max = z_padded.size(1)
            idx = torch.arange(N_max, device=device).unsqueeze(0)
            mask = idx < lengths.unsqueeze(1)
            return z_padded, mask, lengths

        z0_padded, mask0, len0 = _pad_and_mask(z_list_first)
        zL_padded, maskL, lenL = _pad_and_mask(z_list_last)

        z0_padded = z0_padded.detach().float()
        zL_padded = zL_padded.float()

        if self.cpr_statistic == "pr":
            A = torch.randn(D, num_slices, device=device, dtype=torch.float32)
            A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(eps)   # [D, M]

        def _participation_ratio(z_padded, mask, lengths):
            mask_f = mask.unsqueeze(-1).float()                                        # [B, N_max, 1]
            n_valid = lengths.clamp(min=2).float()                                     # [B]

            # Center (giống bước 0 cũ) — bắt buộc để proj mang đúng ý nghĩa "centered"
            mean = (z_padded * mask_f).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
            z_c = (z_padded - mean) * mask_f                                           # [B, N_max, D], padding = 0

            if self.cpr_statistic == "erank":
                # Squared singular values give the covariance spectrum up to scale.
                eigenvalues = torch.linalg.svdvals(z_c).square()
                total = eigenvalues.sum(dim=-1, keepdim=True)
                probabilities = eigenvalues / total.clamp_min(eps)
                entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
                erank = torch.where(total.squeeze(-1) > 0, entropy.exp(), 0.0)
                return erank, lengths >= min_valid_tokens

            # --- Bậc 1: trace(Cov) — CHÍNH XÁC như code cũ ---
            proj = z_c @ A                                                             # [B, N_max, M]
            var_proj = (proj ** 2).sum(dim=1) / (n_valid - 1.0).unsqueeze(-1)          # [B, M] = a^T Cov a
            trace1 = D * var_proj.mean(dim=-1)                                         # [B]  ≈ trace(Cov)

            # --- Bậc 2: trace(Cov^2) — THÊM MỘT MATMUL, KHÔNG LẶP ---
            # Cov @ a_m  =  Z_c^T @ (Z_c @ a_m) / (n-1)   với mỗi m cùng lúc:
            Cov_A = torch.bmm(z_c.transpose(1, 2), proj) / (n_valid - 1.0).view(-1, 1, 1)  # [B, D, M]
            term2 = (Cov_A ** 2).sum(dim=1)                                            # [B, M] = a^T Cov^2 a
            trace2 = D * term2.mean(dim=-1)                                            # [B]  ≈ trace(Cov^2)

            pr = trace1.pow(2) / trace2.clamp_min(eps)                                 # [B]  participation ratio
            valid = lengths >= min_valid_tokens
            return pr, valid

        z0_normed = z0_padded
        zL_normed = zL_padded

        prL, validL = _participation_ratio(zL_normed, maskL, lenL)
        if self.cpr_threshold is None:
            pr0, valid0 = _participation_ratio(z0_normed, mask0, len0)
        else:
            pr0 = torch.full_like(prL, self.cpr_threshold)
            valid0 = validL

        valid = valid0 & validL
        if not valid.any():
            return zL_padded.sum() * 0.0

        loss_per_sample = F.relu(pr0 - prL)

        return loss_per_sample[valid].mean().to(dtype)

    def sketched_std_erank(self, z_list_first: list[torch.Tensor], z_list_last: list[torch.Tensor],
                                num_slices: int = 256, min_valid_tokens: int = 4, eps: float = 1e-8):

        B = len(z_list_last)
        if B == 0:
            return 0.0

        device, dtype = z_list_last[0].device, z_list_last[0].dtype
        D = z_list_last[0].shape[-1]
        assert z_list_first[0].shape[-1] == D, (
            f"z_list_first và z_list_last phải cùng chiều D "
            f"(nhận {z_list_first[0].shape[-1]} và {D}); chiếu qua projector trước nếu khác chiều."
        )

        # ==========================================
        # 0. PADDING & MASKING — Xử lý số lượng token không đồng đều
        # ==========================================
        def _pad_and_mask(z_list):
            lengths = torch.tensor([x.size(0) for x in z_list], device=device)
            z_padded = pad_sequence(z_list, batch_first=True, padding_value=0.0)  # [B, N_max, D]
            N_max = z_padded.size(1)
            idx = torch.arange(N_max, device=device).unsqueeze(0)                 # [1, N_max]
            mask = idx < lengths.unsqueeze(1)                                     # [B, N_max]
            return z_padded, mask, lengths

        z0_padded, mask0, len0 = _pad_and_mask(z_list_first)
        zL_padded, maskL, lenL = _pad_and_mask(z_list_last)

        z0_padded = z0_padded.detach().float()
        zL_padded = zL_padded.float()

        # ==========================================
        # 1. RANDOM PROJECTIONS — Chiếu xuống M hướng để ước lượng Trace
        # ==========================================

        A = torch.randn(D, num_slices, device=device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(eps)  # [D, M]

        def _projected_variance(z_padded, mask, lengths):
            mask_f = mask.unsqueeze(-1).to(dtype=torch.float32)                # [B, N_max, 1]
            n_valid = lengths.clamp(min=2).to(torch.float32)                   # [B]

            proj = z_padded @ A                                                # [B, N_max, M]

            mean_proj = (proj * mask_f).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1) # [B, 1, M]
            centered_proj = (proj - mean_proj) * mask_f                        # [B, N_max, M]
            var_proj = (centered_proj ** 2).sum(dim=1) / (n_valid - 1.0).unsqueeze(-1)    # [B, M]

            valid = lengths >= min_valid_tokens                                # [B]
            return var_proj, valid

        var_0, valid_0 = _projected_variance(z0_padded, mask0, len0)
        var_L, valid_L = _projected_variance(zL_padded, maskL, lenL)

        valid = valid_0 & valid_L
        if not valid.any():
            return zL_padded.sum() * 0.0

        # ==========================================
        # 2. HINGE VARIANCE LOSS — Phạt nếu Erank của Layer L xẹp hơn Layer 0
        # ==========================================
        # std_0 = torch.sqrt(var_0 + eps)
        # std_L = torch.sqrt(var_L + eps)

        # loss_per_slice = F.relu(std_0 - std_L)              # [B, M]
        # loss_per_sample = loss_per_slice.mean(dim=1)        # [B]

        sum_var_0 = var_0.sum(dim=1, keepdim=True).clamp_min(eps)
        sum_var_L = var_L.sum(dim=1, keepdim=True).clamp_min(eps)

        p_0 = var_0 / sum_var_0  # [B, M]
        p_L = var_L / sum_var_L  # [B, M]

        # Tính Shannon Entropy (-sum(p * log(p)))
        entropy_0 = -(p_0 * torch.log(p_0 + eps)).sum(dim=1)  # [B]
        entropy_L = -(p_L * torch.log(p_L + eps)).sum(dim=1)  # [B]

        # Hinge Loss: Ép độ phân tán năng lượng (Entropy) của Layer L >= Layer 0
        loss_per_sample = F.relu(entropy_0 - entropy_L)       # [B]
        # loss_per_sample = (entropy_0 - entropy_L) ** 2

        return loss_per_sample[valid].mean().to(dtype)


    def _compute_modality_distill(self, student_hidden_states, image_features,
                                  text_token_counts, attention_mask):
        """
        Hàm này chỉ còn nhiệm vụ trích xuất text và vision representations 
        của student, cùng với việc tính toán SIGReg loss.
        """

        batch_size = attention_mask.size(0)
        last_layer_idx = len(student_hidden_states) - 1

        reference_layer = getattr(self.args, "cpr_reference_layer", 0)
        regularized_layers = getattr(self.args, "cpr_layers", [])
        if not regularized_layers:
            layer_mapping = self.args.student_layer_mapping
            if 0 in layer_mapping and last_layer_idx in layer_mapping:
                regularized_layers = layer_mapping[1:-1]
            else:
                regularized_layers = [int(last_layer_idx / 2), int(4 * last_layer_idx / 5)]
        if any(layer < 0 or layer > last_layer_idx for layer in [reference_layer] + regularized_layers):
            raise ValueError(f"CPR layer indices must be between 0 and {last_layer_idx}.")
        layers = sorted(set([reference_layer, last_layer_idx] + regularized_layers))

        stu_img_tokens = {l: [] for l in layers}
        stu_text_reps = []

        cur_idx_img = 0
        for i in range(batch_size):
            num_vision_token = 0
            if image_features is not None and cur_idx_img < len(image_features):
                num_vision_token = image_features[cur_idx_img].size(0)
                cur_idx_img += 1

            text_last_hidden, img_last_hidden = get_hidden_text_vision(
                student_hidden_states[last_layer_idx][i],
                text_token_counts[i].item(),
                num_vision_token,
                attention_mask[i]
            )
            stu_text_reps.append(text_last_hidden.mean(dim=0))

            if num_vision_token > 0:
                for l in layers:
                    _, img_hidden = get_hidden_text_vision(
                        student_hidden_states[l][i],
                        text_token_counts[i].item(),
                        num_vision_token,
                        attention_mask[i]
                    )
                    stu_img_tokens[l].append(img_hidden)

        # 1. Gom representations của Text
        stacked_stu_text_reps = torch.stack(stu_text_reps, dim=0)

        # 2. Gom representations của Vision và tính SIGReg
        stu_img_final_reps = None
        sigreg_final = 0.0

        if len(stu_img_tokens[last_layer_idx]) > 0:
            stu_img_final_reps = torch.stack([x.mean(dim=0) for x in stu_img_tokens[last_layer_idx]], dim=0)

            warmup_factor = min(1.0, self.counter / max(1, self.warm_up_sigreg))
            total_sigreg = 0.0
            sigreg_erank_loss = 0.0

            k_layers = 0
            num_slides = self.args.num_layers
            for l in regularized_layers:
                k_layers += 1
                # eos_query = pooling(student_hidden_states[l], attention_mask,
                #                     mode='eos', normalize=True).detach()
                # total_sigreg += self.sigreg_dualview(stu_img_tokens[l], eos_query,
                #                                      tau=0.1, alpha=0.9)
                # total_sigreg += self.sigreg_sinkhorn(stu_img_tokens[l], concept_queries)

                sigreg_erank_loss += self.sketched_participation_ratio_erank(stu_img_tokens[reference_layer],
                                                                             stu_img_tokens[l],
                                                                             num_slices=num_slides)

            if self.args.use_sigreg_loss:
                sigreg_final = sigreg_erank_loss  / max(1, k_layers)

        return stacked_stu_text_reps, stu_img_final_reps, sigreg_final

    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']

        batch_size = student_qry_input['input_ids'].size(0)
        self.counter += batch_size

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device
        dtype = student_qry_reps.dtype

        teacher_qry, teacher_pos = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"]

        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry], dim=0).to(device, dtype=dtype)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos], dim=0).to(device, dtype=dtype)

        tea_img_qry_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_img_token'] is not None else None
        tea_img_pos_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_img_token'] is not None else None

        tea_text_qry_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_text_token'] is not None else None
        tea_text_pos_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_text_token'] is not None else None

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

        kd_simcse = 0.0
        last_stu_qry_hidden_state = pooling(student_qry_hidden_states[-1],
                                            student_qry_input['attention_mask'], 
                                            mode='eos', normalize=True)
        last_stu_pos_hidden_state = pooling(student_pos_hidden_states[-1],
                                            student_pos_input['attention_mask'], 
                                            mode='eos', normalize=True)

        kd_simcse += self.distillcse_kd_loss(last_stu_qry_hidden_state,
                                             last_stu_pos_hidden_state,
                                             teacher_qry_reps, teacher_pos_reps,
                                             tau=self.args.d_cse_temperature)

        # all_stu_reps = torch.cat([last_stu_qry_hidden_state, last_stu_pos_hidden_state], dim=0)
        # all_tea_reps = torch.cat([teacher_qry_reps, teacher_pos_reps], dim=0)
        # kd_simcse += self.structure_loss(all_stu_reps, all_tea_reps) / self.args.d_cse_temperature

        ##################################
        student_special_ids = torch.tensor(
            list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids)
                 - set([student_tokenizer.eos_token_id])),
            device=student_qry_input['input_ids'].device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        # Trích xuất Representations từ QRY
        qry_stu_txt, qry_stu_img, qry_sigreg = self._compute_modality_distill(
            student_hidden_states=student_qry_hidden_states,
            image_features=student_qry_image_features,
            text_token_counts=num_student_text_qry_tokens,
            attention_mask=student_qry_input['attention_mask'], 
        )

        # Trích xuất Representations từ POS
        pos_stu_txt, pos_stu_img, pos_sigreg = self._compute_modality_distill(
            student_hidden_states=student_pos_hidden_states,
            image_features=student_pos_image_features,
            text_token_counts=num_student_text_pos_tokens,
            attention_mask=student_pos_input['attention_mask'], 
        )

        stu_modality_features = []
        tea_modality_features = []
        SIGReg = torch.zeros_like(contrastive_loss)
        num_sigreg_components = 0

        if tea_text_qry_reps is not None:
            stu_modality_features.append(qry_stu_txt)
            tea_modality_features.append(tea_text_qry_reps)
        if tea_text_pos_reps is not None:
            stu_modality_features.append(pos_stu_txt)
            tea_modality_features.append(tea_text_pos_reps)

        if tea_img_qry_reps is not None and qry_stu_img is not None:
            stu_modality_features.append(qry_stu_img)
            tea_modality_features.append(tea_img_qry_reps)
            SIGReg += qry_sigreg
            num_sigreg_components += 1

        if tea_img_pos_reps is not None and pos_stu_img is not None:
            stu_modality_features.append(pos_stu_img)
            tea_modality_features.append(tea_img_pos_reps)
            SIGReg += pos_sigreg
            num_sigreg_components += 1

        if num_sigreg_components > 0:
            SIGReg = SIGReg / num_sigreg_components

        modality_loss = torch.zeros_like(contrastive_loss)
        if len(stu_modality_features) > 0:
            all_stu_modality = torch.cat(stu_modality_features, dim=0)
            all_tea_modality = torch.cat(tea_modality_features, dim=0)
            modality_loss = self.structure_loss(all_stu_modality, all_tea_modality)

        # ==============================================================

        loss_distill = torch.zeros_like(contrastive_loss)
        if self.args.use_distill_cse_loss:
            loss_distill += kd_simcse

        if self.args.use_distill_vison_loss:
            loss_distill += modality_loss

        loss = contrastive_loss
        if self.args.use_distill_loss:
            loss = loss + self.kd_weight * loss_distill
        if self.args.use_sigreg_loss:
            loss = loss + self.args.sigreg_weight * SIGReg

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill,
            'kd_loss_simcse': kd_simcse,
            'sigreg_loss': SIGReg
        }
