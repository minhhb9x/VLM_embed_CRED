import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from src.model.llava.constants import IMAGE_TOKEN_INDEX
import random
import math
from torch.nn.utils.rnn import pad_sequence


def student_token_masks(inputs, image_features, hidden_len, special_ids, config, backbone):
    """Valid/text/image masks aligned with the student's actual hidden sequence.

    FastVLM expands one image placeholder; Qwen2-VL/OneVision generally have
    image token positions already expanded in input_ids.
    """
    ids = inputs["input_ids"]
    valid = inputs["attention_mask"].bool()
    fastvlm = backbone in ("llava_qwen2", "llava_qwen2_old")
    image_id = IMAGE_TOKEN_INDEX if fastvlm else getattr(config, "image_token_id", None)
    if image_id is None:
        image_id = getattr(config, "image_token_index", None)
    input_image = (ids == image_id) & valid if image_id is not None else torch.zeros_like(valid)

    if not fastvlm or not input_image.any():
        if ids.size(1) != hidden_len:
            raise ValueError("This backbone needs an explicit mapping from input IDs to expanded hidden states.")
        if image_id is None and image_features is not None and any(
            feature is not None and feature.size(0) > 0 for feature in image_features
        ):
            raise ValueError("Cannot identify image tokens: config needs image_token_id or image_token_index.")
        text = valid & ids.ge(0) & ~input_image & ~torch.isin(ids, special_ids)
        return valid, text, input_image

    counts = input_image.sum(dim=1).tolist()
    if any(n > 1 for n in counts):
        raise ValueError("FastVLM mixed-batch helper supports at most one image per input.")
    has_image = [n == 1 for n in counts]
    features = [] if image_features is None else list(image_features)
    if len(features) != len(has_image):
        if len(features) != sum(has_image):
            raise ValueError("Image feature count does not match image placeholders.")
        feature_iter = iter(features)
        features = [next(feature_iter) if flag else None for flag in has_image]
    for feature, flag in zip(features, has_image):
        if (feature is not None and feature.size(0) > 0) != flag:
            raise ValueError("Image features are not aligned with input samples.")

    rows = []
    max_len = getattr(config, "tokenizer_model_max_length", None)
    for i, feature in enumerate(features):
        sample_ids = ids[i][valid[i]]
        kinds = (sample_ids.ge(0) & ~torch.isin(sample_ids, special_ids)).long()
        repeats = torch.ones_like(sample_ids)
        if has_image[i]:
            image_positions = sample_ids == image_id
            kinds[image_positions] = 2
            repeats[image_positions] = feature.size(0)
        kinds = torch.repeat_interleave(kinds, repeats)
        rows.append(kinds[:max_len] if max_len is not None else kinds)
    if not rows or max(row.numel() for row in rows) != hidden_len:
        raise ValueError("Reconstructed FastVLM layout does not match hidden-state length.")
    left = getattr(config, "tokenizer_padding_side", "right") == "left"
    layout = torch.stack([
        F.pad(row, (hidden_len - row.numel(), 0) if left else (0, hidden_len - row.numel()), value=-1)
        for row in rows
    ])
    return layout >= 0, layout == 1, layout == 2


def pool_last_valid(hidden, valid_mask, normalize=True):
    positions = torch.arange(hidden.size(1), device=hidden.device)[None, :]
    last = positions.expand_as(valid_mask).masked_fill(~valid_mask.bool(), -1).amax(dim=1)
    if (last < 0).any():
        raise ValueError("Cannot pool a sequence without valid tokens.")
    reps = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
    return F.normalize(reps, p=2, dim=-1) if normalize else reps


def matched_cached_modality(student_reps, caches, key, sample_indices):
    """Filter optional teacher means using exactly the same student sample rows."""
    if student_reps is None:
        return None, None
    if student_reps.size(0) != len(sample_indices):
        raise ValueError("Student modality rows and original sample indices do not match.")
    pairs = [(row, caches[i].get(key)) for row, i in enumerate(sample_indices)
             if caches[i].get(key) is not None]
    if not pairs:
        return None, None
    rows, teacher_reps = zip(*pairs)
    return student_reps[list(rows)], torch.stack(teacher_reps).to(
        device=student_reps.device, dtype=student_reps.dtype
    )


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
        if self.cpr_statistic not in ("pr", "erank", "var_cov"):
            raise ValueError("cpr_statistic must be pr, erank, or var_cov.")
        if self.cpr_threshold is not None and (
            not math.isfinite(self.cpr_threshold) or self.cpr_threshold < 0
        ):
            raise ValueError("cpr_threshold must be finite and nonnegative.")
        self.cpr_var_target = getattr(args, "cpr_var_target", 1.0)
        self.cpr_var_weight = getattr(args, "cpr_var_weight", 25.0)
        self.cpr_cov_weight = getattr(args, "cpr_cov_weight", 1.0)
        if self.cpr_statistic == "var_cov":
            if self.cpr_threshold is not None:
                raise ValueError("var_cov uses cpr_var_target, not cpr_threshold; omit cpr_threshold.")
            if any(not math.isfinite(x) or x < 0 for x in (
                self.cpr_var_target, self.cpr_var_weight, self.cpr_cov_weight
            )):
                raise ValueError("Variance target and variance/covariance weights must be finite and nonnegative.")

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

    def variance_covariance_loss(self, z_list, min_valid_tokens=4, eps=1e-4):
        """VICReg-style V+C, adapted to tokens within each image (not across images)."""
        if not z_list:
            return 0.0
        device, dtype = z_list[0].device, z_list[0].dtype
        with torch.autocast(device_type=device.type, enabled=False):
            lengths = torch.tensor([z.size(0) for z in z_list], device=device)
            z = pad_sequence(z_list, batch_first=True).float()
            valid = lengths >= min_valid_tokens
            if not valid.any():
                return z.sum() * 0.0
            mask = (torch.arange(z.size(1), device=device)[None, :] < lengths[:, None])
            mask = mask.unsqueeze(-1).float()
            mean = (z * mask).sum(dim=1, keepdim=True) / lengths.clamp_min(1)[:, None, None]
            centered = (z - mean) * mask
            covariance = (centered.transpose(1, 2) @ centered) / (lengths - 1).clamp_min(1)[:, None, None]
            variance = covariance.diagonal(dim1=-2, dim2=-1)
            var_loss = F.relu(self.cpr_var_target - (variance + eps).sqrt()).mean(dim=-1)
            off_diagonal = covariance - torch.diag_embed(variance)
            cov_loss = off_diagonal.square().sum(dim=(-2, -1)) / z.size(-1)
            loss = self.cpr_var_weight * var_loss + self.cpr_cov_weight * cov_loss
            return loss[valid].mean().to(dtype)

    def sketched_participation_ratio_erank(self, z_list_first: list[torch.Tensor],
                                           z_list_last: list[torch.Tensor],
                                           num_slices: int = 256,
                                           min_valid_tokens: int = 4, eps: float = 1e-8):

        B = len(z_list_last)
        if B == 0:
            return 0.0

        if self.cpr_statistic == "var_cov":
            return self.variance_covariance_loss(z_list_last, min_valid_tokens)

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
                                  text_token_counts, attention_mask, text_mask, image_mask):
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

        text_indices, image_indices = [], []
        for i in range(batch_size):
            text_hidden = student_hidden_states[last_layer_idx][i][text_mask[i]]
            if text_hidden.size(0) > 0:
                stu_text_reps.append(text_hidden.mean(dim=0))
                text_indices.append(i)
            if image_mask[i].any():
                image_indices.append(i)
                for l in layers:
                    stu_img_tokens[l].append(student_hidden_states[l][i][image_mask[i]])

        stacked_stu_text_reps = torch.stack(stu_text_reps) if stu_text_reps else None

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

        return stacked_stu_text_reps, stu_img_final_reps, sigreg_final, text_indices, image_indices

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

        eos_ids = getattr(student_tokenizer, "eos_token_id", None)
        eos_ids = [] if eos_ids is None else eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        special_ids = (set(getattr(student_tokenizer, "all_special_ids", []))
                       | set(getattr(student_tokenizer, "added_tokens_encoder", {}).values())) - set(eos_ids)
        student_special_ids = torch.tensor(sorted(special_ids), device=device, dtype=torch.long)
        backbone = getattr(student_model, "model_backbone", getattr(self.args, "model_backbone", None))
        qry_mask, qry_text_mask, qry_image_mask = student_token_masks(
            student_qry_input, student_qry_image_features, student_qry_hidden_states[-1].size(1),
            student_special_ids, student_model.config, backbone,
        )
        pos_mask, pos_text_mask, pos_image_mask = student_token_masks(
            student_pos_input, student_pos_image_features, student_pos_hidden_states[-1].size(1),
            student_special_ids, student_model.config, backbone,
        )
        # Original FastVLM attention masks have lengths before image expansion.
        if backbone in ("llava_qwen2", "llava_qwen2_old") and getattr(student_model, "pooling", "eos") in ("eos", "last"):
            student_qry_reps = pool_last_valid(student_qry_hidden_states[-1], qry_mask, student_model.normalize)
            student_pos_reps = pool_last_valid(student_pos_hidden_states[-1], pos_mask, student_model.normalize)

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

        kd_simcse = 0.0
        last_stu_qry_hidden_state = pool_last_valid(student_qry_hidden_states[-1], qry_mask)
        last_stu_pos_hidden_state = pool_last_valid(student_pos_hidden_states[-1], pos_mask)

        kd_simcse += self.distillcse_kd_loss(last_stu_qry_hidden_state,
                                             last_stu_pos_hidden_state,
                                             teacher_qry_reps, teacher_pos_reps,
                                             tau=self.args.d_cse_temperature)

        # all_stu_reps = torch.cat([last_stu_qry_hidden_state, last_stu_pos_hidden_state], dim=0)
        # all_tea_reps = torch.cat([teacher_qry_reps, teacher_pos_reps], dim=0)
        # kd_simcse += self.structure_loss(all_stu_reps, all_tea_reps) / self.args.d_cse_temperature

        ##################################
        num_student_text_qry_tokens = qry_text_mask.sum(dim=1)
        num_student_text_pos_tokens = pos_text_mask.sum(dim=1)

        # Trích xuất Representations từ QRY
        qry_stu_txt, qry_stu_img, qry_sigreg, qry_text_indices, qry_image_indices = self._compute_modality_distill(
            student_hidden_states=student_qry_hidden_states,
            image_features=student_qry_image_features,
            text_token_counts=num_student_text_qry_tokens,
            attention_mask=qry_mask, text_mask=qry_text_mask, image_mask=qry_image_mask,
        )

        # Trích xuất Representations từ POS
        pos_stu_txt, pos_stu_img, pos_sigreg, pos_text_indices, pos_image_indices = self._compute_modality_distill(
            student_hidden_states=student_pos_hidden_states,
            image_features=student_pos_image_features,
            text_token_counts=num_student_text_pos_tokens,
            attention_mask=pos_mask, text_mask=pos_text_mask, image_mask=pos_image_mask,
        )

        stu_modality_features = []
        tea_modality_features = []
        SIGReg = torch.zeros_like(contrastive_loss)
        num_sigreg_components = 0

        for student_reps, caches, key, indices in [
            (qry_stu_txt, teacher_qry, 'mean_last_text_token', qry_text_indices),
            (pos_stu_txt, teacher_pos, 'mean_last_text_token', pos_text_indices),
            (qry_stu_img, teacher_qry, 'mean_last_img_token', qry_image_indices),
            (pos_stu_img, teacher_pos, 'mean_last_img_token', pos_image_indices),
        ]:
            student_reps, teacher_reps = matched_cached_modality(student_reps, caches, key, indices)
            if student_reps is not None:
                stu_modality_features.append(student_reps)
                tea_modality_features.append(teacher_reps)

        # CPR is student-side and must not depend on teacher modality availability.
        if qry_stu_img is not None:
            SIGReg += qry_sigreg
            num_sigreg_components += 1
        if pos_stu_img is not None:
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
