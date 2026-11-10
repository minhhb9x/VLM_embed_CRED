import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text_vision, pooling
import random
import math
from torch.nn.utils.rnn import pad_sequence


def fastvlm_token_masks(inputs, image_features, hidden_len, special_ids, config):
    """Rebuild the token layout for FastVLM inputs with zero or one image."""
    input_ids, input_mask = inputs["input_ids"], inputs["attention_mask"].bool()
    image_counts = ((input_ids == -200) & input_mask).sum(dim=1).tolist()
    if any(n > 1 for n in image_counts):
        raise ValueError("This JEPA helper supports at most one image per input.")
    has_image = [n == 1 for n in image_counts]
    features = [] if image_features is None else list(image_features)
    if len(features) != len(has_image):
        if len(features) != sum(has_image):
            raise ValueError("Image feature count does not match image placeholders.")
        image_iter = iter(features)
        features = [next(image_iter) if flag else None for flag in has_image]
    if any((feat is not None) != flag for feat, flag in zip(features, has_image)):
        raise ValueError("Image features are not aligned with input samples.")

    rows = []
    max_len = getattr(config, "tokenizer_model_max_length", None)
    for i, feat in enumerate(features):
        ids = input_ids[i][input_mask[i]]
        # 0: special token, 1: clean text (including EOS), 2: image.
        kinds = ((ids >= 0) & ~torch.isin(ids, special_ids)).long()
        repeats = torch.ones_like(ids)
        image_positions = ids == -200
        if feat is not None:
            kinds[image_positions] = 2
            repeats[image_positions] = feat.size(0)
        kinds = torch.repeat_interleave(kinds, repeats)
        if max_len is not None:
            kinds = kinds[:max_len]
        rows.append(kinds)
    if not rows or max(row.numel() for row in rows) != hidden_len:
        raise ValueError("Reconstructed token layout does not match FastVLM hidden states.")

    left_padding = getattr(config, "tokenizer_padding_side", "right") == "left"
    layout = torch.stack([
        F.pad(row, (hidden_len - row.numel(), 0) if left_padding
              else (0, hidden_len - row.numel()), value=-1)
        for row in rows
    ])
    return features, layout >= 0, layout == 1, layout == 2


def matched_cached_modality(student_reps, caches, key, sample_indices):
    """Select identical sample rows from student reps and optional cached reps."""
    if student_reps is None:
        return None, None
    pairs = [(row, caches[i].get(key)) for row, i in enumerate(sample_indices)
             if caches[i].get(key) is not None]
    if not pairs:
        return None, None
    rows, teacher_reps = zip(*pairs)
    return student_reps[list(rows)], torch.stack(teacher_reps).to(
        device=student_reps.device, dtype=student_reps.dtype
    )



class TalasJepa(nn.Module):
    def __init__(self, args):
        super(TalasJepa, self).__init__()
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
    
    def sigreg(self, x: torch.Tensor, num_slices: int = 128) -> torch.Tensor:
        device = x.device
        # =====================================================
        # 1. Random projection seed
        #
        # Chỉ rank 0 sinh seed.
        # Sau đó broadcast để tất cả GPU dùng cùng seed.
        # =====================================================
        if self.process_rank == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0

        if self.world_size > 1:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device,)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = seed_tensor.item()

        # =====================================================
        # 2. Local generator
        # =====================================================
        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)

        A = torch.randn(x.size(1), num_slices, generator=g,  device=device, dtype=x.dtype,)

        A = A / A.norm(p=2, dim=0, keepdim=True, ).clamp_min(1e-12)

        # =====================================================
        # 3. Epps-Pulley statistic
        # =====================================================
        t = torch.linspace(-5, 5, 17, device=device, dtype=x.dtype,)

        exp_f = torch.exp(-0.5 * t.square())

        # x:   [N, K]
        # A:   [K, M]
        # x@A: [N, M]
        # x_t: [N, M, T]
        x_t = (x @ A).unsqueeze(-1) * t

        # [M, T]
        ecf = torch.exp(1j * x_t).mean(dim=0)

        # =====================================================
        # 4. Aggregate across GPUs
        # =====================================================
        if self.world_size > 1:
            dist.all_reduce(ecf, op=dist.ReduceOp.SUM,)
            ecf = ecf / self.world_size

        # =====================================================
        # 5. Weighted L2 distance
        # =====================================================
        err = ((ecf - exp_f).abs().square().mul(exp_f))

        global_batch_size = x.size(0) * self.world_size

        sigreg_per_slice = (torch.trapezoid(err, t, dim=1,) * global_batch_size)

        return sigreg_per_slice.mean()

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

        A = torch.randn(D, num_slices, device=device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(eps)   # [D, M]

        def _participation_ratio(z_padded, mask, lengths):
            mask_f = mask.unsqueeze(-1).float()                                        # [B, N_max, 1]
            n_valid = lengths.clamp(min=2).float()                                     # [B]

            # Center (giống bước 0 cũ) — bắt buộc để proj mang đúng ý nghĩa "centered"
            mean = (z_padded * mask_f).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
            z_c = (z_padded - mean) * mask_f                                           # [B, N_max, D], padding = 0

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

        pr0, valid0 = _participation_ratio(z0_normed, mask0, len0)
        prL, validL = _participation_ratio(zL_normed, maskL, lenL)

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

    def sigreg_dualview(self, z_list: list[torch.Tensor], eos_query: torch.Tensor,
                        num_slices: int = 256, tau: float = 0.05, alpha: float = 0.9):
        B = len(z_list)
        if B == 0:
            return 0.0

        device, dtype = z_list[0].device, z_list[0].dtype
        D = z_list[0].shape[-1]

        # ==========================================
        # 0. PADDING & MASK
        # ==========================================
        lengths = torch.tensor([x.size(0) for x in z_list], device=device)
        N_max = lengths.max().item()
        z_padded = pad_sequence(z_list, batch_first=True, padding_value=0.0)  # [B, N_max, D]

        idx = torch.arange(N_max, device=device).unsqueeze(0)   # [1, N_max]
        mask = idx < lengths.view(B, 1)                          # [B, N_max]

        # ==========================================
        # 1. VIEW 1: ATTENTION-WEIGHTED POOLING
        # ==========================================
        q = F.normalize(eos_query.to(device=device, dtype=dtype), p=2, dim=-1)   # [B, D]
        k = F.normalize(z_padded, p=2, dim=-1)                                   # [B, N_max, D]

        score = torch.einsum('bd,bnd->bn', q, k) / tau                           # [B, N_max]
        score = score.masked_fill(~mask, -float('inf'))
        attn_w = torch.softmax(score, dim=-1)                                    # [B, N_max]

        attn_view = torch.einsum('bn,bnd->bd', attn_w, z_padded)                 # [B, D]

        z_k_concepts = attn_view.unsqueeze(0)

        if alpha < 1.0:
            noise = torch.randn_like(z_k_concepts)
            z_mixed = math.sqrt(alpha) * z_k_concepts + math.sqrt(1.0 - alpha) * noise
        else:
            z_mixed = z_k_concepts

        if self.process_rank == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0
        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)
        
        A = torch.randn(D, num_slices, generator=g, device=device, dtype=dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        t = torch.linspace(-5, 5, 17, device=device, dtype=dtype)
        exp_f = torch.exp(-0.5 * t.square())

        x_proj = z_mixed @ A                   # [K, B, num_slices]
        x_t = x_proj.unsqueeze(-1) * t         # [K, B, num_slices, 17]

        ecf_real = torch.cos(x_t).mean(dim=1)  # [K, num_slices, 17]
        ecf_imag = torch.sin(x_t).mean(dim=1)  # [K, num_slices, 17]

        err = ((ecf_real - exp_f).square() + ecf_imag.square()).mul(exp_f)

        loss_sigreg = torch.trapezoid(err, t, dim=-1).mean(dim=-1) * B

        return loss_sigreg.mean()

    def _compute_modality_distill(self, student_hidden_states, image_features, 
                                  text_token_counts, attention_mask, text_mask=None, image_mask=None):
        """
        Hàm này chỉ còn nhiệm vụ trích xuất text và vision representations 
        của student, cùng với việc tính toán SIGReg loss.
        """

        batch_size = attention_mask.size(0)
        last_layer_idx = len(student_hidden_states) - 1

        layer_mapping = self.args.student_layer_mapping
        if 0 in layer_mapping and last_layer_idx in layer_mapping:
            layers = layer_mapping
        else:
            layers = [0, int(last_layer_idx / 2), int(4 * last_layer_idx / 5), last_layer_idx]
        
        stu_img_tokens = {l: [] for l in layers}
        stu_text_reps = []
        
        cur_idx_img = 0
        for i in range(batch_size):
            if image_mask is not None:
                text_last_hidden = student_hidden_states[last_layer_idx][i][text_mask[i]]
                num_vision_token = int(image_mask[i].sum().item())
            else:
                num_vision_token = 0
                if image_features is not None and cur_idx_img < len(image_features):
                    num_vision_token = image_features[cur_idx_img].size(0)
                    cur_idx_img += 1
                text_last_hidden, _ = get_hidden_text_vision(
                    student_hidden_states[last_layer_idx][i],
                    text_token_counts[i].item(), num_vision_token, attention_mask[i]
                )
            stu_text_reps.append(
                text_last_hidden.mean(dim=0) if text_last_hidden.size(0)
                else student_hidden_states[last_layer_idx][i].sum(dim=0) * 0.0
            )

            if num_vision_token > 0:
                for l in layers:
                    if image_mask is not None:
                        img_hidden = student_hidden_states[l][i][image_mask[i]]
                    else:
                        _, img_hidden = get_hidden_text_vision(
                            student_hidden_states[l][i], text_token_counts[i].item(),
                            num_vision_token, attention_mask[i]
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
            for l in layers[1:-1]:
                k_layers += 1
                # eos_query = pooling(student_hidden_states[l], attention_mask, 
                #                     mode='eos', normalize=True).detach()
                # total_sigreg += self.sigreg_dualview(stu_img_tokens[l], eos_query, 
                #                                      tau=0.1, alpha=0.9)
                # total_sigreg += self.sigreg_sinkhorn(stu_img_tokens[l], concept_queries)

                sigreg_erank_loss += self.sketched_participation_ratio_erank(stu_img_tokens[0], 
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

        student_special_ids = torch.tensor(
            list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids)
                 - set([student_tokenizer.eos_token_id])),
            device=student_qry_input['input_ids'].device,
            dtype=torch.long
        )

        # FastVLM expands image placeholders, changing lengths and padding.
        qry_mask, pos_mask = student_qry_input['attention_mask'], student_pos_input['attention_mask']
        qry_text_mask = pos_text_mask = qry_image_mask = pos_image_mask = None
        if getattr(student_model, "model_backbone", None) in ["llava_qwen2", "llava_qwen2_old"]:
            student_qry_image_features, qry_mask, qry_text_mask, qry_image_mask = fastvlm_token_masks(
                student_qry_input, student_qry_image_features,
                student_qry_hidden_states[-1].size(1), student_special_ids, student_model.config
            )
            student_pos_image_features, pos_mask, pos_text_mask, pos_image_mask = fastvlm_token_masks(
                student_pos_input, student_pos_image_features,
                student_pos_hidden_states[-1].size(1), student_special_ids, student_model.config
            )
            student_qry_reps = pooling(student_qry_hidden_states[-1], qry_mask,
                                       mode='eos', normalize=student_model.normalize)
            student_pos_reps = pooling(student_pos_hidden_states[-1], pos_mask,
                                       mode='eos', normalize=student_model.normalize)


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
        last_stu_qry_hidden_state = pooling(student_qry_hidden_states[-1], 
                                            qry_mask,
                                            mode='eos', normalize=True)
        last_stu_pos_hidden_state = pooling(student_pos_hidden_states[-1], 
                                            pos_mask,
                                            mode='eos', normalize=True)
        
        kd_simcse += self.distillcse_kd_loss(last_stu_qry_hidden_state, 
                                             last_stu_pos_hidden_state, 
                                             teacher_qry_reps, teacher_pos_reps, 
                                             tau=self.args.d_cse_temperature)

        # all_stu_reps = torch.cat([last_stu_qry_hidden_state, last_stu_pos_hidden_state], dim=0)
        # all_tea_reps = torch.cat([teacher_qry_reps, teacher_pos_reps], dim=0)
        # kd_simcse += self.structure_loss(all_stu_reps, all_tea_reps) / self.args.d_cse_temperature

        ##################################
        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        # Trích xuất Representations từ QRY
        qry_stu_txt, qry_stu_img, qry_sigreg = self._compute_modality_distill(
            student_hidden_states=student_qry_hidden_states, 
            image_features=student_qry_image_features,
            text_token_counts=num_student_text_qry_tokens, 
            attention_mask=qry_mask, text_mask=qry_text_mask, image_mask=qry_image_mask,
        )

        # Trích xuất Representations từ POS
        pos_stu_txt, pos_stu_img, pos_sigreg = self._compute_modality_distill(
            student_hidden_states=student_pos_hidden_states, 
            image_features=student_pos_image_features,
            text_token_counts=num_student_text_pos_tokens, 
            attention_mask=pos_mask, text_mask=pos_text_mask, image_mask=pos_image_mask,
        )

        stu_modality_features = []
        tea_modality_features = []
        SIGReg = torch.zeros_like(contrastive_loss)
        num_sigreg_components = 0

        qry_image_indices = (qry_image_mask.any(dim=1).nonzero(as_tuple=True)[0].tolist()
                             if qry_image_mask is not None else
                             [i for i, rep in enumerate(teacher_qry)
                              if rep.get('mean_last_img_token') is not None])
        pos_image_indices = (pos_image_mask.any(dim=1).nonzero(as_tuple=True)[0].tolist()
                             if pos_image_mask is not None else
                             [i for i, rep in enumerate(teacher_pos)
                              if rep.get('mean_last_img_token') is not None])
        for student_reps, caches, key, indices in [
            (qry_stu_txt, teacher_qry, 'mean_last_text_token', range(batch_size)),
            (pos_stu_txt, teacher_pos, 'mean_last_text_token', range(batch_size)),
            (qry_stu_img, teacher_qry, 'mean_last_img_token', qry_image_indices),
            (pos_stu_img, teacher_pos, 'mean_last_img_token', pos_image_indices),
        ]:
            student_reps, teacher_reps = matched_cached_modality(student_reps, caches, key, indices)
            if student_reps is not None:
                stu_modality_features.append(student_reps)
                tea_modality_features.append(teacher_reps)

        # CPR depends on student image tokens, including when cache modality means are absent.
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