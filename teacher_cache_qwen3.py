import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from tqdm import tqdm
from transformers import HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.single_wrapper import SingleCollator, SingleDataset
from src.utils import print_rank


def seed_everything(seed: int, rank: int = 0):
    seed = seed + rank
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def ddp_setup():
    if "LOCAL_RANK" not in os.environ:
        return
    local_rank = get_local_rank()
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
            "Lower --nproc_per_node or set CUDA_VISIBLE_DEVICES to expose more GPUs."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()


def to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def release_memory(device):
    gc.collect()
    if torch.cuda.is_available():
        if device is not None and torch.device(device).type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() 
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def prepare_dataset(data_args, model_args):
    return SingleDataset(data_args, model_args)


def mean_or_none(hidden_tokens):
    if hidden_tokens.numel() == 0:
        return None
    return hidden_tokens.float().mean(dim=0).cpu()

def exact_participation_ratio(tokens, eps=1e-12, dtype=torch.float64):
    """
    PR chính xác của các token: PR = tr(C)^2 / tr(C^2), tokens: [n, D].
    Dùng Gram matrix G = Xc Xc^T [n, n] (n = số token ảnh, thường << D):
        tr(C)   = tr(G)   / (n-1)
        tr(C^2) = ||G||_F^2 / (n-1)^2
    => PR = tr(G)^2 / ||G||_F^2  (hệ số n-1 triệt tiêu).
    Trả về tensor scalar (NaN nếu n < 2 hoặc các token trùng nhau hoàn toàn).
    """
    n = tokens.size(0)
    if n < 2:
        return torch.tensor(float("nan"), device=tokens.device)
    x = tokens.to(dtype)
    x = x - x.mean(dim=0, keepdim=True)
    gram = x @ x.T                                   # [n, n]
    tr1 = torch.diagonal(gram).sum()
    tr2 = (gram ** 2).sum()
    if tr2 <= eps:
        return torch.tensor(float("nan"), device=tokens.device)
    pr = (tr1 ** 2 / tr2).float()
    pr = pr / min(n - 1, tokens.size(1))
    return pr


def compute_image_pr_all_layers(hidden_states, image_mask, idx):
    num_image_tokens = int(image_mask.sum().item())

    if num_image_tokens < 2:
        return None

    prs = []

    for layer_hs in hidden_states:
        image_hidden = layer_hs[idx][image_mask, :]
        prs.append(exact_participation_ratio(image_hidden))

    return torch.stack(prs).cpu()


def build_cache_obj(rep, inputs, hidden_states, processor, idx):
    image_token_id = processor.image_token_id
    sample_ids = inputs["input_ids"][idx]
    valid_mask = inputs["attention_mask"][idx].bool()

    image_mask_full = sample_ids.eq(image_token_id) & valid_mask

    token_mask_full = sample_ids.ne(image_token_id) & valid_mask

    last_hidden_state = hidden_states[-1][idx]

    # print(f"Sample {idx}: num_image_tokens={image_mask_full.sum().item()}, num_text_tokens={token_mask_full.sum().item()}")

    # print(f"sample_ids: {sample_ids.tolist()}")

    # decoded = processor.tokenizer.decode(sample_ids, skip_special_tokens=False)
    
    # print(f"decoded: {decoded}")

    assert last_hidden_state.size(0) == sample_ids.size(0), \
        "Mismatch between hidden states and input_ids"
    
    image_hidden = last_hidden_state[image_mask_full, :]
    text_hidden = last_hidden_state[token_mask_full, :]

    return to_device({
        "rep": rep[idx].float(),
        "mean_last_img_token": mean_or_none(image_hidden),
        "mean_last_text_token": mean_or_none(text_hidden),
        "pr_img_layers": compute_image_pr_all_layers(
            hidden_states, image_mask_full, idx
        ),
    }, "cpu")


class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TeacherCacheRunner:
    def __init__(self, teacher, train_data, model_args, training_args):
        self.device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        self.train_data = train_data
        self.model_args = model_args
        self.training_args = training_args

    @torch.inference_mode()
    def run_once(self):
        progress_bar = tqdm(
            total=len(self.train_data),
            desc="Teacher cache",
            disable=not is_main_process(),
        )

        device = self.device
        processor = load_processor(self.model_args, None)

        for batch in self.train_data:
            encoded_dirs = batch.get("encoded_dir")
            if encoded_dirs is None:
                raise KeyError("Batch is missing encoded_dir. Make sure SingleDataset.__getitem__ and SingleCollator return it.")
            
            skip_batch = True
            for encoded_dir in encoded_dirs:
                output_qry_dir = os.path.join(self.training_args.output_dir, encoded_dir, "qry.pt")
                output_pos_dir = os.path.join(self.training_args.output_dir, encoded_dir, "pos.pt")
                
                if not (os.path.exists(output_qry_dir) and os.path.exists(output_pos_dir)):
                    skip_batch = False
                    break
            
            if skip_batch:
                progress_bar.update(1)
                continue

            batch = to_device(batch, self.device)

            with torch.no_grad():
                qry_output = self.teacher.encode_input(batch["qry"])
                pos_output = self.teacher.encode_input(batch["pos"])

            qry_reps, _, _, qry_hidden_states = qry_output
            pos_reps, _, _, pos_hidden_states = pos_output

            if len(encoded_dirs) != qry_reps.size(0) or len(encoded_dirs) != pos_reps.size(0):
                raise ValueError(
                    f"encoded_dir count ({len(encoded_dirs)}) does not match encoded batch size "
                    f"(qry={qry_reps.size(0)}, pos={pos_reps.size(0)})."
                )

            for i, encoded_dir in enumerate(encoded_dirs):
                output_qry_dir = os.path.join(self.training_args.output_dir, encoded_dir, "qry.pt")
                output_pos_dir = os.path.join(self.training_args.output_dir, encoded_dir, "pos.pt")

                if os.path.exists(output_qry_dir) and os.path.exists(output_pos_dir):
                    continue

                os.makedirs(os.path.dirname(output_qry_dir), exist_ok=True)
                os.makedirs(os.path.dirname(output_pos_dir), exist_ok=True)

                qry_obj = build_cache_obj(
                    qry_reps,
                    batch["qry"],
                    qry_hidden_states,
                    processor,
                    i,
                )
                pos_obj = build_cache_obj(
                    pos_reps,
                    batch["pos"],
                    pos_hidden_states,
                    processor,
                    i,
                )

                torch.save(qry_obj, output_qry_dir)
                torch.save(pos_obj, output_pos_dir)

            progress_bar.update(1)
            del qry_reps
            del pos_reps
            del qry_hidden_states
            del pos_hidden_states
            del batch
            release_memory(self.device)

        progress_bar.close()
        return

def build_dataloader(model_args, data_args, training_args):
    train_dataset = prepare_dataset(data_args, model_args)

    if is_dist():
        sampler = DistributedSequentialSampler(
            train_dataset,
            rank=get_rank(),
            world_size=get_world_size(),
        )
    else:
        sampler = SequentialSampler(train_dataset)

    collator = SingleCollator(
        processor=load_processor(model_args, None),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    return DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=False,
        pin_memory=False,
    )


def normalize_torchrun_args():
    for arg in list(sys.argv):
        if arg.startswith("--local_rank="):
            local_rank = arg.split("=", 1)[1]
            sys.argv.remove(arg)
            sys.argv.extend(["--local_rank", local_rank])


def main():
    normalize_torchrun_args()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    seed_everything(training_args.seed, rank=get_rank())

    teacher = MMEBModel.load(model_args, is_trainable=False)
    for n, p in teacher.named_parameters():
        p.requires_grad = False
        
    train_dataloader = build_dataloader(model_args, data_args, training_args)

    print_rank(f"Loaded {len(train_dataloader.dataset)} training samples")
    print_rank(f"Running teacher inference over {len(train_dataloader)} batches")

    runner = TeacherCacheRunner(teacher, train_dataloader, model_args, training_args)
    os.makedirs(training_args.output_dir, exist_ok=True)
    runner.run_once()

    
    if is_dist():
        dist.barrier()


@record
def entrypoint():
    ddp_setup()
    try:
        main()
    finally:
        cleanup_dist()


if __name__ == "__main__":
    entrypoint()
