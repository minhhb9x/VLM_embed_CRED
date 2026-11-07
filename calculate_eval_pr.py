"""Compute absolute image-token PR per sample, then average at each layer.

Copy into the root of VLM_Embed_Gvendi and run with its Python environment.
Layer 0 is the input to the language backbone; subsequent indices are its
transformer outputs. Uses centered, exact PR in FP32 without rank normalization.
"""

import csv
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import HfArgumentParser


@dataclass
class PRArguments:
    output_dir: str = "pr_eval"
    batch_size: int = 4
    device: str = "cuda"
    layers: Optional[List[int]] = None
    side: str = field(default="qry", metadata={"choices": ["qry", "tgt"]})
    max_samples: int = 0  # 0 means all image-bearing inputs in each subset.


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


def absolute_pr(tokens):
    tokens = tokens.float()
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    token_count, hidden_dim = centered.shape
    if token_count <= hidden_dim:
        covariance = centered @ centered.T
    else:
        covariance = centered.T @ centered
    covariance = covariance / (token_count - 1)
    trace = covariance.diagonal().sum()
    return (trace.square() / covariance.square().sum().clamp_min(1e-8)).item()


def image_positions(inputs, output, index, backbone, config):
    ids = inputs["input_ids"][index]
    valid = inputs["attention_mask"][index].bool()
    sequence_length = output.hidden_states[0].size(1)
    if backbone in ("llava_qwen2", "llava_qwen2_old"):
        # FastVLM expands one -200 placeholder into a block of image tokens.
        valid_ids = ids[valid]
        markers = valid_ids.eq(-200).nonzero(as_tuple=True)[0]
        if markers.numel() != 1:
            raise ValueError("Expected one image placeholder per FastVLM input.")
        image_count = output.batch_image_embeds[index].size(0)
        expanded_length = valid_ids.numel() - 1 + image_count
        max_length = getattr(config, "tokenizer_model_max_length", None)
        if max_length is not None and expanded_length > max_length:
            raise ValueError("Input was truncated; increase tokenizer_model_max_length.")
        offset = 0
        if getattr(config, "tokenizer_padding_side", "right") == "left":
            offset = sequence_length - expanded_length
        start = offset + markers.item()
        if start < 0 or start + image_count > sequence_length:
            raise ValueError("Image-token positions exceed the hidden-state length.")
        return torch.arange(start, start + image_count, device=ids.device)

    # OneVision and Qwen VL expose expanded image-token IDs in input_ids.
    image_id = getattr(config, "image_token_index", None)
    if image_id is None:
        image_id = getattr(config, "image_token_id", None)
    if image_id is None or sequence_length != ids.numel():
        raise ValueError(f"Image-token alignment is not supported for {backbone}.")
    return (ids.eq(image_id) & valid).nonzero(as_tuple=True)[0]


@torch.inference_mode()
def main():
    from src.arguments import DataArguments, ModelArguments
    from src.data.collator.eval_collator import EvalCollator
    from src.data.dataset.mmeb_dataset import EvalDataset
    from src.model.model import MMEBModel
    from src.model.processor import load_processor

    parser = HfArgumentParser((ModelArguments, DataArguments, PRArguments))
    parser.set_defaults(dataset_name="TIGER-Lab/MMEB-eval", dataset_split="test")
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    if not data_args.subset_name:
        raise ValueError("Pass the eval subsets with --subset_name.")
    if args.batch_size < 1 or args.max_samples < 0:
        raise ValueError("batch_size must be positive and max_samples nonnegative.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    model = MMEBModel.load(model_args, is_trainable=False)
    processor = load_processor(model_args, data_args)
    model = model.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.eval()
    model.encoder.config.use_cache = False
    backbone = model_args.model_backbone
    collator = EvalCollator(data_args=data_args, model_args=model_args, processor=processor)
    totals = defaultdict(float)
    counts = defaultdict(int)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subset in dict.fromkeys(data_args.subset_name):
        dataset = EvalDataset(
            data_args=data_args, model_args=model_args, subset=subset,
            text_field=f"{args.side}_text", img_path_field=f"{args.side}_img_path",
        )
        # Keep image-bearing inputs. EvalDataset preserves unique (text, image) pairs.
        indices = [i for i, pair in enumerate(dataset.paired_data) if pair["img_path"]]
        if args.max_samples:
            indices = indices[:args.max_samples]
        if not indices:
            print(f"{subset}: no images on {args.side}; skipping.", flush=True)
            continue
        loader = DataLoader(
            Subset(dataset, indices), batch_size=args.batch_size,
            collate_fn=collator, shuffle=False, drop_last=False,
        )
        for inputs in tqdm(loader, desc=f"PR: {subset} / {args.side}"):
            inputs = to_device(inputs, device)
            inputs.pop("image_paths", None)
            inputs.pop("text", None)
            if backbone not in ("llava_qwen2", "llava_qwen2_old"):
                inputs.pop("images", None)
            if backbone in ("llava_onevision", "llava_onevision_old", "llava_next"):
                inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)
                inputs["image_sizes"] = inputs["image_sizes"].squeeze(1)
            output = model.encoder(
                **inputs, output_hidden_states=True, output_attentions=False,
                use_cache=False, return_dict=True,
            )
            layer_count = len(output.hidden_states)
            layers = list(range(layer_count)) if args.layers is None else sorted(set(args.layers))
            if not layers or any(layer < 0 or layer >= layer_count for layer in layers):
                raise ValueError(f"Choose layer indices from 0 to {layer_count - 1}.")
            for index in range(inputs["input_ids"].size(0)):
                positions = image_positions(inputs, output, index, backbone, model.encoder.config)
                if positions.numel() < 4:
                    continue
                for layer in layers:
                    value = absolute_pr(output.hidden_states[layer][index, positions])
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite PR in {subset}, layer {layer}.")
                    totals[subset, layer] += value
                    counts[subset, layer] += 1
            del output, inputs

    csv_path = output_dir / "mean_pr_by_layer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subset", "Side", "Layer", "Samples", "Mean Absolute PR"])
        for subset, layer in sorted(totals):
            count = counts[subset, layer]
            writer.writerow([subset, args.side, layer, count, totals[subset, layer] / count])
        # ALL weights every sample equally, including subsets of different sizes.
        for layer in sorted({layer for _, layer in totals}):
            total = sum(value for (subset, index), value in totals.items() if index == layer)
            count = sum(value for (subset, index), value in counts.items() if index == layer)
            writer.writerow(["ALL", args.side, layer, count, total / count])
    print(f"Saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
