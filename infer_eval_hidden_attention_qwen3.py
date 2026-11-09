import os

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import HfArgumentParser

from infer_eval_hidden_attention import (
    add_query_target_info,
    build_eval_dataset,
    build_loader,
    build_query_target_maps,
    clean_model_inputs,
    ddp_setup,
    fix_local_rank_arg,
    get_device,
    get_rank,
    is_main_process,
    load_infer_model,
    make_runtime_data_args,
    move_to_device,
)
from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.model.processor import QWEN3_VL


def _get_token_id(processor, tokenizer, attribute_name, token_text):
    token_id = getattr(processor, attribute_name, None)
    if token_id is None:
        token_id = getattr(tokenizer, attribute_name, None)
    if token_id is None:
        token_id = tokenizer.convert_tokens_to_ids(token_text)
    if token_id is None or token_id < 0:
        raise ValueError(f"Could not resolve the token id for {token_text}.")
    return int(token_id)


def _stack_valid_hidden_states(hidden_states, sample_idx, valid_mask):
    """Return decoder layers as [num_layers, num_valid_tokens, hidden_dim]."""
    layers = [
        layer[sample_idx, valid_mask, :].detach().cpu().float()
        for layer in hidden_states[:]
        if layer is not None
    ]
    if not layers:
        raise RuntimeError("Qwen3-VL did not return any decoder hidden states.")
    return torch.stack(layers, dim=0).contiguous()


def _stack_valid_attentions(attentions, sample_idx, valid_mask):
    if attentions is None:
        return None

    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
    layers = []
    for attention in attentions:
        if attention is None:
            continue
        sample_attention = attention[sample_idx]
        sample_attention = sample_attention.index_select(1, valid_indices)
        sample_attention = sample_attention.index_select(2, valid_indices)
        layers.append(sample_attention.detach().cpu().float())

    if not layers:
        return None
    return torch.stack(layers, dim=0).contiguous()


def _make_token_masks(
    input_ids,
    attention_mask,
    model_visual_mask,
    image_token_id,
    video_token_id,
    sample_idx,
):
    """
    Build masks in the padding-free token sequence.

    Image tokens are selected from their actual Qwen3-VL placeholder positions.
    Text is every other valid token, so all special tokens are intentionally
    retained. Boolean indexing later preserves order and concatenates the text
    spans before, between, and after image-token blocks.
    """
    valid_mask = attention_mask[sample_idx].bool()
    sample_ids = input_ids[sample_idx]

    image_mask_full = sample_ids.eq(image_token_id) & valid_mask
    video_mask_full = sample_ids.eq(video_token_id) & valid_mask
    if video_mask_full.any():
        raise ValueError(
            "This ERank script is image-specific, but a <|video_pad|> token "
            "was found. Use an image-only subset."
        )

    image_mask = image_mask_full[valid_mask].detach().cpu()
    text_mask = ~image_mask
    valid_input_ids = sample_ids[valid_mask].detach().cpu().long()

    if model_visual_mask is not None:
        visual_mask = model_visual_mask[sample_idx][valid_mask].detach().cpu().bool()
        if not torch.equal(visual_mask, image_mask):
            raise ValueError(
                "The image-token positions from input_ids do not match "
                "Qwen3-VL outputs.visual_pos_masks."
            )
    else:
        visual_mask = image_mask.clone()

    return valid_mask, valid_input_ids, image_mask, text_mask, visual_mask


def _build_saved_item(
    out_path,
    hidden_state,
    attention,
    valid_input_ids,
    image_token_mask,
    text_token_mask,
    visual_pos_mask,
    tokenizer,
    raw_meta,
    input_text,
    image_path,
    subset,
    side,
    sample_idx,
):
    special_ids = torch.tensor(
        sorted(set(getattr(tokenizer, "all_special_ids", []))),
        dtype=torch.long,
    )
    if special_ids.numel() == 0:
        special_token_mask = torch.zeros_like(valid_input_ids, dtype=torch.bool)
    else:
        special_token_mask = torch.isin(valid_input_ids, special_ids)

    return {
        "format_version": 2,
        "backbone": QWEN3_VL,
        "path": out_path,
        "hidden_state": hidden_state,
        "attention": attention,
        "input_ids": valid_input_ids,
        "tokens": tokenizer.convert_ids_to_tokens(valid_input_ids.tolist()),
        "image_token_mask": image_token_mask,
        "text_token_mask": text_token_mask,
        "visual_pos_mask": visual_pos_mask,
        "special_token_mask": special_token_mask,
        "num_image_tokens": int(image_token_mask.sum().item()),
        "num_text_tokens": int(text_token_mask.sum().item()),
        "num_special_text_tokens": int(
            (text_token_mask & special_token_mask).sum().item()
        ),
        "num_valid_tokens": int(valid_input_ids.numel()),
        "hidden_shape": torch.tensor(hidden_state.shape, dtype=torch.long),
        "attention_shape": torch.tensor(
            attention.shape if attention is not None else [], dtype=torch.long
        ),
        "text": raw_meta["text"],
        "input_text": input_text,
        "img_path": raw_meta["img_path"],
        "input_img_path": image_path,
        "subset": subset,
        "side": side,
        "sample_idx": sample_idx,
        "text_includes_special_tokens": True,
        "text_spans_concatenated_across_image_regions": True,
    }


def infer_side_qwen3(
    model,
    processor,
    tokenizer,
    data_args,
    model_args,
    infer_output_path,
    subset,
    side,
    device,
    batch_size,
    query_to_target_indices,
    tgt_dataset,
    dataset=None,
):
    dataset, loader = build_loader(
        data_args,
        model_args,
        processor,
        subset,
        side,
        batch_size,
        dataset=dataset,
    )
    side_dir = os.path.join(infer_output_path, subset, side)
    os.makedirs(side_dir, exist_ok=True)

    image_token_id = _get_token_id(
        processor, tokenizer, "image_token_id", "<|image_pad|>"
    )
    video_token_id = _get_token_id(
        processor, tokenizer, "video_token_id", "<|video_pad|>"
    )
    rank = get_rank()

    with torch.no_grad():
        for sample_indices, batch in tqdm(
            loader,
            total=len(loader),
            desc=f"Infer Qwen3 {side} - {subset} rank{rank}",
            disable=not is_main_process(),
        ):
            input_texts = batch.get("text")
            image_paths = batch.get("image_paths")
            model_inputs = move_to_device(clean_model_inputs(batch), device)

            outputs = model.encoder(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            hidden_states = outputs.hidden_states
            attention_matrix = getattr(outputs, "attentions", None)
            model_visual_mask = getattr(outputs, "visual_pos_masks", None)

            if hidden_states is None:
                raise RuntimeError("Qwen3-VL returned hidden_states=None.")

            for local_idx, sample_idx in enumerate(sample_indices.tolist()):
                idx = int(sample_idx)
                out_path = os.path.join(side_dir, f"{idx:08d}.pt")
                raw_meta = dataset.paired_data[idx]

                (
                    valid_mask,
                    valid_input_ids,
                    image_token_mask,
                    text_token_mask,
                    visual_pos_mask,
                ) = _make_token_masks(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    model_visual_mask=model_visual_mask,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    sample_idx=local_idx,
                )

                hidden_state = _stack_valid_hidden_states(
                    hidden_states, local_idx, valid_mask
                )
                attention = _stack_valid_attentions(
                    attention_matrix, local_idx, valid_mask
                )

                if hidden_state.size(1) != valid_input_ids.numel():
                    raise ValueError(
                        f"Hidden/token length mismatch for subset={subset}, "
                        f"side={side}, sample_idx={idx}: "
                        f"hidden={hidden_state.size(1)}, ids={valid_input_ids.numel()}."
                    )
                if not torch.all(image_token_mask | text_token_mask):
                    raise ValueError("Image/text masks do not cover every valid token.")
                if torch.any(image_token_mask & text_token_mask):
                    raise ValueError("Image/text masks overlap.")

                item = _build_saved_item(
                    out_path=out_path,
                    hidden_state=hidden_state,
                    attention=attention,
                    valid_input_ids=valid_input_ids,
                    image_token_mask=image_token_mask,
                    text_token_mask=text_token_mask,
                    visual_pos_mask=visual_pos_mask,
                    tokenizer=tokenizer,
                    raw_meta=raw_meta,
                    input_text=(
                        input_texts[local_idx]
                        if input_texts is not None
                        else raw_meta["text"]
                    ),
                    image_path=(
                        image_paths[local_idx]
                        if image_paths is not None
                        else raw_meta["img_path"]
                    ),
                    subset=subset,
                    side=side,
                    sample_idx=idx,
                )
                if side == "query":
                    add_query_target_info(
                        item, idx, query_to_target_indices, tgt_dataset
                    )
                torch.save(item, out_path)


def main():
    fix_local_rank_arg()
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_backbone != QWEN3_VL:
        raise ValueError(
            "This script only supports --model_backbone qwen3_vl, got "
            f"{model_args.model_backbone!r}."
        )

    runtime_data_args = make_runtime_data_args(data_args)
    if runtime_data_args.encode_output_path is None:
        raise ValueError(
            "--encode_output_path is required for saving hidden/attention outputs."
        )
    infer_output_path = runtime_data_args.encode_output_path
    if is_main_process():
        os.makedirs(infer_output_path, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    device = get_device()
    model, infer_model_args, processor, tokenizer = load_infer_model(
        model_args, runtime_data_args, device
    )

    for subset in runtime_data_args.subset_name:
        if is_main_process():
            print(f"\033[91mProcessing Qwen3-VL subset {subset}\033[0m")

        qry_dataset = build_eval_dataset(
            runtime_data_args, infer_model_args, subset, "query"
        )
        tgt_dataset = build_eval_dataset(
            runtime_data_args, infer_model_args, subset, "target"
        )
        query_to_target_indices = build_query_target_maps(
            runtime_data_args,
            infer_model_args,
            subset,
            qry_dataset,
            tgt_dataset,
        )

        for side, dataset in (("query", qry_dataset), ("target", tgt_dataset)):
            infer_side_qwen3(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                data_args=runtime_data_args,
                model_args=infer_model_args,
                infer_output_path=infer_output_path,
                subset=subset,
                side=side,
                device=device,
                batch_size=training_args.per_device_eval_batch_size,
                query_to_target_indices=query_to_target_indices,
                tgt_dataset=tgt_dataset,
                dataset=dataset,
            )


if __name__ == "__main__":
    ddp_setup()
    main()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
