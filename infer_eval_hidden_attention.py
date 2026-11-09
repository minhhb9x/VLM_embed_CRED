import os
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.data.collator.eval_collator import EvalCollator
from src.data.dataset.mmeb_dataset import EvalDataset
from src.model.model import MMEBModel
from src.model.processor import load_processor
from functools import partial
from src.model.processor import VLM_IMAGE_TOKENS



POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
    "ImageNet-1K": POS_MOD_CLASS_LABEL,
    "ImageNet_1K": POS_MOD_CLASS_LABEL,
    "HatefulMemes": POS_MOD_CLASS_LABEL,
    "SUN397": POS_MOD_CLASS_LABEL,
    "N24News": POS_MOD_CLASS_LABEL,
    "VOC2007": POS_MOD_CLASS_LABEL,
    "Place365": POS_MOD_CLASS_LABEL,
    "ImageNet-A": POS_MOD_CLASS_LABEL,
    "ImageNet-R": POS_MOD_CLASS_LABEL,
    "ObjectNet": POS_MOD_CLASS_LABEL,
    "Country211": POS_MOD_CLASS_LABEL,
    "OK-VQA": POS_MOD_ANSWER,
    "A-OKVQA": POS_MOD_ANSWER,
    "DocVQA": POS_MOD_ANSWER,
    "InfographicsVQA": POS_MOD_ANSWER,
    "ChartQA": POS_MOD_ANSWER,
    "Visual7W": POS_MOD_ANSWER,
    "ScienceQA": POS_MOD_ANSWER,
    "GQA": POS_MOD_ANSWER,
    "TextVQA": POS_MOD_ANSWER,
    "VizWiz": POS_MOD_ANSWER,
    "MSCOCO_i2t": POS_MOD_IMAGE_CAPTION,
    "VisualNews_i2t": POS_MOD_IMAGE_CAPTION,
}

def move_image_token_to_end(text, image_token):
    if not text or image_token not in text:
        return text

    text = text.rstrip()

    num_images = text.count(image_token)

    # Xóa placeholder khỏi vị trí cũ
    text_parts = [
        part.strip()
        for part in text.split(image_token)
        if part.strip()
    ]
    text_without_images = " ".join(text_parts)

    # Đưa placeholder xuống cuối phần nội dung
    image_suffix = " ".join([image_token] * num_images)

    parts = [text_without_images, image_suffix]

    return "\n".join(part for part in parts if part)

class IndexedDataset(Dataset):
    def __init__(self, dataset, text_transform=None):
        self.dataset = dataset
        self.text_transform = text_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, image = self.dataset[idx]

        if self.text_transform is not None:
            text = self.text_transform(text)

        # if idx < 5:
        #     print(f"IndexedDataset[{idx}]:")
        #     print(f"text => {text}")
        #     print(f"image => {image}")

        return idx, (text, image)


class StrideDistributedSampler(Sampler):
    def __init__(self, dataset, shuffle=False, seed=42):
        self.dataset = dataset
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        n = len(self.dataset)

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(
                n,
                generator=generator,
            ).tolist()
        else:
            indices = list(range(n))

        # Chia theo stride sau khi shuffle
        indices = indices[self.rank::self.world_size]

        return iter(indices)

    def __len__(self):
        total = len(self.dataset)
        return (total + self.world_size - 1 - self.rank) // self.world_size


class IndexedEvalCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, examples):
        indices, raw_examples = zip(*examples)
        return torch.tensor(indices, dtype=torch.long), self.collator(list(raw_examples))


def fix_local_rank_arg():
    for arg in list(sys.argv):
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append("--local_rank")
            sys.argv.append(rank)


def ddp_setup():
    if "LOCAL_RANK" not in os.environ:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
            "Lower --nproc_per_node or set CUDA_VISIBLE_DEVICES to expose more GPUs."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process():
    return get_rank() == 0


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    return torch.device("cpu")


def make_runtime_data_args(data_args):
    runtime_args = deepcopy(data_args)
    runtime_args.caching_dir = None
    return runtime_args


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def move_to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        values = [move_to_device(value, device) for value in obj]
        return tuple(values) if isinstance(obj, tuple) else values
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def get_special_ids_for_text_count(tokenizer):
    if tokenizer is None:
        return set()

    eos_ids = tokenizer.eos_token_id
    eos_ids = [] if eos_ids is None else eos_ids if isinstance(eos_ids, list) else [eos_ids]
    return (
        set(getattr(tokenizer, "all_special_ids", []))
        | set(tokenizer.added_tokens_encoder.values())
    ) - set(eos_ids)


def load_infer_model(model_args, data_args, device):
    processor = load_processor(model_args, data_args)
    tokenizer = getattr(processor, "tokenizer", None)
    model = MMEBModel.load(model_args, is_trainable=False).to(device)
    freeze_module(model)
    return model, model_args, processor, tokenizer


def clean_model_inputs(batch):
    ignored_keys = {
        "text",
        "texts",
        "images",
        "image_paths",
        "global_dataset_name",
    }
    return {key: value for key, value in batch.items() if key not in ignored_keys}


def count_text_tokens(inputs, special_ids_tensor, idx):
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return 0
    valid_mask = input_ids.ge(0) & inputs["attention_mask"].bool()
    if special_ids_tensor.numel() > 0:
        valid_mask = valid_mask & ~torch.isin(input_ids, special_ids_tensor.to(input_ids.device))
    return int(valid_mask[idx].sum().item())


def count_image_tokens(image_features, idx):
    if image_features is None or idx >= len(image_features) or image_features[idx] is None:
        return 0
    return int(image_features[idx].size(0))


def get_clean_token_slice(attention_mask, hidden_len, num_text_tokens, num_image_tokens):
    num_valid_tokens = num_text_tokens + num_image_tokens
    if num_valid_tokens <= 0:
        raise ValueError("No valid token found for hidden/attention inference.")
    if num_valid_tokens > hidden_len:
        raise ValueError(
            f"num_valid_tokens={num_valid_tokens} exceeds hidden_len={hidden_len}. "
            f"num_image_tokens={num_image_tokens}, num_text_tokens={num_text_tokens}."
        )

    left_padding = bool((attention_mask[0].eq(0) & attention_mask[-1].eq(1)).item())
    if left_padding:
        return slice(hidden_len - num_valid_tokens, hidden_len)
    return slice(0, num_valid_tokens)


def slice_hidden_states(hidden_states, local_idx, token_slice):
    return torch.stack(
        [
            hidden_state[local_idx, token_slice, :].detach().cpu().float()
            for hidden_state in hidden_states[1:]
            if hidden_state is not None
        ],
        dim=0,
    ).contiguous()


def slice_attention_matrix(attention_matrix, local_idx, token_slice):
    if attention_matrix is None:
        return None

    attention_layers = [
        attention[local_idx, :, token_slice, :][:, :, token_slice].detach().cpu().float()
        for attention in attention_matrix
        if attention is not None
    ]
    if not attention_layers:
        return None
    return torch.stack(attention_layers, dim=0).contiguous()


def get_side_fields(side):
    if side == "query":
        return "qry_text", "qry_img_path"
    if side == "target":
        return "tgt_text", "tgt_img_path"
    raise ValueError(f"Unsupported side: {side}")


def build_eval_dataset(data_args, model_args, subset, side):
    text_field, img_path_field = get_side_fields(side)
    mod_instruction = None
    if side == "target" and data_args.tgt_prefix_mod:
        mod_instruction = POS_MOD_DICT.get(subset, None)

    return EvalDataset(
        data_args=data_args,
        model_args=model_args,
        subset=subset,
        text_field=text_field,
        img_path_field=img_path_field,
        mod_instruction=mod_instruction,
    )


def build_loader(data_args, model_args, processor, subset, side, batch_size, dataset=None, text_transform=None):
    if dataset is None:
        dataset = build_eval_dataset(data_args, model_args, subset, side)
    indexed_dataset = IndexedDataset(dataset, text_transform)
    collator = IndexedEvalCollator(EvalCollator(data_args=data_args, model_args=model_args, processor=processor))
    loader = DataLoader(
        indexed_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=StrideDistributedSampler(indexed_dataset),
        drop_last=False,
        num_workers=0,
    )
    return dataset, loader


def pair_key(pair):
    return pair["text"], pair["img_path"]


def row_pair_keys(row, text_field, img_path_field):
    text_value = row[text_field]
    img_path_value = row[img_path_field]

    if isinstance(text_value, str):
        if text_value:
            return [(text_value, img_path_value)]
        if isinstance(img_path_value, list):
            return [(text_value, img_path) for img_path in img_path_value]
        return [(text_value, img_path_value)]

    if isinstance(text_value, list):
        assert isinstance(img_path_value, list) and len(img_path_value) == len(text_value)
        return list(zip(text_value, img_path_value))

    raise TypeError(f"Unsupported text field type: {type(text_value)}")


def append_unique(values, value):
    if value not in values:
        values.append(value)


def load_eval_rows_for_mapping(data_args, model_args, subset):
    eval_data = load_dataset(
        data_args.dataset_name,
        subset,
        split=data_args.dataset_split,
    )
    if (subset == "WebQA" or subset == "EDIS") and "qry_text" in eval_data.column_names and model_args.model_backbone == "llava_qwen2":
        eval_data = eval_data.map(
            lambda x: {"qry_text": x["qry_text"].replace("<|image_1|>", "").strip()}
        )
    return eval_data


def build_query_target_maps(data_args, model_args, subset, qry_dataset, tgt_dataset):
    qry_key_to_idx = {pair_key(pair): idx for idx, pair in enumerate(qry_dataset.paired_data)}
    tgt_key_to_idx = {pair_key(pair): idx for idx, pair in enumerate(tgt_dataset.paired_data)}
    query_to_target_indices = {}
    eval_data = load_eval_rows_for_mapping(data_args, model_args, subset)

    for row in eval_data:
        qry_keys = row_pair_keys(row, "qry_text", "qry_img_path")
        tgt_keys = row_pair_keys(row, "tgt_text", "tgt_img_path")
        if not tgt_keys:
            continue
        positive_tgt_idx = tgt_key_to_idx.get(tgt_keys[0])
        if positive_tgt_idx is None:
            continue

        for qry_key in qry_keys:
            qry_idx = qry_key_to_idx.get(qry_key)
            if qry_idx is None:
                continue
            append_unique(query_to_target_indices.setdefault(qry_idx, []), positive_tgt_idx)
    if is_main_process():
        num_mapped_queries = len(query_to_target_indices)
        print(
            f"Built query-to-target mapping for subset={subset}: "
            f"{num_mapped_queries}/{len(qry_dataset.paired_data)} queries mapped, "
            f"{len(tgt_dataset.paired_data)} targets."
        )
        for qry_idx in list(query_to_target_indices.keys())[:5]:
            tgt_indices = query_to_target_indices[qry_idx]
            qry_pair = qry_dataset.paired_data[qry_idx]
            tgt_pairs = [tgt_dataset.paired_data[tgt_idx] for tgt_idx in tgt_indices]
            print(
                f"  query[{qry_idx}] {pair_key(qry_pair)} -> "
                f"target_indices={tgt_indices}, targets={[pair_key(pair) for pair in tgt_pairs]}"
            )
    return query_to_target_indices


def first_or_missing(values):
    return values[0] if values else -1


def target_info(tgt_dataset, target_idx):
    if target_idx < 0:
        return None, None
    pair = tgt_dataset.paired_data[target_idx]
    return pair["text"], pair["img_path"]


def add_query_target_info(item, sample_idx, query_to_target_indices, tgt_dataset):
    target_indices = query_to_target_indices.get(sample_idx, [])
    target_idx = first_or_missing(target_indices)
    target_label, target_img_path = target_info(tgt_dataset, target_idx)

    item["positive_target_idx"] = target_idx
    item["positive_target_label"] = target_label
    item["positive_target_img_path"] = target_img_path

    if len(target_indices) > 1:
        labels, img_paths = [], []
        for idx in target_indices:
            label, img_path = target_info(tgt_dataset, idx)
            labels.append(label)
            img_paths.append(img_path)
        item["positive_target_indices"] = target_indices
        item["positive_target_labels"] = labels
        item["positive_target_img_paths"] = img_paths


def build_saved_item(
    out_path,
    hidden_state,
    attention,
    raw_meta,
    input_text,
    image_path,
    subset,
    side,
    sample_idx,
    num_image_tokens,
    num_text_tokens,
    num_valid_tokens,
    last_image_token=False,
    has_eos_id=False,
):
    return {
        "path": out_path,
        "hidden_state": hidden_state,
        "attention": attention,
        "num_image_tokens": num_image_tokens,
        "num_text_tokens": num_text_tokens,
        "num_valid_tokens": num_valid_tokens,
        "hidden_shape": torch.tensor(hidden_state.shape, dtype=torch.long),
        "attention_shape": torch.tensor(
            attention.shape if attention is not None else [],
            dtype=torch.long,
        ),
        "text": raw_meta["text"],
        "input_text": input_text,
        "img_path": raw_meta["img_path"],
        "input_img_path": image_path,
        "subset": subset,
        "side": side,
        "sample_idx": sample_idx,
        "last_image_token": last_image_token,
        "has_eos_id": has_eos_id
    }


def infer_side(
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
    text_transform=None,
    last_image_token=False,
):
    dataset, loader = build_loader(data_args, model_args, processor, subset, side, batch_size, dataset=dataset, text_transform=text_transform)
    side_dir = os.path.join(infer_output_path, subset, side)
    os.makedirs(side_dir, exist_ok=True)
    rank = get_rank()
    special_ids = get_special_ids_for_text_count(tokenizer)
    special_ids_tensor = torch.tensor(sorted(special_ids), device=device, dtype=torch.long)

    MAX_INFER_BATCHES = 50

    with torch.no_grad():
        for batch_idx, (sample_indices, batch) in enumerate(
                tqdm(loader,
                    total=min(len(loader), MAX_INFER_BATCHES),
                    desc=f"Infer {side} - {subset} rank{rank}",
                    disable=not is_main_process(),)):
            
            # if not batch_idx in range(0, 10):
            #     continue
            
            if MAX_INFER_BATCHES == 0:
                exit(0)
            else:
                MAX_INFER_BATCHES -= 1

            input_texts = batch.get("text")
            image_paths = batch.get("image_paths")
            model_inputs = move_to_device(batch, device)
            _, image_features, attention_matrix, hidden_states = model.encode_input(model_inputs)

            last_hidden_batch = hidden_states[-1].detach()
            has_eos_id = (model_inputs["input_ids"] == tokenizer.eos_token_id).any().item()
            for local_idx, sample_idx in enumerate(sample_indices.tolist()):
                idx = int(sample_idx)
                out_path = os.path.join(side_dir, f"{idx:08d}.pt")
                raw_meta = dataset.paired_data[idx]

                num_image_tokens = count_image_tokens(image_features, local_idx)
                num_text_tokens = count_text_tokens(model_inputs, special_ids_tensor, local_idx)

                num_valid_tokens = num_image_tokens + num_text_tokens
                token_slice = get_clean_token_slice(
                    model_inputs["attention_mask"][local_idx],
                    last_hidden_batch.size(1),
                    num_text_tokens,
                    num_image_tokens,
                )
                hidden_state = slice_hidden_states(hidden_states, local_idx, token_slice)
                attention = slice_attention_matrix(attention_matrix, local_idx, token_slice)
                if hidden_state.size(1) != num_valid_tokens:
                    raise ValueError(
                        f"Token count mismatch at subset={subset} side={side} sample_idx={idx}: "
                        f"saved_tokens={hidden_state.size(1)}, image+text={num_valid_tokens}, "
                        f"num_image_tokens={num_image_tokens}, num_text_tokens={num_text_tokens}, "
                        f"hidden_len={last_hidden_batch.size(1)}, input_len={model_inputs['input_ids'].size(1)}."
                    )
                item = build_saved_item(
                    out_path=out_path,
                    hidden_state=hidden_state,
                    attention=attention,
                    raw_meta=raw_meta,
                    input_text=input_texts[local_idx] if input_texts is not None else raw_meta["text"],
                    image_path=image_paths[local_idx] if image_paths is not None else raw_meta["img_path"],
                    subset=subset,
                    side=side,
                    sample_idx=idx,
                    num_image_tokens=num_image_tokens,
                    num_text_tokens=num_text_tokens,
                    num_valid_tokens=num_valid_tokens,
                    last_image_token=last_image_token,
                    has_eos_id=has_eos_id
                )
                if side == "query":
                    add_query_target_info(item, idx, query_to_target_indices, tgt_dataset)

                torch.save(item, out_path)


def main():
    fix_local_rank_arg()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--last_image_token", action="store_true", help="Use the last image token for hidden/attention inference.")
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    text_transform = None
    if extra_args.last_image_token:
        text_transform = partial(
            move_image_token_to_end,
            image_token=VLM_IMAGE_TOKENS[model_args.model_backbone],
        )

    print(f"Convert image token to the end: {extra_args.last_image_token}")

    runtime_data_args = make_runtime_data_args(data_args)
    if runtime_data_args.encode_output_path is None:
        raise ValueError("--encode_output_path is required for saving hidden/attention outputs.")
    infer_output_path = runtime_data_args.encode_output_path
    if is_main_process():
        os.makedirs(infer_output_path, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    device = get_device()
    model, infer_model_args, processor, tokenizer = load_infer_model(model_args, runtime_data_args, device)
    for subset in runtime_data_args.subset_name:
        if is_main_process():
            print(f"\033[91mProcessing {subset}\033[0m")
        qry_dataset = build_eval_dataset(runtime_data_args, infer_model_args, subset, "query")
        tgt_dataset = build_eval_dataset(runtime_data_args, infer_model_args, subset, "target")
        query_to_target_indices = build_query_target_maps(
            runtime_data_args,
            infer_model_args,
            subset,
            qry_dataset,
            tgt_dataset,
        )
        for side, dataset in (("query", qry_dataset), ("target", tgt_dataset)):
            infer_side(
                model,
                processor,
                tokenizer,
                runtime_data_args,
                infer_model_args,
                infer_output_path,
                subset,
                side,
                device,
                training_args.per_device_eval_batch_size,
                query_to_target_indices,
                tgt_dataset,
                dataset=dataset,
                text_transform=text_transform,
                last_image_token=extra_args.last_image_token
            )


if __name__ == "__main__":
    ddp_setup()
    main()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
