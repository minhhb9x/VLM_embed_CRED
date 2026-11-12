import argparse
import os

import torch
import torch.nn.functional as F


def compute_participation_ratio(
    hidden_state: torch.Tensor,
    eps: float = 1e-10,
    normalize_by_min_dim: bool = False,
    use_trace: bool = True,
) -> torch.Tensor:
    x = hidden_state.float()
    n, d = x.shape
    if n < 2 or d == 0:
        return x.new_zeros(())

    # Cùng cách centering với er_statistic_centered.py.
    x = x - x[:1]
    x = x - x.mean(dim=0, keepdim=True)

    if use_trace:
        cov = (x.T @ x) / (n - 1)
        t1 = torch.trace(cov)
        t2 = cov.square().sum()  # tr(cov²), vì cov đối xứng
    else:
        eigenvalues = torch.linalg.svdvals(x).square() / (n - 1)
        t1 = eigenvalues.sum()
        t2 = eigenvalues.square().sum()

    # Tránh ngưỡng eps tuyệt đối làm lệch PR khi covariance rất nhỏ.
    tiny = torch.finfo(x.dtype).tiny
    pr = t1.square() / t2.clamp_min(tiny)

    if normalize_by_min_dim:
        pr = pr / min(n - 1, d)

    return pr


def get_image_token_slice(obj: dict, hidden_state: torch.Tensor) -> slice:
    """Locate the contiguous image-token block in a padding-free sequence."""
    num_image_tokens = int(obj.get("num_image_tokens", 0))
    num_valid_tokens = int(obj.get("num_valid_tokens", hidden_state.size(1)))

    if hidden_state.size(1) != num_valid_tokens:
        raise ValueError(
            f"Saved hidden length ({hidden_state.size(1)}) does not match "
            f"num_valid_tokens ({num_valid_tokens})."
        )

    if bool(obj.get("last_image_token", False)):
        image_end = num_valid_tokens - int(bool(obj.get("has_eos_id", False)))
        image_start = image_end - num_image_tokens
    else:
        image_start = 0
        image_end = num_image_tokens

    if image_start < 0 or image_end > num_valid_tokens or image_end <= image_start:
        raise ValueError(
            f"Invalid image-token range [{image_start}, {image_end}) for "
            f"num_valid_tokens={num_valid_tokens} and num_image_tokens={num_image_tokens}."
        )

    return slice(image_start, image_end)


def extract_text_tokens(obj: dict, hidden_state: torch.Tensor, image_slice: slice) -> torch.Tensor:
    """
    Remove the image-token block and keep the text-token sequence.

    If num_text_tokens excludes a terminal EOS, the EOS is removed too.
    If num_text_tokens includes EOS, all non-image tokens are kept.
    """
    before_image = hidden_state[:, :image_slice.start, :]
    after_image = hidden_state[:, image_slice.stop:, :]
    text_hidden = torch.cat([before_image, after_image], dim=1)

    num_text_tokens = int(obj.get("num_text_tokens", 0))
    if num_text_tokens <= 0:
        return text_hidden

    num_text_tokens = int(obj["num_text_tokens"])
    if text_hidden.size(1) != num_text_tokens:
        raise ValueError(
            f"Extracted {text_hidden.size(1)} text tokens, "
            f"expected {num_text_tokens}."
        )

    # Chỉ đúng khi has_eos_id xác nhận EOS là token text cuối.
    if obj.get("has_eos_id", False):
        text_hidden = text_hidden[:, :-1, :]

    return text_hidden

def load_hidden_layers(
    pt_path: str,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
    obj = torch.load(pt_path, map_location="cpu")

    num_image_tokens = int(obj.get("num_image_tokens", 0))
    if num_image_tokens <= 0:
        return None, None, None

    hidden_state = obj["hidden_state"].float()
    image_slice = get_image_token_slice(obj, hidden_state)

    image_hidden_layers = hidden_state[:, image_slice, :]
    text_hidden_layers = extract_text_tokens(obj, hidden_state, image_slice)

    if image_hidden_layers.size(1) != num_image_tokens:
        raise ValueError(
            f"Extracted {image_hidden_layers.size(1)} image tokens from {pt_path}, "
            f"expected {num_image_tokens}."
        )

    if text_hidden_layers.size(1) <= 0:
        raise ValueError(f"No text tokens extracted from {pt_path}.")

    if normalize:
        image_hidden_layers = F.normalize(image_hidden_layers, p=2, dim=-1)
        text_hidden_layers = F.normalize(text_hidden_layers, p=2, dim=-1)

    last_token_all_layers = hidden_state[:, -1, :].clone()
    return image_hidden_layers, text_hidden_layers, last_token_all_layers



def compute_per_sample_layer_participation_ratio(
    token_hidden_layers: torch.Tensor,
    device: torch.device,
    normalize_by_min_dim: bool = False,
) -> torch.Tensor:
    """Input [num_layers, num_tokens, hidden_dim] -> output [num_layers]."""
    layer_eranks = []

    for layer_hidden in token_hidden_layers:
        erank = compute_participation_ratio(
            layer_hidden.to(device),
            normalize_by_min_dim=normalize_by_min_dim,
        )
        layer_eranks.append(erank.cpu())

    return torch.stack(layer_eranks, dim=0)



def compute_dataset_layer_participation_ratio(
    hidden_samples: list[torch.Tensor],
    device: torch.device,
    normalize_by_min_dim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each layer compute:
      1. all_token_erank: concatenate all tokens across samples.
      2. mean_pooled_erank: mean-pool tokens within each sample, then eRank across samples.
    """
    all_token_layer_eranks = []
    mean_pooled_layer_eranks = []
    num_layers = hidden_samples[0].size(0)

    for layer_idx in range(num_layers):
        all_tokens = torch.cat([hidden[layer_idx] for hidden in hidden_samples], dim=0).to(device)
        all_token_erank = compute_participation_ratio(
            all_tokens,
            normalize_by_min_dim=normalize_by_min_dim,
        ).cpu()
        all_token_layer_eranks.append(all_token_erank)

        mean_pooled = torch.stack(
            [hidden[layer_idx].mean(dim=0) for hidden in hidden_samples],
            dim=0,
        ).to(device)
        mean_pooled_erank = compute_participation_ratio(
            mean_pooled,
            normalize_by_min_dim=normalize_by_min_dim,
        ).cpu()
        mean_pooled_layer_eranks.append(mean_pooled_erank)

    return (
        torch.stack(all_token_layer_eranks, dim=0),
        torch.stack(mean_pooled_layer_eranks, dim=0),
    )


def summarize_per_sample_eranks(per_sample_layer_eranks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stacked = torch.stack(per_sample_layer_eranks, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    return stacked, mean, std


def get_pt_files(pt_dir: str, num_samples: int) -> list[str]:
    if not os.path.isdir(pt_dir):
        raise FileNotFoundError(f"PT directory does not exist: {pt_dir}")

    pt_files = [
        os.path.join(pt_dir, filename)
        for filename in os.listdir(pt_dir)
        if filename.endswith(".pt")
    ]

    if not pt_files:
        raise RuntimeError(f"No .pt files found in: {pt_dir}")

    pt_files.sort()

    if num_samples > 0:
        pt_files = pt_files[:num_samples]

    return pt_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", default="infer/rkd_meta_cls/ImageNet-1K/query")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of first .pt files to use. Use <= 0 to process all files.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_file", type=str, default="participation_ratio_results.txt")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize image/text tokens along hidden dimension before PR.")
    parser.add_argument(
        "--normalize_by_min_dim",
        action="store_true",
        help="Divide participation ratio by min(n-1, d). If omitted, return raw participation ratio.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    image_per_sample_eranks = []
    text_per_sample_eranks = []
    image_hidden_samples = []
    text_hidden_samples = []
    last_token_all_layers_samples = []
    loaded_files = []

    pt_files = get_pt_files(pt_dir=args.pt_dir, num_samples=args.num_samples)

    print(f"Found {len(pt_files)} .pt files to process.")
    print(f"Participation ratio normalization: {'min(n-1, d)' if args.normalize_by_min_dim else 'none'}")
    print("First files:")
    for pt_path in pt_files[:10]:
        print(f"  {os.path.basename(pt_path)}")
    if len(pt_files) > 10:
        print("  ...")

    for file_idx, pt_path in enumerate(pt_files):
        print(f"[{file_idx + 1}/{len(pt_files)}] Loading {os.path.basename(pt_path)}")

        image_hidden_layers, text_hidden_layers, last_token_all_layers = load_hidden_layers(
            pt_path,
            normalize=args.normalize,
        )

        if image_hidden_layers is None:
            print(f"Skip no-image file: {pt_path}")
            continue

        image_per_sample_eranks.append(
            compute_per_sample_layer_participation_ratio(
                image_hidden_layers,
                device,
                normalize_by_min_dim=args.normalize_by_min_dim,
            )
        )
        text_per_sample_eranks.append(
            compute_per_sample_layer_participation_ratio(
                text_hidden_layers,
                device,
                normalize_by_min_dim=args.normalize_by_min_dim,
            )
        )

        image_hidden_samples.append(image_hidden_layers)
        text_hidden_samples.append(text_hidden_layers)
        last_token_all_layers_samples.append(last_token_all_layers)
        loaded_files.append(pt_path)

    if not image_per_sample_eranks:
        raise RuntimeError("No image/text tokens loaded.")

    image_per_sample_eranks, image_per_sample_mean, image_per_sample_std = summarize_per_sample_eranks(image_per_sample_eranks)
    text_per_sample_eranks, text_per_sample_mean, text_per_sample_std = summarize_per_sample_eranks(text_per_sample_eranks)

    image_all_token_eranks, image_mean_pooled_eranks = compute_dataset_layer_participation_ratio(
        image_hidden_samples,
        device,
        normalize_by_min_dim=args.normalize_by_min_dim,
    )
    text_all_token_eranks, text_mean_pooled_eranks = compute_dataset_layer_participation_ratio(
        text_hidden_samples,
        device,
        normalize_by_min_dim=args.normalize_by_min_dim,
    )

    last_token_all_layers_samples = torch.stack(last_token_all_layers_samples, dim=0)
    last_token_layer_eranks_raw = []
    last_token_layer_eranks_norm = []

    for layer_idx in range(last_token_all_layers_samples.size(1)):
        layer_last_tokens = last_token_all_layers_samples[:, layer_idx, :]

        raw_erank = compute_participation_ratio(
            layer_last_tokens.to(device),
            normalize_by_min_dim=args.normalize_by_min_dim,
        ).cpu()
        last_token_layer_eranks_raw.append(raw_erank)

        layer_last_tokens_norm = F.normalize(layer_last_tokens, p=2, dim=-1)
        norm_erank = compute_participation_ratio(
            layer_last_tokens_norm.to(device),
            normalize_by_min_dim=args.normalize_by_min_dim,
        ).cpu()
        last_token_layer_eranks_norm.append(norm_erank)

    last_token_layer_eranks_raw = torch.stack(last_token_layer_eranks_raw, dim=0)
    last_token_layer_eranks_norm = torch.stack(last_token_layer_eranks_norm, dim=0)
    num_layers = image_hidden_samples[0].size(0)

    output_lines = [
        f"Requested samples: {args.num_samples}",
        f"Loaded files: {len(loaded_files)}",
        f"L2-normalized image/text tokens: {args.normalize}",
        f"Participation ratio normalization: {'min(n-1, d)' if args.normalize_by_min_dim else 'none'}",
        "File selection order: filename sort",
        f"Per-sample image participation ratio shape: {tuple(image_per_sample_eranks.shape)}",
        f"Per-sample text participation ratio shape: {tuple(text_per_sample_eranks.shape)}",
        f"Last-token hidden-state shape: {tuple(last_token_all_layers_samples.shape)}",
        "",
        "Loaded sample files:",
    ]

    for pt_path in loaded_files:
        output_lines.append(f"  {os.path.basename(pt_path)}")

    output_lines.extend(["", "IMAGE TOKEN PRTICIPATION RATIO", "Image participation ratio per layer:"])
    for layer_idx, mean_erank, std_erank, all_token_erank, mean_pooled_erank in zip(
        range(num_layers),
        image_per_sample_mean,
        image_per_sample_std,
        image_all_token_eranks,
        image_mean_pooled_eranks,
    ):
        output_lines.append(
            f"  layer {layer_idx:02d}: "
            f"per_sample_mean={mean_erank.item():.6f}, "
            f"per_sample_std={std_erank.item():.6f}, "
            f"all_token_pr={all_token_erank.item():.6f}, "
            f"mean_pooled_pr={mean_pooled_erank.item():.6f}"
        )

    output_lines.extend(
        [
            "",
            f"Last layer image per-sample mean participation ratio: {image_per_sample_mean[-1].item():.6f}",
            f"Last layer image all-token participation ratio: {image_all_token_eranks[-1].item():.6f}",
            f"Last layer image mean-pooled participation ratio: {image_mean_pooled_eranks[-1].item():.6f}",
            "",
            "TEXT TOKEN PRTICIPATION RATIO",
            "Text participation ratio per layer:",
        ]
    )

    for layer_idx, mean_erank, std_erank, all_token_erank, mean_pooled_erank in zip(
        range(num_layers),
        text_per_sample_mean,
        text_per_sample_std,
        text_all_token_eranks,
        text_mean_pooled_eranks,
    ):
        output_lines.append(
            f"  layer {layer_idx:02d}: "
            f"per_sample_mean={mean_erank.item():.6f}, "
            f"per_sample_std={std_erank.item():.6f}, "
            f"all_token_pr={all_token_erank.item():.6f}, "
            f"mean_pooled_pr={mean_pooled_erank.item():.6f}"
        )

    output_lines.extend(
        [
            "",
            f"Last layer text per-sample mean participation ratio: {text_per_sample_mean[-1].item():.6f}",
            f"Last layer text all-token participation ratio: {text_all_token_eranks[-1].item():.6f}",
            f"Last layer text mean-pooled participation ratio: {text_mean_pooled_eranks[-1].item():.6f}",
            "",
            "LAST TOKEN PRTICIPATION RATIO",
            "Last-token participation ratio across samples per hidden layer:",
        ]
    )

    for layer_idx, raw_erank, norm_erank in zip(
        range(num_layers),
        last_token_layer_eranks_raw,
        last_token_layer_eranks_norm,
    ):
        output_lines.append(
            f"  layer {layer_idx:02d}: "
            f"raw={raw_erank.item():.6f}, "
            f"normalized={norm_erank.item():.6f}"
        )

    output_lines.extend(
        [
            "",
            f"Last hidden layer last-token raw effective rank: {last_token_layer_eranks_raw[-1].item():.6f}",
            f"Last hidden layer last-token normalized effective rank: {last_token_layer_eranks_norm[-1].item():.6f}",
        ]
    )

    output_text = "\n".join(output_lines)
    print()
    print(output_text)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print()
    print(f"Saved output to: {args.output_file}")


if __name__ == "__main__":
    main()
