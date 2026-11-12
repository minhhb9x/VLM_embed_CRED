#!/usr/bin/env python3
"""Benchmark centered token PR and Shannon eRank.

Requires PyTorch. Example:
  python benchmark_pr.py --device cuda --hidden-dim 896 --output pr_forward.csv
  python benchmark_pr.py --device cuda --hidden-dim 896 --mode backward \
      --output pr_backward.csv

Every method receives the same [B, N, D] input for each trial. Centering,
FP32 conversion, and sketch generation are included in measurements.
The full batch is processed together. Reported times and memory are means
over measurements divided by B. Memory is the mean of per-trial peaks,
amortized over the batch. Relative errors are averaged over individual samples.
Forward measures a per-image statistic vector. Backward measures its mean.backward()
through the input tokens. This isolates the statistic, rather than measuring
the full CPR hinge with a detached reference, padding, or model inference.
Shannon eRank uses squared singular values of centered tokens, as in
AblationCred: exp(-sum(p * log(p))). It is not divided by min(N, D).
Each method's error is measured against its own statistic's FP64 reference.
"""

import argparse
import csv
import gc
import json
from pathlib import Path
import platform
import statistics
import time

import torch


def centered_tokens(tokens):
    tokens = tokens.float()
    return tokens - tokens.mean(dim=1, keepdim=True)


def matrix_free_pr(tokens, projections=256, eps=1e-8):
    """Use the same spherical projection moments as TalasJepa."""
    centered = centered_tokens(tokens)
    _, token_count, hidden_dim = centered.shape
    directions = torch.randn(
        hidden_dim, projections, device=tokens.device, dtype=torch.float32
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(eps)
    projected = centered @ directions
    trace = hidden_dim * (
        projected.square().sum(dim=1) / (token_count - 1)
    ).mean(dim=-1)
    covariance_directions = (
        centered.transpose(1, 2) @ projected
    ) / (token_count - 1)
    trace_squared = hidden_dim * covariance_directions.square().sum(dim=1).mean(
        dim=-1
    )
    return trace.square() / trace_squared.clamp_min(eps)


def exact_pr(tokens, eps=1e-8):
    """Choose the smaller N x N Gram matrix or D x D covariance matrix."""
    centered = centered_tokens(tokens)
    token_count, hidden_dim = centered.shape[-2:]
    if token_count <= hidden_dim:
        matrix = centered @ centered.transpose(1, 2)
    else:
        matrix = centered.transpose(1, 2) @ centered
    covariance = matrix / (token_count - 1)
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return trace.square() / covariance.square().sum(dim=(-2, -1)).clamp_min(eps)


def svd_pr(tokens, driver="gesvd", eps=1e-8):
    """Compute the same PR from squared singular values, not Shannon eRank."""
    centered = centered_tokens(tokens)
    kwargs = {"driver": driver} if tokens.is_cuda else {}
    singular_values = torch.linalg.svdvals(centered, **kwargs)
    eigenvalues = singular_values.square() / (tokens.shape[1] - 1)
    return eigenvalues.sum(dim=-1).square() / eigenvalues.square().sum(
        dim=-1
    ).clamp_min(eps)


def shannon_erank(tokens, driver="gesvd", eps=1e-8):
    """Absolute Shannon eRank of each sample's centered covariance spectrum."""
    centered = centered_tokens(tokens)
    kwargs = {"driver": driver} if tokens.is_cuda else {}
    eigenvalues = torch.linalg.svdvals(centered, **kwargs).square()
    total = eigenvalues.sum(dim=-1, keepdim=True)
    probabilities = eigenvalues / total.clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    return torch.where(total.squeeze(-1) > 0, entropy.exp(), 0.0)


def reference_pr(tokens):
    """FP64 exact-PR reference, evaluated outside the benchmark."""
    centered = tokens.double()
    centered = centered - centered.mean(dim=1, keepdim=True)
    token_count, hidden_dim = centered.shape[-2:]
    if token_count <= hidden_dim:
        gram = centered @ centered.transpose(1, 2)
    else:
        gram = centered.transpose(1, 2) @ centered
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return trace.square() / gram.square().sum(dim=(-2, -1))


def reference_erank(tokens, eps=1e-8):
    """FP64 Gram eigendecomposition reference, outside the benchmark."""
    centered = tokens.double()
    centered = centered - centered.mean(dim=1, keepdim=True)
    token_count, hidden_dim = centered.shape[-2:]
    if token_count <= hidden_dim:
        gram = centered @ centered.transpose(1, 2)
    else:
        gram = centered.transpose(1, 2) @ centered
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eigenvalues.sum(dim=-1, keepdim=True)
    probabilities = eigenvalues / total.clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    return torch.where(total.squeeze(-1) > 0, entropy.exp(), 0.0)


def make_tokens(args, token_count, seed):
    generator = torch.Generator(device=args.device).manual_seed(seed)
    return torch.randn(
        args.batch_size, token_count, args.hidden_dim,
        generator=generator,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )


def run_once(function, tokens, backward):
    result = function(tokens)
    if backward:
        result.mean().backward()
    return result


def measure(function, original_tokens, args):
    tokens = original_tokens.detach().requires_grad_(args.mode == "backward")
    backward = args.mode == "backward"
    context = torch.enable_grad if backward else torch.no_grad
    wall_times = []
    cuda_times = []
    baseline_bytes = None
    extra_peak_bytes = None
    with context():
        for _ in range(args.warmup):
            result = run_once(function, tokens, backward)
            del result
            tokens.grad = None
        if tokens.is_cuda:
            torch.cuda.synchronize(tokens.device)
            baseline_bytes = torch.cuda.memory_allocated(tokens.device)
            torch.cuda.reset_peak_memory_stats(tokens.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            if tokens.is_cuda:
                torch.cuda.synchronize(tokens.device)
                start.record()
            wall_start = time.perf_counter()
            result = run_once(function, tokens, backward)
            if tokens.is_cuda:
                end.record()
                torch.cuda.synchronize(tokens.device)
            wall_times.append((time.perf_counter() - wall_start) * 1000)
            if tokens.is_cuda:
                cuda_times.append(start.elapsed_time(end))
            del result
            tokens.grad = None
        if tokens.is_cuda:
            extra_peak_bytes = (
                torch.cuda.max_memory_allocated(tokens.device) - baseline_bytes
            )
    return wall_times, cuda_times, baseline_bytes, extra_peak_bytes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=896)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[144, 256, 576, 1024, 2048, 4096])
    parser.add_argument("--projections", type=int, default=256)
    parser.add_argument("--mode", choices=["forward", "backward"], default="forward")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--error-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svd-driver", choices=["gesvd", "gesvdj"], default="gesvd")
    parser.add_argument("--output", type=Path, default=Path("pr_benchmark.csv"))
    args = parser.parse_args()
    positive = [args.hidden_dim, args.batch_size, args.projections, args.repeats,
                args.trials, args.error_draws]
    if min(positive) < 1 or args.warmup < 0 or min(args.tokens) < 2:
        parser.error("Use positive sizes/counts, N >= 2, and warmup >= 0.")
    args.device = torch.device(args.device)
    if args.device.type not in ("cpu", "cuda"):
        parser.error("Choose a CPU or CUDA device.")
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Activate your CUDA PyTorch environment.")
        index = args.device.index
        if index is None:
            index = torch.cuda.current_device()
        args.device = torch.device("cuda", index)
        torch.cuda.set_device(args.device)
    return args


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    methods = {
        "Matrix-free PR": lambda x: matrix_free_pr(x, args.projections),
        "Exact PR": exact_pr,
        "Exact SVD PR": lambda x: svd_pr(x, args.svd_driver),
        "Shannon eRank": lambda x: shannon_erank(x, args.svd_driver),
    }
    metadata = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device)
               if args.device.type == "cuda" else None,
        "platform": platform.platform(),
        "settings": {key: str(value) if isinstance(value, (Path, torch.device))
                     else value for key, value in vars(args).items()},
        "input_distribution": "independent standard Gaussian entries",
        "compute_dtype": "float32",
        "reference": "Same quantized input in float64: exact PR for PR methods; "
                     "Gram eigendecomposition Shannon eRank for Shannon eRank.",
        "shannon_erank": "Centered covariance spectrum, per sample, squared "
                         "singular values; absolute eRank without rank normalization.",
        "timing": "Mean full-batch runtime divided by B, in ms/sample. "
                  "Centering and fresh projection generation included; model excluded.",
        "memory": "Mean across trials, divided by B, in MiB/sample. Extra peak is "
                  "peak PyTorch allocated tensor bytes minus live pre-call bytes. "
                  "These are amortized batch memory measurements. "
                  "Includes input gradients in backward mode. External CUDA/library "
                  "allocations and reserved allocator cache are not measured.",
        "error": "Mean per-sample relative error against the matching statistic's "
                 "float64 reference, across batch elements, trials, and projection "
                 "draws, in percent. Shannon eRank is not compared against PR.",
        "backward_objective": "mean(PR) or mean(Shannon eRank), "
                              "excluding CPR reference and hinge",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    fields = [
        "Tokens (N)", "Hidden Dim (D)", "Batch Size (B)", "Projections (M)",
        "Mode", "Method", "Status",
        "Mean Time (ms/sample)", "Mean GPU Time (ms/sample)",
        "Mean Baseline Memory (MiB/sample)", "Mean Extra Peak Memory (MiB/sample)",
        "Mean Relative Error (%)", "Timing Runs", "Error Samples",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for token_count in args.tokens:
            records = {
                name: {"wall": [], "cuda": [], "baseline": [], "peak": [],
                       "errors": [], "status": "ok"} for name in methods
            }
            for trial in range(args.trials):
                seed = args.seed + trial
                tokens = make_tokens(args, token_count, seed)
                with torch.no_grad():
                    pr_reference = reference_pr(tokens).cpu()
                names = list(methods)
                offset = trial % len(names)
                names = names[offset:] + names[:offset]
                for name in names:
                    record = records[name]
                    if record["status"] != "ok":
                        continue
                    try:
                        # Keep reference computation outside timing/memory measurements.
                        with torch.no_grad():
                            reference = (reference_erank(tokens).cpu()
                                         if name == "Shannon eRank" else pr_reference)
                        torch.manual_seed(seed + 10000)
                        wall, cuda, baseline, peak = measure(
                            methods[name], tokens, args
                        )
                        record["wall"].extend(wall)
                        record["cuda"].extend(cuda)
                        if peak is not None:
                            record["baseline"].append(baseline / 1024 ** 2)
                            record["peak"].append(peak / 1024 ** 2)
                        draws = args.error_draws if name == "Matrix-free PR" else 1
                        with torch.no_grad():
                            for _ in range(draws):
                                value = methods[name](tokens).double().cpu()
                                if not torch.isfinite(value).all():
                                    raise RuntimeError(
                                        f"Non-finite statistic from {name}, "
                                        f"N={token_count}"
                                    )
                                error = (value - reference).abs() / reference
                                record["errors"].extend((100 * error).tolist())
                    except torch.cuda.OutOfMemoryError:
                        record["status"] = "cuda_oom"
                        print(f"OOM: {name}, N={token_count}",
                              flush=True)
                    gc.collect()
                    if tokens.is_cuda:
                        torch.cuda.empty_cache()
                del tokens
            for name, record in records.items():
                row = {
                    "Tokens (N)": token_count,
                    "Hidden Dim (D)": args.hidden_dim,
                    "Batch Size (B)": args.batch_size,
                    "Projections (M)": args.projections,
                    "Mode": args.mode,
                    "Method": name,
                    "Status": record["status"],
                }
                if record["status"] == "ok":
                    row.update({
                        "Mean Time (ms/sample)": (
                            statistics.mean(record["wall"]) / args.batch_size
                        ),
                        "Mean GPU Time (ms/sample)": (
                            statistics.mean(record["cuda"]) / args.batch_size
                            if record["cuda"] else ""
                        ),
                        "Mean Baseline Memory (MiB/sample)": (
                            statistics.mean(record["baseline"]) / args.batch_size
                            if record["baseline"] else ""
                        ),
                        "Mean Extra Peak Memory (MiB/sample)": (
                            statistics.mean(record["peak"]) / args.batch_size
                            if record["peak"] else ""
                        ),
                        "Mean Relative Error (%)": statistics.mean(record["errors"]),
                        "Timing Runs": len(record["wall"]),
                        "Error Samples": len(record["errors"]),
                    })
                    print(
                        f"N={token_count:4} {name:16} "
                        f"mean={row['Mean Time (ms/sample)']:.3f} ms/sample "
                        f"error={row['Mean Relative Error (%)']:.4f}%",
                        flush=True,
                    )
                writer.writerow(row)
                stream.flush()
    print(f"Saved {args.output} and {args.output.with_suffix('.metadata.json')}")


if __name__ == "__main__":
    main()
