#!/usr/bin/env python3
"""Causal iso-N/G benchmark for SonicMoE's QuACK varlen-M expert GEMMs.

The benchmark calls the exact expert up/SwiGLU/down kernels used by SonicMoE,
but supplies a controlled expert-prefix vector.  No routing assignment is
added, removed, or moved during a timed comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _require_safe_gpu() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"4", "5", "6", "7"}:
        raise RuntimeError(
            "Synthetic benchmark requires exactly one physical GPU from 4,5,6,7; "
            f"got CUDA_VISIBLE_DEVICES={visible!r}"
        )


def features(counts: list[int], block_m: int) -> dict[str, float | int]:
    active = [x for x in counts if x > 0]
    q = sum(math.ceil(x / block_m) for x in active)
    full = sum(x // block_m for x in active)
    tails = [x % block_m for x in active if x % block_m]
    n = sum(active)
    return {
        "N": n,
        "G": len(active),
        "Q": q,
        "full_tiles": full,
        "tail_count": len(tails),
        "tail_rows": sum(tails),
        "padded_rows": q * block_m - n,
        "padding_amplification": q * block_m / n if n else 0.0,
    }


def fixed_histograms(experts: int, active: int, block_m: int) -> dict[str, list[int]]:
    if active % 4:
        raise ValueError("active experts must be divisible by four")
    base = [block_m] * active
    variants: dict[str, list[int]] = {
        "aligned": base,
        "boundary_heavy": [block_m + d for _ in range(active // 4) for d in (1, 1, -1, -1)],
        "plus_minus_1": [block_m + (1 if i % 2 == 0 else -1) for i in range(active)],
        "fragmented": [block_m // 2] * (active // 2) + [block_m + block_m // 2] * (active // 2),
        "skewed": [2 * block_m - 1] * (active // 2) + [1] * (active // 2),
    }
    rng = np.random.default_rng(1100)
    raw = rng.dirichlet(np.full(active, 0.7)) * (active * block_m - active)
    vals = np.maximum(1, np.floor(raw).astype(int))
    remainder = active * block_m - int(vals.sum())
    order = np.argsort(raw - np.floor(raw))[::-1]
    for i in range(abs(remainder)):
        idx = int(order[i % active])
        vals[idx] += 1 if remainder > 0 else -1
        if vals[idx] < 1:
            vals[idx] = 1
    # Correct any last unit discrepancy without changing G.
    vals[0] += active * block_m - int(vals.sum())
    variants["dirichlet"] = vals.tolist()
    trace_profile = [
        x
        for x in ([0.22, 0.31, 0.38, 0.47, 0.55, 0.63, 0.72, 0.81,
                   0.91, 0.98, 1.02, 1.08, 1.14, 1.21, 1.29, 1.36,
                   0.44, 0.58, 0.69, 0.77, 0.86, 0.94, 1.05, 1.18,
                   1.27, 1.39, 1.48, 1.57, 1.66, 1.75, 1.84, 1.93])[:active]
    ]
    trace_scale = (active * block_m) / sum(trace_profile)
    variants["real_trace_like"] = [max(1, int(round(x * trace_scale))) for x in trace_profile]
    target = active * block_m
    variants["real_trace_like"][0] += target - sum(variants["real_trace_like"])
    for name, vals2 in list(variants.items()):
        if min(vals2) <= 0 or sum(vals2) != active * block_m or len(vals2) != active:
            raise AssertionError((name, min(vals2), sum(vals2), len(vals2)))
        variants[name] = vals2 + [0] * (experts - active)
    return variants


def boundary_hist(experts: int, active: int, block_m: int, multiple: int, delta: int) -> list[int]:
    counts = [multiple * block_m] * active
    counts[0] += delta
    counts[1] -= delta
    if min(counts) <= 0:
        raise ValueError(counts[:2])
    return counts + [0] * (experts - active)


class ExpertKernel:
    def __init__(self, hidden: int, intermediate: int, experts: int, max_n: int, seed: int):
        # Importing sonicmoe installs its intended reduced SM90 QuACK config space.
        import sonicmoe  # noqa: F401
        from quack.gemm_interface import gemm, gemm_act

        self.gemm = gemm
        self.gemm_act = gemm_act
        self.hidden = hidden
        self.intermediate = intermediate
        self.experts = experts
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.x = torch.randn(max_n, hidden, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
        self.w1 = torch.randn(experts, hidden, 2 * intermediate, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
        self.w2 = torch.randn(experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
        self.mid = torch.empty(max_n, intermediate, device="cuda", dtype=torch.bfloat16)
        self.out = torch.empty(max_n, hidden, device="cuda", dtype=torch.bfloat16)

    def call(self, counts: list[int]) -> None:
        n = sum(counts)
        offsets = torch.tensor([0] + np.cumsum(counts).tolist(), device="cuda", dtype=torch.int32)
        self.gemm_act(
            self.x[:n], self.w1, activation="swiglu", cu_seqlens_m=offsets,
            preact_out=None, postact_out=self.mid[:n], store_preact=False,
            tuned=False,
        )
        self.gemm(
            self.mid[:n], self.w2, out=self.out[:n], cu_seqlens_m=offsets,
            tuned=False,
        )


def time_kernel(kernel: ExpertKernel, counts: list[int], warmup: int, reps: int) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        kernel.call(counts)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel.call(counts)
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    arr = np.asarray(times)
    return {
        "median_ms": float(np.median(arr)),
        "p25_ms": float(np.percentile(arr, 25)),
        "p75_ms": float(np.percentile(arr, 75)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr)),
        "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) else 0.0,
        "samples_ms": times,
    }


def run_shape(name: str, hidden: int, intermediate: int, args: argparse.Namespace) -> dict:
    block_m = args.block_m
    hists = fixed_histograms(args.experts, args.active_experts, block_m)
    max_n = max(sum(x) for x in hists.values())
    max_n = max(max_n, args.active_experts * 3 * block_m)
    kernel = ExpertKernel(hidden, intermediate, args.experts, max_n, args.seed)
    records = []
    for hist_name, counts in hists.items():
        timing = time_kernel(kernel, counts, args.warmup, args.reps)
        records.append({"kind": "histogram", "name": hist_name, "counts": counts, **features(counts, block_m), **timing})
    for multiple in (1, 2, 3):
        for delta in range(-8, 9):
            counts = boundary_hist(args.experts, args.active_experts, block_m, multiple, delta)
            timing = time_kernel(kernel, counts, args.warmup, args.reps)
            records.append({
                "kind": "boundary", "name": f"{multiple}B_{delta:+d}",
                "multiple": multiple, "delta": delta, "counts": counts,
                **features(counts, block_m), **timing,
            })
    lut = []
    for rows in sorted(set([1, 2, 4, 8, 16, 32, 48, 64, 96, block_m - 1, block_m])):
        counts = [rows] + [0] * (args.experts - 1)
        timing = time_kernel(kernel, counts, args.warmup, args.reps)
        lut.append({"rows": rows, "counts": counts, **features(counts, block_m), **timing})
    del kernel
    torch.cuda.empty_cache()
    return {"name": name, "hidden": hidden, "intermediate": intermediate, "records": records, "tail_lut": lut}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--active-experts", type=int, default=32)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1100)
    parser.add_argument("--shape", choices=["both", "sonic", "qwen3"], default="both")
    args = parser.parse_args()
    _require_safe_gpu()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = {
        "schema": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "experts": args.experts,
        "top_k_context": 8,
        "active_experts": args.active_experts,
        "fixed_kernel_config": {"tile_m": args.block_m, "source": "QuACK SM90 default, tuned=False"},
        "warmup": args.warmup,
        "reps": args.reps,
        "shapes": [],
    }
    if args.shape in {"both", "sonic"}:
        payload["shapes"].append(run_shape("sonic_30b_like", 4096, 1024, args))
    if args.shape in {"both", "qwen3"}:
        payload["shapes"].append(run_shape("qwen3_30b_a3b", 2048, 768, args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
