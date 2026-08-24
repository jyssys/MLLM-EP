#!/usr/bin/env python3
"""QuACK 0.6.4 closure for the preregistered Qwen3 BF16 tile boundary test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from benchmark_sonic import ExpertKernel, boundary_hist, features, time_kernel


def require_physical_gpu4() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=4; no other GPU may be visible")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 20 or args.reps < 100:
        raise ValueError("closure requires warmup >=20 and measured reps >=100")
    require_physical_gpu4()

    # Importing Sonic installs its intended QuACK configuration hooks.
    import sonicmoe  # noqa: F401
    from quack.gemm_config import cta_tile_shape_m, default_config

    cfg = default_config(torch.device("cuda"))
    block_m = cta_tile_shape_m(
        cfg.tile_m, cfg.cluster_m, cfg.device_capacity
    )
    experts, active, hidden, intermediate = 128, 32, 2048, 768
    aligned = [block_m] * active + [0] * (experts - active)
    boundary_heavy = [
        block_m + delta
        for _ in range(active // 4)
        for delta in (1, 1, -1, -1)
    ] + [0] * (experts - active)
    cases = {"aligned": aligned, "boundary_heavy": boundary_heavy}
    for multiple in (1, 2, 3):
        for delta in (-1, 0, 1):
            cases[f"{multiple}B_{delta:+d}"] = boundary_hist(
                experts, active, block_m, multiple, delta
            )

    kernel = ExpertKernel(
        hidden, intermediate, experts, active * 3 * block_m, seed=1100
    )
    records = []
    for round_id in range(args.rounds):
        names = list(cases)
        random.Random(1100 + round_id).shuffle(names)
        for name in names:
            counts = cases[name]
            timing = time_kernel(kernel, counts, args.warmup, args.reps)
            records.append(
                {
                    "round": round_id,
                    "name": name,
                    "counts": counts,
                    **features(counts, block_m),
                    **timing,
                }
            )

    payload = {
        "schema": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sonic_commit": subprocess.check_output(
            ["git", "-C", "/home/esjung/external/sonic-moe", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "quack": importlib.metadata.version("quack-kernels"),
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "apache_tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
            "cuda_python": importlib.metadata.version("cuda-python"),
            "gpu": torch.cuda.get_device_name(0),
            "physical_visible_device": 4,
        },
        "shape": {
            "dtype": "bfloat16",
            "H": hidden,
            "I": intermediate,
            "E": experts,
            "K_context": 8,
            "G": active,
            "primary_N": sum(aligned),
        },
        "runtime_config": {
            "gemm_config": repr(cfg),
            "logical_tile_m": cfg.tile_m,
            "cta_block_m": block_m,
            "source": "quack.gemm_config.default_config + cta_tile_shape_m",
            "tuned": False,
        },
        "warmup": args.warmup,
        "measured_reps": args.reps,
        "independent_rounds": args.rounds,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
