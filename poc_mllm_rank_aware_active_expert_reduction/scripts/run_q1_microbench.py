#!/usr/bin/env python3
"""Actual installed sparse Triton expert-kernel controlled sweeps.

This is rank-local physical kernel evidence. It intentionally does not label a
single-rank kernel call as physical EP communication or full-model latency.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.contract import ALLOWED_GPUS, atomic_json


def balanced_counts(paths: int, active: int) -> np.ndarray:
    counts = np.full(active, paths // active, dtype=np.int32)
    counts[: paths % active] += 1
    assert int(counts.sum()) == paths and np.count_nonzero(counts) == active
    return counts


def summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()), "median_ms": float(np.median(array)),
        "p10_ms": float(np.percentile(array, 10)), "p90_ms": float(np.percentile(array, 90)),
        "stdev_ms": float(array.std(ddof=1)), "samples_ms": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--local-experts", type=int, choices=(16, 32, 64), action="append")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.physical_gpu not in ALLOWED_GPUS:
        raise ValueError(args.physical_gpu)
    if "torch" in globals():
        raise RuntimeError("CUDA visibility must be configured before torch import")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)

    import torch
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    if torch.cuda.device_count() != 1:
        raise RuntimeError(torch.cuda.device_count())
    torch.manual_seed(3407)
    device = torch.device("cuda:0")
    local_expert_values = args.local_experts or [64, 32, 16]
    observations = []
    # 256 MiB is safely above H100's L2. Reading/writing it on the same stream
    # before the timed event provides a cold-weight-cache diagnostic without
    # charging the flush itself to expert execution.
    cache_flush = torch.zeros((128 * 1024 * 1024,), device=device, dtype=torch.bfloat16)
    # Actual checkpoint tensor layout: gate_up=[E,1536,2048], down=[E,2048,768].
    per_expert_elements = 1536 * 2048 + 2048 * 768
    per_expert_bytes = per_expert_elements * 2
    for local_experts in local_expert_values:
        w1 = torch.empty((local_experts, 1536, 2048), device=device,
                         dtype=torch.bfloat16).normal_(0, 0.01)
        w2 = torch.empty((local_experts, 2048, 768), device=device,
                         dtype=torch.bfloat16).normal_(0, 0.01)
        cases = []
        # Sweep A: fixed paths, varying distinct experts.
        for active in (2, 4, 8, 16):
            if active <= local_experts:
                cases.append(("A_fixed_32_paths", 32, active))
        # Sweep B: fixed distinct experts, varying paths.
        for paths in (8, 16, 32, 64):
            cases.append(("B_fixed_4_experts", paths, 4))
        # Extra joint grid supports an interpretable latency surface.
        for paths in (8, 16, 32, 64):
            for active in (2, 8, 16):
                if active <= paths and active <= local_experts:
                    cases.append(("joint_grid", paths, active))
        seen = set()
        cases = [case for case in cases if not (case in seen or seen.add(case))]
        for sweep, paths, active in cases:
            counts = balanced_counts(paths, active)
            ids = np.repeat(np.arange(active, dtype=np.int32), counts).reshape(-1, 1)
            topk_ids = torch.from_numpy(ids).to(device=device)
            topk_weights = torch.ones((paths, 1), device=device, dtype=torch.float32)
            hidden = torch.randn((paths, 2048), device=device, dtype=torch.bfloat16) * 0.02
            for _ in range(args.warmup):
                fused_experts(hidden, w1, w2, topk_weights, topk_ids, inplace=False)
            torch.cuda.synchronize()
            for cache_condition in ("hot_repeated", "cold_l2_flushed"):
                for round_index in range(args.rounds):
                    values = []
                    for _ in range(args.repeats):
                        # Identical large preconditioning kernel keeps clocks
                        # and HBM state comparable. The hot lane then executes
                        # one unmeasured expert call to repopulate its weights.
                        cache_flush.add_(1)
                        if cache_condition == "hot_repeated":
                            warm_result = fused_experts(
                                hidden, w1, w2, topk_weights, topk_ids, inplace=False)
                            del warm_result
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        result = fused_experts(hidden, w1, w2, topk_weights, topk_ids,
                                               inplace=False)
                        end.record(); end.synchronize()
                        values.append(float(start.elapsed_time(end)))
                        del result
                    observations.append({
                        "sweep": sweep, "cache_condition": cache_condition,
                        "local_experts": local_experts,
                        "token_expert_paths": paths, "active_experts": active,
                        "active_weight_bytes": active * per_expert_bytes,
                        "round": round_index, "expert_histogram": counts.tolist(),
                        **summary(values),
                    })
        del w1, w2
        torch.cuda.empty_cache()
    payload = {
        "schema_version": "rank_aware_q1_microbench.v1",
        "status": "MEASURED_ACTUAL_INSTALLED_FUSED_EXPERT_KERNEL",
        "evidence_scope": "RANK_LOCAL_KERNEL_ONLY_NOT_EP_COMMUNICATION_OR_FULL_MODEL",
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "shape": {"hidden": 2048, "intermediate": 768, "dtype": "bfloat16"},
        "per_expert_weight_bytes": per_expert_bytes,
        "cache_flush_bytes": int(cache_flush.numel() * cache_flush.element_size()),
        "warmup": args.warmup, "repeats": args.repeats, "rounds": args.rounds,
        "observations": observations,
    }
    atomic_json(args.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "observations"}
                     | {"observation_count": len(observations)}, indent=2))


if __name__ == "__main__":
    main()
