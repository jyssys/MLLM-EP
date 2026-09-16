#!/usr/bin/env python3
"""Extend rank-local compute calibration to current-block and batched shapes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.replay_rank_local_compute import owner_local_expert_stage
from virtual_ep.discovery_trace import DiscoveryTrace


def current_routes(trace, index):
    valid = trace.arrays["current_position_class"][index] >= 2
    return trace.arrays["current_expert_ids"][index][valid]


def cases(trace: DiscoveryTrace, ep: int) -> list[dict]:
    rng = np.random.default_rng(20260917 + ep)
    invocations = [current_routes(trace, i) for i in range(len(trace.arrays["request_id"]))]
    route_sets = [(f"single_{i}", routes) for i, routes in enumerate(invocations)]
    for scale in (2, 4):
        for index in range(0, len(invocations) - scale + 1, max(scale, len(invocations) // 200)):
            picked = rng.choice(len(invocations), size=scale, replace=False)
            route_sets.append((f"combined{scale}_{index}", np.concatenate([invocations[i] for i in picked])))
    all_cases = []
    local_experts = 256 // ep
    for sample_id, routes in route_sets:
        owners = routes // local_experts
        for rank in range(ep):
            received = np.any(owners == rank, axis=1)
            recv_ids = np.where(owners[received] == rank, routes[received] - rank * local_experts, -1).astype(np.int16)
            expert = recv_ids[recv_ids >= 0]
            counts = np.bincount(expert, minlength=local_experts)
            active = counts[counts > 0]
            if counts.sum():
                all_cases.append({
                    "sample_id": sample_id, "rank": rank, "recv_ids": recv_ids,
                    "counts": counts, "assignments": int(counts.sum()),
                    "active_experts": int(len(active)), "max_rows_per_expert": int(active.max()),
                    "cv_rows_per_expert": float(active.std() / active.mean()),
                })
    # Deterministic quantile coverage over four shape features.
    chosen = []; seen = set()
    for field in ("assignments", "active_experts", "max_rows_per_expert", "cv_rows_per_expert"):
        order = sorted(range(len(all_cases)), key=lambda i: all_cases[i][field])
        for q in np.linspace(0, 1, 64):
            index = order[round(q * (len(order) - 1))]
            if index not in seen:
                seen.add(index); chosen.append(all_cases[index])
    return chosen[:256]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ep", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device-index", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    torch.cuda.set_device(args.device_index); device = torch.device(f"cuda:{args.device_index}")
    trace = DiscoveryTrace.load(args.trace)
    local_experts = 256 // args.ep
    generator = torch.Generator(device=device).manual_seed(20260918 + args.ep)
    gate_up = torch.randn(local_experts, 2048, 1024, dtype=torch.bfloat16,
                          device=device, generator=generator) * .01
    down = torch.randn(local_experts, 512, 2048, dtype=torch.bfloat16,
                       device=device, generator=generator) * .01
    measured = []
    for case_id, case in enumerate(cases(trace, args.ep)):
        recv_ids = torch.as_tensor(case["recv_ids"], device=device).long()
        recv_weights = torch.where(recv_ids >= 0, 1.0 / 8, 0.0).float()
        x = torch.randn(len(recv_ids), 2048, dtype=torch.bfloat16, device=device,
                        generator=generator) * .02
        for _ in range(args.warmup):
            owner_local_expert_stage(x, recv_ids, recv_weights, local_experts, gate_up, down)
        torch.cuda.synchronize(); samples = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); output = owner_local_expert_stage(
                x, recv_ids, recv_weights, local_experts, gate_up, down
            ); end.record(); end.synchronize(); samples.append(start.elapsed_time(end))
        measured.append({
            "ep_size": args.ep, "sample_id": f"discovery_{case['sample_id']}_rank{case['rank']}",
            "request_id": -1, "block_id": -1, "iteration_id": -1, "nfe": -1,
            "layer_id": -1, "virtual_rank": case["rank"],
            "active_experts": case["active_experts"], "assignments": case["assignments"],
            "recv_unique_activation_rows": len(recv_ids),
            "max_rows_per_expert": case["max_rows_per_expert"],
            "cv_rows_per_expert": case["cv_rows_per_expert"],
            "latency_ms": float(np.median(samples)), "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90)), "split": "discovery_shape_calibration",
            "backend": "DeepEP_receive_format_plus_torch_grouped_mm_bf16_MLP",
        })
        del x, recv_ids, recv_weights, output
    with args.base_model.open(newline="") as stream:
        base = list(csv.DictReader(stream))
    rows = base + measured
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    args.output.with_suffix(".audit.json").write_text(json.dumps({
        "scope": "base full-row model plus measured current-block/combined shapes",
        "ep": args.ep, "base_samples": len(base), "added_samples": len(measured),
        "assignment_range_added": [min(row["assignments"] for row in measured),
                                   max(row["assignments"] for row in measured)],
        "backend": "torch._grouped_mm BF16 full routed MLP",
        "excluded": ["dispatch", "combine", "replicated-state bridge all-gather"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
