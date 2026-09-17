#!/usr/bin/env python3
"""Replay actual post-pruning owner-rank route cases with production contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.replay_rank_local_compute import owner_local_expert_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    torch.cuda.set_device(args.device_index)
    device = torch.device(f"cuda:{args.device_index}")
    generator = torch.Generator(device=device).manual_seed(20260918)
    local_experts, hidden, intermediate = 32, 2048, 512
    gate_up = torch.randn(
        local_experts, hidden, 2 * intermediate, dtype=torch.bfloat16,
        device=device, generator=generator,
    ) * .01
    down = torch.randn(
        local_experts, intermediate, hidden, dtype=torch.bfloat16,
        device=device, generator=generator,
    ) * .01
    rows = []
    for path in sorted(args.cases.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            recv_ids_np = z["recv_ids"]
            recv_weights_np = z["recv_weights"]
            counts = z["counts"]
            budget = float(z["budget"])
            source_invocation = int(z["source_invocation"])
            rank = int(z["virtual_rank"])
        recv_ids = torch.as_tensor(recv_ids_np, device=device).long()
        recv_weights = torch.as_tensor(recv_weights_np, device=device, dtype=torch.float32)
        hidden_input = torch.randn(
            len(recv_ids), hidden, dtype=torch.bfloat16,
            device=device, generator=generator,
        ) * .02
        for _ in range(args.warmup):
            owner_local_expert_stage(
                hidden_input, recv_ids, recv_weights, local_experts, gate_up, down
            )
        torch.cuda.synchronize(); samples = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = owner_local_expert_stage(
                hidden_input, recv_ids, recv_weights, local_experts, gate_up, down
            )
            end.record(); end.synchronize(); samples.append(start.elapsed_time(end))
        active = counts[counts > 0]
        rows.append({
            "ep_size": 8,
            "sample_id": f"lowutil_b{budget:g}_inv{source_invocation}_r{rank}",
            "request_id": -1, "block_id": -1, "iteration_id": -1,
            "nfe": -1, "layer_id": -1, "virtual_rank": rank,
            "active_experts": int(len(active)), "assignments": int(active.sum()),
            "recv_unique_activation_rows": int(len(recv_ids)),
            "max_rows_per_expert": int(active.max(initial=0)),
            "cv_rows_per_expert": float(active.std() / active.mean()) if len(active) else 0.0,
            "latency_ms": float(np.median(samples)),
            "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90)),
            "split": "train",
            "backend": "DeepEP_receive_format_plus_torch_grouped_mm_bf16_MLP_post_pruning",
        })
        del recv_ids, recv_weights, hidden_input, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.output.with_suffix(".audit.json").write_text(json.dumps({
        "scope": "actual post-pruning owner-local route format and grouped BF16 expert MLP",
        "cases": len(rows), "ep": 8, "local_experts": 32,
        "excluded": ["dispatch", "combine", "TP", "replicated-state bridge"],
        "generation_run": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
