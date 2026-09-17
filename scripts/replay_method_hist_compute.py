#!/usr/bin/env python3
"""Measure grouped expert MLP envelopes for method-generated histograms.

This is a single-GPU target-rank replay.  It does not invent EP ranks or use
EP2/P scaling.  EP1 uses all 256 experts locally; EP4/EP8 replay one virtual
owner rank at a time with the same production grouped-MMLP contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.replay_rank_local_compute import grouped_mlp


def load_histograms(trace_dirs: list[Path], trace_format: str) -> list[tuple[str, np.ndarray]]:
    rows = []
    for directory in trace_dirs:
        for path in sorted(directory.glob("request_*.npz")):
            with np.load(path, allow_pickle=False) as z:
                key = "hist_s1" if trace_format == "selective" else "hist"
                for index, hist in enumerate(z[key].reshape(-1, 256)):
                    rows.append((f"{directory.parent.name}_{path.stem}_{index}", hist.astype(np.int64)))
    return rows


def diverse(cases: list[tuple[str, np.ndarray]], ep: int, maximum: int):
    local = []
    width = 256 // ep
    for label, hist in cases:
        for rank in range(ep):
            counts = hist[rank * width:(rank + 1) * width]
            active = counts[counts > 0]
            if len(active):
                local.append((label, rank, counts, np.asarray([
                    len(active), active.sum(), active.max(), active.std() / active.mean()
                ], dtype=float)))
    selected_indices = []; seen = set()
    for feature in range(4):
        order = sorted(range(len(local)), key=lambda i: local[i][3][feature])
        for q in np.linspace(0, 1, min(32, maximum // 4)):
            idx = order[round(q * (len(order) - 1))]
            if idx not in seen:
                seen.add(idx); selected_indices.append(idx)
    for idx in range(len(local)):
        if len(selected_indices) >= maximum: break
        if idx not in seen:
            seen.add(idx); selected_indices.append(idx)
    return [local[idx] for idx in selected_indices[:maximum]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--trace-format", choices=("selective", "team"), required=True)
    parser.add_argument("--ep", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--maximum-cases", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    torch.cuda.set_device(args.device_index); device = torch.device(f"cuda:{args.device_index}")
    local_experts = 256 // args.ep; hidden = 2048; intermediate = 512
    generator = torch.Generator(device=device).manual_seed(20260919 + args.ep)
    gate_up = torch.randn(local_experts, hidden, 2 * intermediate,
                          dtype=torch.bfloat16, device=device, generator=generator) * .01
    down = torch.randn(local_experts, intermediate, hidden,
                       dtype=torch.bfloat16, device=device, generator=generator) * .01
    cases = diverse(load_histograms(args.trace_dirs, args.trace_format),
                    args.ep, args.maximum_cases)
    measured = []
    for case_id, (label, rank, counts, features) in enumerate(cases):
        assignments = int(counts.sum())
        x = torch.randn(assignments, hidden, dtype=torch.bfloat16,
                        device=device, generator=generator) * .02
        for _ in range(args.warmup): grouped_mlp(x, counts, gate_up, down)
        torch.cuda.synchronize(); samples = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); output = grouped_mlp(x, counts, gate_up, down)
            end.record(); end.synchronize(); samples.append(start.elapsed_time(end))
        measured.append({
            "ep_size": args.ep, "sample_id": f"method_{case_id}_{label}_r{rank}",
            "request_id": -1, "block_id": -1, "iteration_id": -1, "nfe": -1,
            "layer_id": -1, "virtual_rank": rank,
            "active_experts": int(features[0]), "assignments": assignments,
            "recv_unique_activation_rows": -1,
            "max_rows_per_expert": int(features[2]),
            "cv_rows_per_expert": float(features[3]),
            "latency_ms": float(np.median(samples)),
            "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90)), "split": "train",
            "backend": "torch._grouped_mm_BF16_routed_MLP_histogram_replay",
        })
        del x, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measured[0])); writer.writeheader(); writer.writerows(measured)
    args.output.with_suffix(".audit.json").write_text(json.dumps({
        "scope": "grouped routed expert MLP compute only", "ep": args.ep,
        "measured_cases": len(measured), "local_experts": local_experts,
        "trace_format": args.trace_format,
        "excluded": ["dispatch", "combine", "TP", "replicated-state bridge"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
