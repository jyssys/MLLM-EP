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

from scripts.replay_rank_local_compute import owner_local_expert_stage


def load_histograms(
    trace_dirs: list[Path], trace_format: str, ep: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rows = []
    for directory in trace_dirs:
        for path in sorted(directory.glob("request_*.npz")):
            with np.load(path, allow_pickle=False) as z:
                if trace_format == "selective":
                    key = "hist_s1"
                elif trace_format == "route_pruning":
                    key = "histogram"
                else:
                    key = "hist"
                histograms = z[key].reshape(-1, 256)
                if trace_format == "route_pruning":
                    unique = z[f"unique_ep{ep}"].reshape(-1, ep, ep).sum(axis=1)
                else:
                    unique = np.full((len(histograms), ep), -1, dtype=np.int64)
                for index, hist in enumerate(histograms):
                    rows.append((
                        f"{directory.parent.name}_{path.stem}_{index}",
                        hist.astype(np.int64), unique[index].astype(np.int64),
                    ))
    return rows


def diverse(cases: list[tuple[str, np.ndarray]], ep: int, maximum: int):
    local = []
    width = 256 // ep
    for label, hist, unique_by_rank in cases:
        for rank in range(ep):
            counts = hist[rank * width:(rank + 1) * width]
            active = counts[counts > 0]
            if len(active):
                local.append((label, rank, counts, int(unique_by_rank[rank]), np.asarray([
                    len(active), active.sum(), active.max(), active.std() / active.mean()
                ], dtype=float)))
    selected_indices = []; seen = set()
    for feature in range(4):
        order = sorted(range(len(local)), key=lambda i: local[i][4][feature])
        for q in np.linspace(0, 1, min(32, maximum // 4)):
            idx = order[round(q * (len(order) - 1))]
            if idx not in seen:
                seen.add(idx); selected_indices.append(idx)
    for idx in range(len(local)):
        if len(selected_indices) >= maximum: break
        if idx not in seen:
            seen.add(idx); selected_indices.append(idx)
    return [local[idx] for idx in selected_indices[:maximum]]


def build_recv_ids(counts: np.ndarray, unique_rows: int, top_k: int = 8) -> np.ndarray:
    """Construct a valid packed receive layout preserving rows and expert counts."""
    assignments = int(counts.sum())
    if unique_rows < 0:
        unique_rows = assignments
    if assignments > unique_rows * top_k or counts.max(initial=0) > unique_rows:
        raise ValueError(
            f"invalid packed shape: assignments={assignments}, unique_rows={unique_rows}, "
            f"max_count={counts.max(initial=0)}"
        )
    result = np.full((unique_rows, top_k), -1, dtype=np.int64)
    occupancy = np.zeros(unique_rows, dtype=np.int16)
    cursor = 0
    for expert, count in enumerate(counts):
        placed = 0
        examined = 0
        while placed < int(count):
            row = cursor % unique_rows
            cursor += 1
            examined += 1
            if occupancy[row] < top_k and not np.any(result[row] == expert):
                result[row, occupancy[row]] = expert
                occupancy[row] += 1
                placed += 1
            if examined > unique_rows * top_k * 2:
                raise RuntimeError("could not construct packed receive layout")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--trace-format", choices=("selective", "team", "route_pruning"), required=True)
    parser.add_argument("--ep", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
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
    cases = diverse(load_histograms(args.trace_dirs, args.trace_format, args.ep),
                    args.ep, args.maximum_cases)
    measured = []
    for case_id, (label, rank, counts, unique_rows, features) in enumerate(cases):
        assignments = int(counts.sum())
        recv_ids_np = build_recv_ids(counts, unique_rows)
        recv_ids = torch.as_tensor(recv_ids_np, device=device)
        recv_weights = torch.where(recv_ids >= 0, 1.0 / 8, 0.0).float()
        x = torch.randn(len(recv_ids_np), hidden, dtype=torch.bfloat16,
                        device=device, generator=generator) * .02
        for _ in range(args.warmup):
            owner_local_expert_stage(
                x, recv_ids, recv_weights, local_experts, gate_up, down,
            )
        torch.cuda.synchronize(); samples = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); output = owner_local_expert_stage(
                x, recv_ids, recv_weights, local_experts, gate_up, down,
            )
            end.record(); end.synchronize(); samples.append(start.elapsed_time(end))
        measured.append({
            "ep_size": args.ep, "sample_id": f"method_{case_id}_{label}_r{rank}",
            "request_id": -1, "block_id": -1, "iteration_id": -1, "nfe": -1,
            "layer_id": -1, "virtual_rank": rank,
            "active_experts": int(features[0]), "assignments": assignments,
            "recv_unique_activation_rows": len(recv_ids_np),
            "max_rows_per_expert": int(features[2]),
            "cv_rows_per_expert": float(features[3]),
            "latency_ms": float(np.median(samples)),
            "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90)), "split": "train",
            "backend": "DeepEP_receive_format_plus_torch_grouped_mm_bf16_MLP_route_pruning",
        })
        del x, recv_ids, recv_weights, output
    added_cases = len(measured)
    base_cases = 0
    if args.base_model:
        with args.base_model.open(newline="") as stream:
            base = list(csv.DictReader(stream))
        base_cases = len(base)
        measured = base + measured
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(measured[0]), lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(measured)
    args.output.with_suffix(".audit.json").write_text(json.dumps({
        "scope": "grouped routed expert MLP compute only", "ep": args.ep,
        "base_cases": base_cases, "added_cases": added_cases,
        "total_cases": len(measured), "local_experts": local_experts,
        "trace_format": args.trace_format,
        "excluded": ["dispatch", "combine", "TP", "replicated-state bridge"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
