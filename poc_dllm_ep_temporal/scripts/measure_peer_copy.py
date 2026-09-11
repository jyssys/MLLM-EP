#!/usr/bin/env python3
"""Measure expert-sized peer copies and their visible overlap cost.

Run only as:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python .../measure_peer_copy.py --output ...
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import statistics

import torch


EXPECTED_VISIBLE = "4,5,6,7"


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    alpha = index - lower
    return ordered[lower] * (1 - alpha) + ordered[upper] * alpha


def elapsed_on_destination(action, device: int) -> float:
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            start.record(stream)
            action(stream)
            end.record(stream)
        end.synchronize()
        return float(start.elapsed_time(end))


def copy_sample(source: torch.Tensor, destination: torch.Tensor, device: int) -> float:
    return elapsed_on_destination(lambda _: destination.copy_(source, non_blocking=True), device)


def compute_sample(left: torch.Tensor, right: torch.Tensor, repeats: int, device: int) -> float:
    def run(_):
        value = left
        for _ in range(repeats):
            value = torch.matmul(value, right)
        value.sum()
    return elapsed_on_destination(run, device)


def joined_sample(source, destination, left, right, repeats, device):
    with torch.cuda.device(device):
        copy_stream = torch.cuda.Stream(device=device)
        compute_stream = torch.cuda.Stream(device=device)
        join_stream = torch.cuda.Stream(device=device)
        start = torch.cuda.Event(enable_timing=True)
        copy_done = torch.cuda.Event()
        compute_done = torch.cuda.Event()
        end = torch.cuda.Event(enable_timing=True)
        start.record(join_stream)
        copy_stream.wait_event(start)
        compute_stream.wait_event(start)
        with torch.cuda.stream(copy_stream):
            destination.copy_(source, non_blocking=True)
            copy_done.record(copy_stream)
        with torch.cuda.stream(compute_stream):
            value = left
            for _ in range(repeats):
                value = torch.matmul(value, right)
            value.sum()
            compute_done.record(compute_stream)
        join_stream.wait_event(copy_done)
        join_stream.wait_event(compute_done)
        end.record(join_stream)
        end.synchronize()
        return float(start.elapsed_time(end))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--compute-repeats", type=int, default=8)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise SystemExit(f"refusing GPU launch: CUDA_VISIBLE_DEVICES must equal {EXPECTED_VISIBLE}")
    if torch.cuda.device_count() != 4:
        raise SystemExit("expected exactly four visible GPUs")

    # BF16 w13 + w2 bytes for one immutable expert.
    elements = 3 * args.hidden_size * args.intermediate_size
    rows = []
    for src in range(4):
        source = torch.empty(elements, dtype=torch.bfloat16, device=f"cuda:{src}").normal_()
        for dst in range(4):
            if src == dst:
                continue
            destination = torch.empty_like(source, device=f"cuda:{dst}")
            left = torch.randn(1024, args.hidden_size, dtype=torch.bfloat16, device=f"cuda:{dst}")
            right = torch.randn(args.hidden_size, args.hidden_size, dtype=torch.bfloat16, device=f"cuda:{dst}")
            for _ in range(args.warmup):
                copy_sample(source, destination, dst)
                compute_sample(left, right, args.compute_repeats, dst)
                joined_sample(source, destination, left, right, args.compute_repeats, dst)
            copies = [copy_sample(source, destination, dst) for _ in range(args.repetitions)]
            computes = [compute_sample(left, right, args.compute_repeats, dst) for _ in range(args.repetitions)]
            concurrent = [joined_sample(source, destination, left, right, args.compute_repeats, dst) for _ in range(args.repetitions)]
            copy_median = statistics.median(copies)
            compute_median = statistics.median(computes)
            concurrent_median = statistics.median(concurrent)
            rows.append({
                "physical_src": src + 4,
                "physical_dst": dst + 4,
                "expert_bytes": elements * 2,
                "copy_p50_ms": copy_median,
                "copy_p90_ms": percentile(copies, 0.9),
                "bandwidth_GBps": elements * 2 / copy_median / 1e6,
                "compute_p50_ms": compute_median,
                "concurrent_p50_ms": concurrent_median,
                "visible_copy_p50_ms": max(0.0, concurrent_median - compute_median),
            })
            del destination, left, right
        del source

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
