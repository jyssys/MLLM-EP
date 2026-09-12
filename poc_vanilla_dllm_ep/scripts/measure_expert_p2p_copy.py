#!/usr/bin/env python3
"""Measure exact SDAR expert-copy cost and visible cost under useful compute."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch


EXPECTED_UUIDS = (
    "f217c8a0-1142-20f4-d84b-af29f3a47a0d",
    "a77f3471-67d4-20b0-9fab-e502d4de5adb",
)


def percentile(values, q):
    values = sorted(values)
    return values[min(int((len(values) - 1) * q), len(values) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 0,1")
    uuids = tuple(str(torch.cuda.get_device_properties(i).uuid) for i in range(2))
    if uuids != EXPECTED_UUIDS:
        raise SystemExit(f"unexpected physical mapping: {uuids}")
    if not torch.cuda.can_device_access_peer(0, 1):
        raise SystemExit("GPU0 cannot peer-access GPU1")

    expert_bytes = 3 * 2048 * 768 * 2
    results = []
    compute_a = torch.randn((8192, 2048), dtype=torch.float16, device="cuda:0")
    compute_b = torch.randn((2048, 768), dtype=torch.float16, device="cuda:0")
    compute_stream = torch.cuda.Stream(device=0)
    copy_stream = torch.cuda.Stream(device=1)
    for expert_count in (1, 2, 4, 8):
        elements = expert_bytes * expert_count // 2
        source = torch.randn(elements, dtype=torch.float16, device="cuda:0")
        destination = torch.empty(elements, dtype=torch.float16, device="cuda:1")

        def copy_once():
            with torch.cuda.device(1), torch.cuda.stream(copy_stream):
                destination.copy_(source, non_blocking=True)

        def compute_once():
            with torch.cuda.device(0), torch.cuda.stream(compute_stream):
                torch.mm(compute_a, compute_b)

        for _ in range(args.warmup):
            copy_once()
            compute_once()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        copy_ms = []
        compute_ms = []
        concurrent_ms = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            copy_once()
            torch.cuda.synchronize(1)
            copy_ms.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            compute_once()
            torch.cuda.synchronize(0)
            compute_ms.append((time.perf_counter() - start) * 1000)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            start = time.perf_counter()
            copy_once()
            compute_once()
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            concurrent_ms.append((time.perf_counter() - start) * 1000)
        copy_median = statistics.median(copy_ms)
        compute_median = statistics.median(compute_ms)
        concurrent_median = statistics.median(concurrent_ms)
        results.append({
            "expert_count": expert_count,
            "bytes": expert_bytes * expert_count,
            "copy_median_ms": copy_median,
            "copy_p90_ms": percentile(copy_ms, 0.90),
            "compute_median_ms": compute_median,
            "concurrent_median_ms": concurrent_median,
            "visible_copy_over_compute_ms": max(concurrent_median - compute_median, 0.0),
            "overlap_efficiency": (
                copy_median + compute_median - concurrent_median
            ) / max(min(copy_median, compute_median), 1e-12),
            "effective_copy_gbps": (expert_bytes * expert_count / 1e9)
            / (copy_median / 1000),
        })
    payload = {
        "physical_gpu_uuids": uuids,
        "expert_bytes": expert_bytes,
        "dtype": "fp16",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
