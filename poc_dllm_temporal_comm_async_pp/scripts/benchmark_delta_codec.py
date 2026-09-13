#!/usr/bin/env python3
"""Measure unfused sender+receiver delta codec cost on one allowed H100.

This deliberately uses ordinary PyTorch operations: it is a conservative
feasible-cost diagnostic, not a claim about an optimized fused codec.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def timed(fn, warmup: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    values = torch.tensor(samples)
    return tuple(float(torch.quantile(values, q).item()) for q in (0.5, 0.9, 0.99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="32,64,128,256,512,1024,2048,4096")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("codec diagnostic expects exactly one visible allowed GPU")
    device = torch.device("cuda:0")
    rows = []
    for count in map(int, args.rows.split(",")):
        previous = torch.randn(count, args.hidden, device=device, dtype=torch.bfloat16)
        current = previous + 0.1 * torch.randn_like(previous)

        def fp8_codec():
            delta = current - previous
            packed = delta.to(torch.float8_e4m3fn)
            return previous + packed.to(torch.bfloat16)

        def int8_codec():
            delta = (current - previous).float()
            scale = delta.abs().amax(dim=1, keepdim=True).clamp_min_(1e-8) / 127.0
            packed = torch.round(delta / scale).clamp_(-127, 127).to(torch.int8)
            return previous + (packed.float() * scale).to(torch.bfloat16)

        reference = current.float()
        for codec, fn, bytes_per_element, metadata_bytes in (
            ("fp8", fp8_codec, 1.0, 0),
            ("int8_row_scaled", int8_codec, 1.0, 2 * count),
        ):
            reconstruction = fn().float()
            rel_l2 = float((reconstruction - reference).norm() / reference.norm())
            p50, p90, p99 = timed(fn, args.warmup, args.repeats)
            rows.append(
                {
                    "rows": count,
                    "hidden": args.hidden,
                    "codec": codec,
                    "wire_bytes": int(count * args.hidden * bytes_per_element + metadata_bytes),
                    "bf16_bytes": count * args.hidden * 2,
                    "codec_p50_ms": p50,
                    "codec_p90_ms": p90,
                    "codec_p99_ms": p99,
                    "reconstruction_rel_l2": rel_l2,
                    "evidence_boundary": "unfused PyTorch sender+receiver local cost",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
