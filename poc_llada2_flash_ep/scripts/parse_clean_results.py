#!/usr/bin/env python3
"""Parse clean dInfer logs into a reproducible batch-scaling table."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


POINT = re.compile(
    r"clean_(?P<topology>[^_]+(?:_[^_]+)?)_(?P<mode>dynamic\d+|naive)_b(?P<batch>\d+)_r(?P<repeat>\d+)\.log$"
)
ITER = re.compile(
    r"\[iter\s+(?P<offset>\d+)\]nfe=\s*(?P<nfe>\d+), token number=\s*(?P<tokens>\d+), "
    r"sample_time=(?P<seconds>[0-9.]+)"
)
TOTAL = re.compile(
    r"Forward:\s*(?P<nfe>\d+), Time:\s*(?P<seconds>[0-9.]+), "
    r"FPS:\s*(?P<fps>[0-9.]+)\([^)]+\), TPS:\s*(?P<tps>[0-9.]+)"
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = []
    invalid = []
    for path in sorted(args.log_dir.glob("clean_*.log")):
        match = POINT.match(path.name)
        if not match:
            continue
        text = path.read_text(errors="replace")
        namespaces = re.findall(r"Namespace\([^\n]+", text)
        expected_topology = match.group("topology")
        if not namespaces:
            continue
        actual_ep4 = "ep_size=4" in namespaces[-1] and "moe_a2a_backend='deepep'" in namespaces[-1]
        expected_ep4 = expected_topology == "ep4"
        if actual_ep4 != expected_ep4:
            invalid.append(f"{path}: expected {expected_topology}, actual namespace={namespaces[-1]}")
            continue
        totals = list(TOTAL.finditer(text))
        if not totals:
            continue
        total = totals[-1]
        iterations = list(ITER.finditer(text))
        sample_times = [float(item.group("seconds")) for item in iterations]
        token_counts = [int(item.group("tokens")) for item in iterations]
        nfe_counts = [int(item.group("nfe")) for item in iterations]
        batch = int(match.group("batch"))
        wall = float(total.group("seconds"))
        rows.append(
            {
                "topology": match.group("topology"),
                "mode": match.group("mode"),
                "batch": batch,
                "repeat": int(match.group("repeat")),
                "request_count": sum(min(batch, 32 - int(item.group("offset"))) for item in iterations),
                "batch_waves": len(iterations),
                "total_nfe": int(total.group("nfe")),
                "total_generated_tokens": sum(token_counts),
                "wall_seconds": wall,
                "forward_per_second": float(total.group("fps")),
                "generated_tokens_per_second": float(total.group("tps")),
                "request_throughput_per_second": 32.0 / wall,
                "wave_latency_p50_seconds": percentile(sample_times, 50),
                "wave_latency_p95_seconds": percentile(sample_times, 95),
                "mean_ms_per_forward": 1000.0 * wall / int(total.group("nfe")),
                "nfe_min": min(nfe_counts) if nfe_counts else 0,
                "nfe_max": max(nfe_counts) if nfe_counts else 0,
                "source_log": str(path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with args.output.open("w", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    invalid_path = args.output.with_name("INVALID_CLEAN_RUNS.txt")
    invalid_path.write_text("\n".join(invalid) + ("\n" if invalid else ""))


if __name__ == "__main__":
    main()
