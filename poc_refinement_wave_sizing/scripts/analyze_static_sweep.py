#!/usr/bin/env python3
"""Parse clean submitted-batch wave-size runs and GPU sampler logs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


LOG_RE = re.compile(
    r"clean_(?P<topology>ep4|tp4)_b(?P<batch>\d+)_mini(?P<mini>\d+)_r(?P<repeat>[^_]+)_g(?P<generation>\d+)\.log$"
)
TOTAL_RE = re.compile(
    r"Forward:\s*(?P<nfe>\d+), Time:\s*(?P<wall>[0-9.]+), "
    r"FPS:\s*(?P<fps>[0-9.]+)\([^)]+\), TPS:\s*(?P<tps>[0-9.]+)"
)


def read_gpu_log(path: Path) -> tuple[float, float, float]:
    memory: list[float] = []
    utilization: list[float] = []
    if not path.exists():
        return float("nan"), float("nan"), float("nan")
    for row in csv.reader(path.open()):
        if len(row) < 6:
            continue
        try:
            used = float(row[3].strip())
            util = float(row[5].strip())
        except ValueError:
            continue
        if used > 50_000:
            memory.append(used)
            utilization.append(util)
    if not memory:
        return float("nan"), float("nan"), float("nan")
    return max(memory) / 1024, float(np.median(utilization)), float(np.percentile(utilization, 95))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_root", type=Path)
    args = parser.parse_args()
    log_dir = args.task_root / "logs"
    rows: list[dict] = []
    for path in sorted(log_dir.glob("clean_*.log")):
        match = LOG_RE.match(path.name)
        if not match:
            continue
        totals = list(TOTAL_RE.finditer(path.read_text(errors="replace")))
        if not totals:
            continue
        total = totals[-1]
        groups = match.groupdict()
        gpu_path = log_dir / path.name.replace("clean_", "gpu_").replace(".log", ".csv")
        peak_hbm, util_median, util_p95 = read_gpu_log(gpu_path)
        wall = float(total.group("wall"))
        rows.append(
            {
                "topology": groups["topology"],
                "submitted_batch": int(groups["batch"]),
                "mini_batch_size": int(groups["mini"]),
                "physical_m_full_wave": int(groups["mini"]) * 32,
                "repeat": groups["repeat"],
                "generation": int(groups["generation"]),
                "nfe": int(total.group("nfe")),
                "bct_seconds": wall,
                "request_throughput_per_second": int(groups["batch"]) / wall,
                "model_forwards_per_second": float(total.group("fps")),
                "generated_tokens_per_second": float(total.group("tps")),
                "peak_hbm_gib_per_rank": peak_hbm,
                "gpu_util_median_percent": util_median,
                "gpu_util_p95_percent": util_p95,
                "source_log": str(path),
            }
        )
    write_csv(args.task_root / "STATIC_WAVE_POINTS.csv", rows)

    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["topology"], row["submitted_batch"], row["mini_batch_size"])].append(row)
    summary: list[dict] = []
    for (topology, batch, mini), values in sorted(grouped.items()):
        bct = np.asarray([float(row["bct_seconds"]) for row in values])
        tps = np.asarray([float(row["generated_tokens_per_second"]) for row in values])
        nfe = np.asarray([float(row["nfe"]) for row in values])
        summary.append(
            {
                "topology": topology,
                "submitted_batch": batch,
                "mini_batch_size": mini,
                "physical_m_full_wave": mini * 32,
                "restarts": len(values),
                "bct_median_seconds": float(np.median(bct)),
                "bct_min_seconds": float(np.min(bct)),
                "bct_max_seconds": float(np.max(bct)),
                "throughput_median_tokens_per_second": float(np.median(tps)),
                "nfe_median": float(np.median(nfe)),
                "peak_hbm_max_gib_per_rank": float(np.nanmax([row["peak_hbm_gib_per_rank"] for row in values])),
                "gpu_util_median_percent": float(np.nanmedian([row["gpu_util_median_percent"] for row in values])),
            }
        )
    write_csv(args.task_root / "STATIC_WAVE_SUMMARY.csv", summary)


if __name__ == "__main__":
    main()
