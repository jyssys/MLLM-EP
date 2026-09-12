#!/usr/bin/env python3
"""Aggregate matched vanilla EP1/2/4 clean runs without rank double counting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    weight = position - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = payload["stage_rows"]
    request_latencies = [float(x["elapsed_s"]) for x in payload["records"]]
    rank_imbalances = []
    for row in rows:
        counts = row["rank_assignment_counts"]
        mean = sum(counts) / len(counts) if counts else 0.0
        rank_imbalances.append(max(counts) / mean if mean else 0.0)
    match = re.search(r"ep(\d+)_restart(\d+)", path.name)
    if match is None:
        raise ValueError(path)
    output_blob = json.dumps(
        [(x["id"], x["output"]) for x in payload["records"]],
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    calls = max((int(row["call_id"]) for row in rows), default=-1) + 1
    return {
        "path": str(path),
        "ep": int(match.group(1)),
        "restart": int(match.group(2)),
        "requests": len(request_latencies),
        "request_wall_s": float(payload["request_wall_s"]),
        "request_mean_s": statistics.mean(request_latencies),
        "request_median_s": statistics.median(request_latencies),
        "request_p90_s": quantile(request_latencies, 0.90),
        "model_forwards": calls / 48,
        "assignments": sum(int(row["assignments"]) for row in rows),
        "remote_assignments": sum(int(row["remote_assignments_from_source"]) for row in rows),
        "remote_dispatch_bytes": sum(int(row["remote_dispatch_bytes_hidden"]) for row in rows),
        "remote_combine_bytes": sum(int(row["remote_combine_bytes_hidden"]) for row in rows),
        "fanout_median": statistics.median(
            int(row["destination_rank_fanout"]) for row in rows
        ),
        "max_over_mean_rank_load_median": statistics.median(rank_imbalances),
        "output_sha256": hashlib.sha256(output_blob).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--run-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    runs = [summarize(path) for path in sorted(args.input_dir.glob("*_rank0.json"))]
    args.run_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.run_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    by_ep: dict[int, list[dict]] = defaultdict(list)
    for run in runs:
        by_ep[run["ep"]].append(run)
    summary = {}
    for ep, group in sorted(by_ep.items()):
        summary[str(ep)] = {
            "restarts": len(group),
            "request_wall_median_s": statistics.median(x["request_wall_s"] for x in group),
            "request_median_of_run_medians_s": statistics.median(
                x["request_median_s"] for x in group
            ),
            "request_mean_of_run_means_s": statistics.mean(x["request_mean_s"] for x in group),
            "model_forwards_median": statistics.median(x["model_forwards"] for x in group),
            "remote_assignments_median": statistics.median(
                x["remote_assignments"] for x in group
            ),
            "remote_total_bytes_median": statistics.median(
                x["remote_dispatch_bytes"] + x["remote_combine_bytes"] for x in group
            ),
            "fanout_median": statistics.median(x["fanout_median"] for x in group),
            "output_hashes": sorted({x["output_sha256"] for x in group}),
        }
    all_hashes = {x["output_sha256"] for x in runs}
    summary["all_outputs_exactly_equal"] = len(all_hashes) == 1
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
