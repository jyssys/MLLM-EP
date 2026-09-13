#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[tuple[str, str, str], list[tuple[str, dict]]] = defaultdict(list)
    for path in args.result_root.glob("pairwise/*/wave*/r*/pairwise_rank*.json"):
        payload = json.loads(path.read_text())
        parts = path.parts
        dataset = parts[-4]
        wave = parts[-3].removeprefix("wave")
        restart = parts[-2]
        for row in payload["rows"]:
            grouped[(dataset, wave, row["pair"])].append((restart, row))
    fields = [
        "dataset", "wave", "pair", "samples", "rank_critical_serial_median_ms",
        "rank_critical_concurrent_median_ms", "real_saving_median_ms",
        "real_saving_p10_ms", "real_saving_p90_ms", "eta_median",
        "compute_slowdown_median", "min_compute_cosine", "max_compute_rel_l2",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key, tagged_rows in sorted(grouped.items()):
            rows = [row for _, row in tagged_rows]
            by_iteration: dict[tuple[str, int], list[dict]] = defaultdict(list)
            for restart, row in tagged_rows:
                by_iteration[(restart, int(row["iteration"]))].append(row)
            critical = []
            for iteration_rows in by_iteration.values():
                critical.append({
                    "serial": max(float(x["serial_ms"]) for x in iteration_rows),
                    "concurrent": max(float(x["concurrent_ms"]) for x in iteration_rows),
                    "saving": max(float(x["serial_ms"]) for x in iteration_rows)
                    - max(float(x["concurrent_ms"]) for x in iteration_rows),
                })
            savings = [x["saving"] for x in critical]
            writer.writerow({
                "dataset": key[0], "wave": key[1], "pair": key[2],
                "samples": len(critical),
                "rank_critical_serial_median_ms": statistics.median(x["serial"] for x in critical),
                "rank_critical_concurrent_median_ms": statistics.median(x["concurrent"] for x in critical),
                "real_saving_median_ms": statistics.median(savings),
                "real_saving_p10_ms": quantile(savings, 0.1),
                "real_saving_p90_ms": quantile(savings, 0.9),
                "eta_median": statistics.median(float(x["eta"]) for x in rows),
                "compute_slowdown_median": statistics.median(
                    float(x.get("b_slowdown", 1.0) or 1.0) for x in rows
                ),
                "min_compute_cosine": min(float(x.get("compute_cosine", 1.0)) for x in rows),
                "max_compute_rel_l2": max(float(x.get("compute_rel_l2", 0.0)) for x in rows),
            })


if __name__ == "__main__":
    main()
