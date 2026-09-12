#!/usr/bin/env python3
"""Join same-shape EP1/2/4 replay rows and calculate a zero-switching oracle."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


STAGES = ("router", "prepare", "dispatch", "expert", "combine", "moe")


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    fraction = position - lo
    return values[lo] * (1 - fraction) + values[hi] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    rows_out = []
    per_ep_m: dict[tuple[int, int], list[float]] = defaultdict(list)
    for rank0_path in sorted(args.input_dir.glob("ep*_rank0.json")):
        rank0 = json.loads(rank0_path.read_text())
        ep = int(rank0["topology"]["ep"])
        payloads = [
            json.loads(path.read_text())
            for path in sorted(args.input_dir.glob(rank0_path.name.replace("_rank0.json", "_rank*.json")))
        ]
        joined: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for payload in payloads:
            for row in payload["stage_rows"]:
                joined[(int(row["microbenchmark_m"]), int(row["call_id"]))].append(row)
        samples: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        imbalance: dict[int, list[float]] = defaultdict(list)
        for (m, _), rank_rows in joined.items():
            for stage in STAGES:
                samples[m][stage].append(max(float(row[f"{stage}_ms"]) for row in rank_rows))
            counts = rank_rows[0]["rank_assignment_counts"]
            imbalance[m].append(max(counts) / max(sum(counts) / ep, 1e-12))
        for m, stages in sorted(samples.items()):
            row = {"ep": ep, "m": m, "repeats": len(stages["moe"])}
            for stage in STAGES:
                row[f"{stage}_p50_ms"] = statistics.median(stages[stage])
                row[f"{stage}_p90_ms"] = percentile(stages[stage], 0.90)
            row["max_over_mean_rank_load_p50"] = statistics.median(imbalance[m])
            rows_out.append(row)
            per_ep_m[(ep, m)] = stages["moe"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    medians = {
        (ep, m): statistics.median(values) for (ep, m), values in per_ep_m.items()
    }
    by_m = {}
    for m in sorted({m for _, m in medians}):
        available = {ep: medians[(ep, m)] for ep in (1, 2, 4) if (ep, m) in medians}
        winner = min(available, key=available.get)
        by_m[str(m)] = {
            "winner_ep": winner,
            "latency_ms": available,
            "best_ms": available[winner],
        }
    best_static = min(
        (1, 2, 4),
        key=lambda ep: sum(medians[(ep, m)] for m in {m for _, m in medians}),
    )
    static_total = sum(medians[(best_static, m)] for m in {m for _, m in medians})
    oracle_total = sum(min(medians[(ep, m)] for ep in (1, 2, 4)) for m in {m for _, m in medians})
    summary = {
        "best_static_ep_equal_weight_shapes": best_static,
        "best_static_total_ms": static_total,
        "zero_switching_oracle_total_ms": oracle_total,
        "zero_switching_oracle_gain_fraction": 1 - oracle_total / static_total,
        "by_m": by_m,
        "warning": "Operator-only equal-shape oracle; request weighting and topology transition costs are not included.",
    }
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
