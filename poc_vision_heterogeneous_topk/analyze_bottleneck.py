#!/usr/bin/env python3
"""Join clean request timing with observer-heavy layer/MoE attribution."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np


def median_by_length(summary: Path) -> dict[int, dict]:
    rows = json.loads(summary.read_text())["results"]
    output = {}
    for length in sorted({int(r["length"]) for r in rows}):
        selected = [r for r in rows if int(r["length"]) == length]
        output[length] = {
            "ttft_ms": float(np.median([r["ttft_p50_ms"] for r in selected])),
            "fleet_wall_ms": float(np.median([r["fleet_wall_ms"] for r in selected])),
            "prompt_throughput_tok_s": float(np.median(
                [r["prompt_throughput_tok_s"] for r in selected])),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--instrumented", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    clean = median_by_length(args.clean / "summary.json")
    observed = median_by_length(args.instrumented / "summary.json")
    wanted_m = {length // 2: length for length in observed}
    invocations = []
    with (args.instrumented / "trace" / "invocations.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("phase") == "prefill" and row.get("M") in wanted_m:
                invocations.append(row)
    result = []
    for m, length in sorted(wanted_m.items()):
        samples = []
        for dp in (0, 1):
            grouped: dict[str, list[dict]] = collections.defaultdict(list)
            for row in invocations:
                if int(row["M"]) == m and int(row["dp_rank"]) == dp:
                    grouped[row["route_id"]].append(row)
            critical = []
            for route_id, ranks in grouped.items():
                per_rank = []
                for row in ranks:
                    stages = {x["stage"]: float(x["cuda_ms"]) for x in row["stage_records"]}
                    per_rank.append({
                        "moe_ms": float(row["cuda_ms"]),
                        "layout_ms": stages.get("deepep_layout", 0.0),
                        "dispatch_ms": stages.get("deepep_dispatch", 0.0),
                        "expert_ms": stages.get("expert", 0.0),
                        "combine_ms": stages.get("deepep_combine", 0.0),
                        "rank_max_mean": float(row["rank_max_mean"]),
                    })
                critical.append((min(int(x["local_invocation_id"]) for x in ranks), {
                    key: max(x[key] for x in per_rank) for key in per_rank[0]
                }))
            critical.sort(key=lambda x: x[0])
            chunks = [critical[i:i + 48] for i in range(0, len(critical), 48)]
            for iteration, chunk in enumerate(chunks):
                if len(chunk) != 48 or iteration == 0:  # first shape use is warmup
                    continue
                sums = {key: sum(row[key] for _, row in chunk) for key in chunk[0][1]}
                samples.append({"dp": dp, "iteration": iteration, **sums})
        row = {"prompt_length": length, "per_tp_M": m, "samples": len(samples),
               "clean_ttft_ms": clean[length]["ttft_ms"],
               "instrumented_ttft_ms": observed[length]["ttft_ms"],
               "observer_ttft_tax_pct": 100 * (observed[length]["ttft_ms"] /
                                                  clean[length]["ttft_ms"] - 1)}
        for metric in ("moe_ms", "layout_ms", "dispatch_ms", "expert_ms", "combine_ms"):
            values = np.asarray([s[metric] for s in samples])
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p90"] = float(np.quantile(values, .9))
            row[f"{metric}_clean_ttft_share"] = float(np.median(values) / clean[length]["ttft_ms"])
        rank_values = np.asarray([s["rank_max_mean"] / 48 for s in samples])
        row["layer_mean_rank_max_mean_median"] = float(np.median(rank_values))
        result.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(result[0])
    with (args.output / "bottleneck_atlas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(result)
    (args.output / "summary.json").write_text(json.dumps({
        "rows": result,
        "method": "per-DP per-layer critical EP-rank, summed over 48 sequential layers",
        "warmup": "first 48-layer chunk per M/DP excluded",
        "warning": "stage durations come from observer-heavy same-device CUDA events; request timing is clean",
    }, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
