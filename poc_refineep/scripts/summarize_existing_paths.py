#!/usr/bin/env python3
"""Aggregate per-rank DeepEP results using critical-rank request timing."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob("rank*.jsonl")):
        with path.open() as stream:
            rows.extend(json.loads(line) for line in stream)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["policy"], row["repeat"])].append(row)
    request_rows = []
    for (case_id, policy, repeat), ranks in groups.items():
        if len(ranks) != 4:
            raise RuntimeError(f"missing rank for {(case_id, policy, repeat)}")
        critical = max(ranks, key=lambda row: row["comm_semantics_ms"])
        request_rows.append({
            **{key: critical[key] for key in (
                "case_id", "dataset", "wave", "layer", "phase", "shape_class",
                "global_fresh_m", "max_tokens_contract", "policy", "repeat"
            )},
            "layout_ms": max(row["layout_ms"] for row in ranks),
            "dispatch_ms": max(row["dispatch_ms"] for row in ranks),
            "weight_prep_ms": max(row["weight_prep_ms"] for row in ranks),
            "combine_ms": max(row["combine_ms"] for row in ranks),
            "comm_semantics_ms": max(row["comm_semantics_ms"] for row in ranks),
            "max_relative_l2": max(row["relative_l2_vs_normal"] for row in ranks),
            "max_relative_l2_vs_identity": max(row["relative_l2_vs_identity"] for row in ranks),
            "source_assignments_global": sum(row["source_assignments"] for row in ranks),
            "received_assignments_global": sum(row["received_assignments"] for row in ranks),
            "critical_rank": critical["rank"],
        })
    case_groups = defaultdict(list)
    for row in request_rows:
        case_groups[(row["case_id"], row["policy"])].append(row)
    summary = []
    metric_names = ["layout_ms", "dispatch_ms", "weight_prep_ms", "combine_ms", "comm_semantics_ms"]
    for (case_id, policy), values in sorted(case_groups.items()):
        first = values[0]
        summary.append({
            **{key: first[key] for key in (
                "case_id", "dataset", "wave", "layer", "phase", "shape_class",
                "global_fresh_m", "max_tokens_contract", "policy"
            )},
            **{name: median(row[name] for row in values) for name in metric_names},
            "p90_comm_semantics_ms": sorted(row["comm_semantics_ms"] for row in values)[int(0.9 * (len(values)-1))],
            "max_relative_l2": max(row["max_relative_l2"] for row in values),
            "max_relative_l2_vs_identity": max(row["max_relative_l2_vs_identity"] for row in values),
            "source_assignments_global": first["source_assignments_global"],
            "received_assignments_global": first["received_assignments_global"],
            "assignment_count_match": first["source_assignments_global"] == first["received_assignments_global"],
            "repeats": len(values),
            "timing_unit": "critical-rank wall via same-GPU CUDA events",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps({"raw_rank_rows": len(rows), "request_repeats": len(request_rows), "summary_rows": len(summary)}))


if __name__ == "__main__":
    main()
