#!/usr/bin/env python3
"""Recompute compact time-block model metrics from the final atlas.

This deliberately keeps the model descriptive.  It reports the incremental
held-out value of fanout after workload/rank-load controls and never treats a
row-level fit as a causal speedup claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_discovery import linear_metrics


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    numeric = [
        "M", "cuda_ms", "active_experts", "rank_max_mean", "rank_cv",
        "expert_cv", "fanout_mean", "fanout_f4", "deepep_dispatch_ms",
        "expert_ms", "deepep_combine_ms", "deepep_event_wait_ms",
    ]
    for i, r in enumerate(rows):
        for k in numeric:
            r[k] = float(r.get(k, 0) or 0)
        # Atlas rows are in source/time order; retain that order as the
        # deterministic time-block split key.
        r["timestamp_ns"] = i
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = load(args.input)
    models = {
        "model0_M": ["M"],
        "model1_distribution": ["M", "active_experts", "expert_cv"],
        "model2_distribution_plus_rank": [
            "M", "active_experts", "expert_cv", "rank_max_mean", "rank_cv"
        ],
        "model3_plus_fanout": [
            "M", "active_experts", "expert_cv", "rank_max_mean", "rank_cv",
            "fanout_mean", "fanout_f4"
        ],
    }
    result = {name: linear_metrics(rows, feats) for name, feats in models.items()}
    m2 = result["model2_distribution_plus_rank"]
    m3 = result["model3_plus_fanout"]
    if m2.get("status") == m3.get("status") == "OK":
        result["model2_to_model3_rmse_reduction_pct"] = (
            100.0 * (m2["rmse"] - m3["rmse"]) / m2["rmse"]
            if m2["rmse"] else None
        )
    result["rows"] = len(rows)
    result["split"] = "time_block_70_30_in_atlas_order"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
