#!/usr/bin/env python3
"""Bounded post-gate diagnostics from exact live Qwen EP8 histograms.

The capture intentionally stores expert counts, not token identities.  This
script therefore evaluates *diagnostic upper bounds* for capacity/packing and
does not claim a routed-token mutation or a GPU speedup.  It also makes the
temporal and action-choice limitation explicit instead of fitting an RL policy
to an unobservable action outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _histograms(row: pd.Series) -> np.ndarray:
    data = json.loads(row.expert_histograms)
    return np.concatenate([np.asarray(data[str(i)], dtype=float) for i in range(8)])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args(); out = args.result
    inv = pd.read_csv(out / "invocation_metrics.csv")
    rows = []
    for _, row in inv.iterrows():
        hist = _histograms(row)
        rank = hist.reshape(8, 16).sum(axis=1)
        total = float(hist.sum())
        # Fixed route/placement (A0) is observed CUDA time.  Capacity factors
        # below are a declared *count-only* sensitivity model: they cap each
        # expert and preserve the rank mapping, so dropped mass is reported and
        # never interpreted as a correctness-preserving intervention.
        caps = {}
        for name, factor in (("A1_CAPACITY_MILD", 1.25), ("A2_CAPACITY_STRONG", 1.50)):
            caps[name] = int(np.ceil(total / 128.0 * factor))
        cap_loads = {name: np.minimum(hist, cap).reshape(8, 16).sum(axis=1)
                     for name, cap in caps.items()}
        # EPLB-like ideal packing is an upper-bound placement calculation only.
        # It sorts the exact expert counts and greedily assigns 16 experts/rank.
        bins = np.zeros(8, dtype=float)
        for expert_load in np.sort(hist)[::-1]:
            bins[int(np.argmin(bins))] += expert_load
        base = float(row.expert_max_ms)
        mean = float(row.expert_mean_ms)
        rec = {
            "request_id": row.request_id, "condition": row.condition,
            "wave": int(row.wave), "layer": int(row.layer),
            "observed_A0_expert_ms": base,
            "observed_expert_mean_ms": mean,
            "observed_rank_ratio": float(row.rank_ratio),
            "total_assignments": total,
            "capacity_mild_cap_assignments": caps["A1_CAPACITY_MILD"],
            "capacity_strong_cap_assignments": caps["A2_CAPACITY_STRONG"],
            "capacity_mild_drop_fraction": float(1 - np.minimum(hist, caps["A1_CAPACITY_MILD"]).sum() / max(total, 1)),
            "capacity_strong_drop_fraction": float(1 - np.minimum(hist, caps["A2_CAPACITY_STRONG"]).sum() / max(total, 1)),
            "A1_rank_ratio_proxy": float(cap_loads["A1_CAPACITY_MILD"].max() / max(cap_loads["A1_CAPACITY_MILD"].mean(), 1e-9)),
            "A2_rank_ratio_proxy": float(cap_loads["A2_CAPACITY_STRONG"].max() / max(cap_loads["A2_CAPACITY_STRONG"].mean(), 1e-9)),
            "EPLB_ideal_rank_ratio": float(bins.max() / max(bins.mean(), 1e-9)),
            "EPLB_ideal_load_reduction_proxy": float(1 - bins.max() / max(rank.max(), 1e-9)),
            "active_experts": int((hist > 0).sum()),
        }
        rows.append(rec)
    proxy = pd.DataFrame(rows)
    proxy.to_csv(out / "capacity_action_proxy.csv", index=False)

    # Route temporal persistence is exact for this fixed repeated schedule.
    temporal = pd.read_csv(out / "temporal_metrics.csv")
    temporal_summary = (temporal.groupby("condition", dropna=False)
                        .agg(layers=("layer", "size"),
                             hot_expert_recurrence=("hot_expert_recurrence", "median"),
                             hot_expert_unique=("hot_expert_unique", "median"))
                        .reset_index())
    temporal_summary.to_csv(out / "temporal_condition_summary.csv", index=False)

    # Do not select a learned policy.  This table is an explicit accounting of
    # what is and is not identifiable from the bounded capture.
    action = {
        "A0_NO_OP": {"status": "observed", "cost": "expert_max_ms"},
        "A1_CAPACITY_MILD": {"status": "count_proxy_only", "factor": 1.25},
        "A2_CAPACITY_STRONG": {"status": "count_proxy_only", "factor": 1.50},
        "A3_EPLB_SMALL": {"status": "not_measured", "reason": "no token IDs/weight migration trace"},
        "A4_EPLB_LARGE": {"status": "not_measured", "reason": "no token IDs/weight migration trace"},
    }
    (out / "action_evaluation_status.json").write_text(json.dumps(action, indent=2) + "\n")
    summary = {
        "stage0b_capacity_positive_control": "COUNT_PROXY_ONLY_NOT_GPU_VALIDATED",
        "stage1_temporal": "ROUTE_PERSISTENCE_MEASURED_FIXED_REPEATED_PROMPTS",
        "stage2_dynamic_oracle": "NOT_RUN_VALID_ACTION_COST_UNAVAILABLE",
        "stage3_rl": "NOT_RUN",
        "reason": "The read-only live artifact has exact local expert histograms and timings but not token IDs, alternate routing outcomes, or expert-weight migration timings. Claiming a capacity/EPLB GPU gain would violate the bounded evidence requirement.",
        "median_ideal_EPLB_rank_ratio": float(proxy.EPLB_ideal_rank_ratio.median()),
        "median_ideal_EPLB_load_reduction_proxy": float(proxy.EPLB_ideal_load_reduction_proxy.median()),
        "median_capacity_mild_drop_fraction": float(proxy.capacity_mild_drop_fraction.median()),
        "median_capacity_strong_drop_fraction": float(proxy.capacity_strong_drop_fraction.median()),
        "median_hot_expert_recurrence": float(temporal.hot_expert_recurrence.median()) if len(temporal) else None,
    }
    (out / "gated_stage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
