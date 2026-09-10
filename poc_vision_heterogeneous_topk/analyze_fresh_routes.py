#!/usr/bin/env python3
"""Fast routing-only P1/P2/P3 oracle over the fresh Qwen3-VL capture."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from poc_vision_heterogeneous_topk.analyze_policy_oracle import (
    budget_keep,
    random_keep,
    semantic_keep,
    tail_aware_keep,
)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    source = {x["sample_id"]: int(x["source_dp_rank"]) for x in manifest["schedule"]}
    rows = []
    for sample in manifest["samples"]:
        sid, n = sample["sample_id"], int(sample["prompt_tokens"])
        start, end = map(int, sample["visual_span"])
        modality = np.full(n, "other", dtype="<U6"); modality[start:end] = "vision"
        text_start, text_end = map(int, sample["post_visual_span"]); modality[text_start:text_end] = "text"
        for layer in manifest["policy"]["layers"]:
            parts = [np.load(args.capture / "raw" / f"router.{sid}.dp{source[sid]}.tp{tp}.layer{layer}.npz")
                     for tp in (0, 1)]
            ids = np.concatenate([x["topk_ids"] for x in parts], axis=0)[:n].astype(np.int64)
            weights = np.concatenate([x["topk_weights"] for x in parts], axis=0)[:n].astype(np.float32)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            base_load = np.bincount((ids // 32).reshape(-1), minlength=4)
            policies = {"full_k8": np.ones_like(ids, dtype=bool)}
            for k in (6, 4, 2, 1):
                policies[f"random_k{k}"] = random_keep(modality, k, 20260911 + layer)
                policies[f"semantic_k{k}"] = semantic_keep(weights, modality, k)
                policies[f"tail_k{k}"] = tail_aware_keep(ids, weights, modality, k, .01)
            for budget in (.01, .02, .05, .10, .20, .30, .40, .50):
                for policy in ("random", "semantic", "tail"):
                    policies[f"budget_{policy}_{budget:.2f}"] = budget_keep(ids, weights, modality, budget, policy)
            vision_total = int((modality == "vision").sum()) * 8
            for policy, keep in policies.items():
                kept_ranks = (ids[keep] // 32).reshape(-1)
                loads = np.bincount(kept_ranks, minlength=4)
                dropped = ~keep
                dropped_vision = dropped & (modality[:, None] == "vision")
                rows.append({"sample": sid, "category": sample["category"], "edge": sample["fixed_edge"],
                             "layer": layer, "policy": policy, "vision_assignments": vision_total,
                             "vision_drop_fraction": float(dropped_vision.sum() / max(vision_total, 1)),
                             "router_mass_drop_fraction": float(weights[dropped].sum() / max(weights.sum(), 1e-12)),
                             "baseline_max_rank": int(base_load.max()), "policy_max_rank": int(loads.max()),
                             "max_rank_reduction_pct": float(100 * (1 - loads.max() / base_load.max())),
                             "rank_loads": ":".join(map(str, loads.tolist()))})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fresh_route_policy_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for policy in sorted({x["policy"] for x in rows}):
        g = [x for x in rows if x["policy"] == policy]
        summary.append({"policy": policy, "sample_layers": len(g),
                        "vision_drop_fraction_median": float(np.median([x["vision_drop_fraction"] for x in g])),
                        "router_mass_drop_fraction_median": float(np.median([x["router_mass_drop_fraction"] for x in g])),
                        "max_rank_reduction_pct_median": float(np.median([x["max_rank_reduction_pct"] for x in g])),
                        "max_rank_reduction_pct_p10": float(np.quantile([x["max_rank_reduction_pct"] for x in g], .1))})
    with (args.output / "fresh_route_policy_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
