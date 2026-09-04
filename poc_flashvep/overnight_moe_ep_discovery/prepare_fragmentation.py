"""Prepare a balanced-rank, within-rank expert-fragmentation replay.

The cases keep M, top-k, total assignments, and destination-rank counts fixed;
only the number of active experts inside each destination rank changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def build_cases(m: int = 512, active_values: tuple[int, ...] = (1, 2, 4, 8, 16)) -> list[dict]:
    cases: list[dict] = []
    for active in active_values:
        routes = np.empty((m, 8), dtype=np.int64)
        for token in range(m):
            for rank in range(4):
                if active == 1:
                    local = (0, 0)
                else:
                    first = (2 * token) % active
                    local = (first, (first + 1) % active)
                routes[token, 2 * rank] = rank * 32 + local[0]
                routes[token, 2 * rank + 1] = rank * 32 + local[1]
        flat = routes.reshape(-1)
        counts = np.bincount(flat, minlength=128)
        rank = counts.reshape(4, 32).sum(axis=1)
        cases.append({
            "case_id": f"balanced_rank_A{active}_M{m}",
            "request_id": "fragmentation_control",
            "category": "controlled",
            "modality": "synthetic_route_diagnostic",
            "layer": 24,
            "M": m,
            "routes": routes.tolist(),
            "token_count": m,
            "total_assignments": int(flat.size),
            "active_experts": int(np.count_nonzero(counts)),
            "active_experts_per_rank": active,
            "rank_assignments": rank.astype(int).tolist(),
            "expert_counts": counts.astype(int).tolist(),
        })
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--m", type=int, default=512)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cases = build_cases(args.m)
    (args.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    rows = []
    for case in cases:
        rows.append({
            k: case[k] for k in (
                "case_id", "M", "token_count", "total_assignments",
                "active_experts", "active_experts_per_rank",
            )
        } | {
            "rank_assignments": json.dumps(case["rank_assignments"]),
            "route_sha256": hashlib.sha256(
                np.asarray(case["routes"], dtype=np.int64).tobytes()
            ).hexdigest(),
        })
    with (args.output / "route_statistics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "H01_balance_fragmentation_paradox",
        "m": args.m, "active_experts_per_rank": [1, 2, 4, 8, 16],
        "top_k": 8, "global_experts": 128, "ep": 4,
        "invariants": ["M", "total_assignments", "rank_assignments", "hidden_size", "dtype"],
        "changed_factor": "active experts within each rank",
        "route_construction": "two assignments per destination rank per token; cyclic balanced IDs",
        "model_activation": "validated BF16 Qwen3-VL layer-24 capture rows",
        "placement": "expert_id // 32", "physical_gpus": [1, 2, 3, 4],
    }, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
