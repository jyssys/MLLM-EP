"""Prepare uniform versus skewed per-expert distributions at fixed M/F/A."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def _assignment_list(total: int, skew: bool, seed: int) -> list[int]:
    if not skew:
        values = [i % 16 for i in range(total)]
    else:
        # Same 16 active experts and same total/rank load, but a long-tail
        # distribution.  Counts are deterministic and sum exactly to total.
        weights = np.asarray([420, 200, 120, 80, 60, 40, 30, 20, 15, 10, 8, 6, 5, 4, 3, 3], dtype=float)
        counts = np.floor(weights / weights.sum() * total).astype(int)
        counts[0] += total - int(counts.sum())
        values = [expert for expert, count in enumerate(counts) for _ in range(int(count))]
        random.Random(seed).shuffle(values)
        # Avoid duplicate expert IDs within a token's two assignments.
        for i in range(0, len(values) - 1, 2):
            if values[i] == values[i + 1]:
                for j in range(i + 2, len(values)):
                    if values[j] != values[i] and values[j] != values[i + 1]:
                        values[i + 1], values[j] = values[j], values[i + 1]
                        break
    return values


def build_case(m: int, shape: str, layer: int) -> dict:
    routes = np.empty((m, 8), dtype=np.int64)
    for rank in range(4):
        values = _assignment_list(m * 2, shape == "skew", 1000 + rank + m)
        for token in range(m):
            pair = values[2 * token : 2 * token + 2]
            if pair[0] == pair[1]:
                pair[1] = (pair[1] + 1) % 16
            # F4: two assignments to each EP rank, fixed rank load.
            routes[token, 2 * rank] = rank * 32 + pair[0]
            routes[token, 2 * rank + 1] = rank * 32 + pair[1]
    flat = routes.reshape(-1); counts = np.bincount(flat, minlength=128)
    rank_counts = counts.reshape(4, 32).sum(axis=1)
    return {
        "case_id": f"distribution_M{m}_F4_A16_{shape}", "request_id": "distribution_control",
        "category": "controlled", "modality": "synthetic_route_diagnostic", "layer": layer,
        "M": m, "routes": routes.tolist(), "token_count": m, "total_assignments": int(flat.size),
        "fanout_ranks": 4, "active_experts_per_rank": 16, "active_experts": int(np.count_nonzero(counts)),
        "distribution_shape": shape, "rank_assignments": rank_counts.astype(int).tolist(),
        "expert_counts": counts.astype(int).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--m-values", default="128,512"); ap.add_argument("--layer", type=int, default=24)
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    ms = [int(x) for x in args.m_values.split(",") if x]
    cases = [build_case(m, shape, args.layer) for m in ms for shape in ("uniform", "skew")]
    random.Random(5150).shuffle(cases)
    (args.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "H7_expert_token_distribution_shape", "M_values": ms,
        "shapes": ["uniform", "skew"], "fanout": 4, "active_per_rank": 16,
        "invariants": ["M", "top_k", "total_assignments", "rank_assignments", "activation"],
        "physical_gpus": [1, 2, 3, 4], "layer": args.layer,
    }, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__": main()
