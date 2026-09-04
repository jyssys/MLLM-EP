"""Prepare a same-work EP fanout/communication-geometry replay.

All cases use identical M, top-k, hidden activation and total assignments.
The only changed factor is how many destination EP ranks receive each token.
Aggregate rank assignment counts are balanced for M divisible by four.
This is a route-ID diagnostic on the real DeepEP/TritonExperts path, not a
model-routing intervention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def build_case(m: int, fanout: int, layer: int) -> dict:
    if fanout not in (1, 2, 4):
        raise ValueError("fanout must be one of 1, 2, 4")
    routes = np.empty((m, 8), dtype=np.int64)
    for token in range(m):
        if fanout == 1:
            ranks = [token % 4]
            counts = [8]
        elif fanout == 2:
            ranks = [token % 4, (token + 1) % 4]
            counts = [4, 4]
        else:
            ranks = [0, 1, 2, 3]
            counts = [2, 2, 2, 2]
        offset = 0
        for rank, n in zip(ranks, counts):
            # Cycle local expert IDs so the aggregate rank work is equal,
            # while preserving the requested destination-rank fanout.
            ids = [(token * n + j) % 32 for j in range(n)]
            for local in ids:
                routes[token, offset] = rank * 32 + local
                offset += 1
    flat = routes.reshape(-1)
    counts = np.bincount(flat, minlength=128)
    rank = counts.reshape(4, 32).sum(axis=1)
    return {
        "case_id": f"balanced_fanout_F{fanout}_M{m}",
        "request_id": "fanout_control",
        "category": "controlled",
        "modality": "synthetic_route_diagnostic",
        "layer": layer, "M": m, "routes": routes.tolist(),
        "token_count": m, "total_assignments": int(flat.size),
        "fanout_ranks": fanout,
        "active_experts": int(np.count_nonzero(counts)),
        "rank_assignments": rank.astype(int).tolist(),
        "expert_counts": counts.astype(int).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--m-values", default="128,512,1024")
    ap.add_argument("--layer", type=int, default=24)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    ms = [int(x) for x in args.m_values.split(",") if x]
    cases = [build_case(m, f, args.layer) for m in ms for f in (1, 2, 4)]
    (args.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    rows = []
    for c in cases:
        rows.append({
            "case_id": c["case_id"], "M": c["M"], "fanout_ranks": c["fanout_ranks"],
            "total_assignments": c["total_assignments"], "active_experts": c["active_experts"],
            "rank_assignments": json.dumps(c["rank_assignments"]),
            "route_sha256": hashlib.sha256(np.asarray(c["routes"], dtype=np.int64).tobytes()).hexdigest(),
        })
    with (args.output / "route_statistics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "H6_fanout_communication_geometry",
        "M_values": ms, "fanout_values": [1, 2, 4], "top_k": 8,
        "global_experts": 128, "ep": 4, "invariants": ["M", "top_k", "total_assignments", "rank_assignments", "activation"],
        "changed_factor": "destination EP ranks per token", "layer": args.layer,
        "route_construction": "cyclic balanced rank destinations; local expert IDs cycle",
        "model_activation": "validated BF16 Qwen3-VL layer-24 capture rows",
        "placement": "expert_id // 32", "physical_gpus": [1, 2, 3, 4],
    }, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
