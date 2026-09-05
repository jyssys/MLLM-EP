"""Prepare a randomized, factorial M x fanout x active-expert replay grid.

The generated routes keep hidden activations, top-k, total assignments, and
aggregate EP-rank load fixed.  Fanout controls the number of destination
ranks per token; active controls the number of local experts exercised per
rank.  This is a route-shape diagnostic on the real DeepEP/TritonExperts
path, not a model-routing intervention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def build_case(m: int, fanout: int, active: int, layer: int) -> dict:
    if fanout not in (1, 2, 4):
        raise ValueError("fanout must be 1, 2, or 4")
    if active < 8 or active > 32 or active & (active - 1):
        raise ValueError("active must be one of 8, 16, 32")
    routes = np.empty((m, 8), dtype=np.int64)
    for token in range(m):
        if fanout == 1:
            ranks, per_rank = [token % 4], [8]
        elif fanout == 2:
            ranks, per_rank = [token % 4, (token + 1) % 4], [4, 4]
        else:
            ranks, per_rank = [0, 1, 2, 3], [2, 2, 2, 2]
        offset = 0
        for rank, count in zip(ranks, per_rank, strict=True):
            start = (token * count + rank) % active
            ids = [(start + j) % active for j in range(count)]
            for local in ids:
                routes[token, offset] = rank * 32 + local
                offset += 1
    flat = routes.reshape(-1)
    counts = np.bincount(flat, minlength=128)
    rank_counts = counts.reshape(4, 32).sum(axis=1)
    return {
        "case_id": f"regime_M{m}_F{fanout}_A{active}",
        "request_id": "regime_grid",
        "category": "controlled",
        "modality": "synthetic_route_diagnostic",
        "layer": layer,
        "M": m,
        "routes": routes.tolist(),
        "token_count": m,
        "total_assignments": int(flat.size),
        "fanout_ranks": fanout,
        "active_experts_per_rank": active,
        "active_experts": int(np.count_nonzero(counts)),
        "rank_assignments": rank_counts.astype(int).tolist(),
        "expert_counts": counts.astype(int).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--m-values", default="128,256,512,1024")
    ap.add_argument("--fanout-values", default="1,2,4")
    ap.add_argument("--active-values", default="8,16,32")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    ms = [int(x) for x in args.m_values.split(",") if x]
    fanouts = [int(x) for x in args.fanout_values.split(",") if x]
    actives = [int(x) for x in args.active_values.split(",") if x]
    cases = [build_case(m, f, a, args.layer) for m in ms for f in fanouts for a in actives]
    random.Random(args.seed).shuffle(cases)
    (args.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    rows = []
    for index, c in enumerate(cases):
        rows.append({
            "order": index,
            "case_id": c["case_id"], "M": c["M"], "fanout_ranks": c["fanout_ranks"],
            "active_experts_per_rank": c["active_experts_per_rank"],
            "total_assignments": c["total_assignments"],
            "active_experts": c["active_experts"],
            "rank_assignments": json.dumps(c["rank_assignments"]),
            "route_sha256": hashlib.sha256(np.asarray(c["routes"], dtype=np.int64).tobytes()).hexdigest(),
        })
    with (args.output / "route_statistics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "H1_H2_execution_regime_factorial_grid",
        "M_values": ms, "fanout_values": fanouts, "active_experts_per_rank": actives,
        "top_k": 8, "global_experts": 128, "ep": 4, "layer": args.layer,
        "random_seed": args.seed, "case_order": "deterministically shuffled",
        "invariants": ["M within matched slice", "top_k", "total_assignments", "rank_assignments", "activation"],
        "changed_factors": ["destination EP ranks per token", "active local experts per rank"],
        "model_activation": "validated BF16 Qwen3-VL layer-24 capture rows",
        "placement": "expert_id // 32", "physical_gpus": [1, 2, 3, 4],
        "warning": "synthetic route-shape diagnostic; real-route transfer is analyzed separately",
    }, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
