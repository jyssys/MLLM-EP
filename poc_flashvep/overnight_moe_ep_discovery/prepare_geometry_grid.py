"""Prepare balanced sender-to-destination geometry controls for DeepEP replay."""
from __future__ import annotations

import argparse, hashlib, json, random, csv
from pathlib import Path
import numpy as np


def build_case(m: int, shape: str, active: int = 16, layer: int = 24) -> dict:
    if m % 4:
        raise ValueError("M must be divisible by four for balanced rank load")
    routes = np.empty((m, 8), dtype=np.int64)
    for token in range(m):
        if shape == "concentrated":
            # Keep aggregate rank load exactly balanced while concentrating
            # each token on one of two fixed rank pairs.
            ranks = (0, 1) if (token // 2) % 2 == 0 else (2, 3)
        elif shape == "cyclic":
            ranks = (token % 4, (token + 1) % 4)
        else:
            raise ValueError(shape)
        for j, rank in enumerate(ranks):
            start = (token * 4 + rank) % active
            for k in range(4):
                routes[token, j * 4 + k] = rank * 32 + ((start + k) % active)
    flat = routes.reshape(-1)
    counts = np.bincount(flat, minlength=128)
    rank_counts = counts.reshape(4, 32).sum(axis=1)
    return {
        "case_id": f"regime_M{m}_F2_A{active}_{shape}",
        "request_id": "geometry_grid", "category": "controlled_geometry",
        "modality": "synthetic_route_diagnostic", "layer": layer, "M": m,
        "routes": routes.tolist(), "token_count": m,
        "total_assignments": int(flat.size), "fanout_ranks": 2,
        "active_experts_per_rank": active, "active_experts": int(np.count_nonzero(counts)),
        "rank_assignments": rank_counts.astype(int).tolist(),
        "expert_counts": counts.astype(int).tolist(), "geometry": shape,
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--m-values", default="512"); ap.add_argument("--active", type=int, default=16)
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--seed", type=int, default=606)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for rep in range(a.reps):
        for m in map(int, a.m_values.split(",")):
            for shape in ("concentrated", "cyclic"):
                c = build_case(m, shape, a.active, a.layer); c["rep"] = rep
                c["base_case_id"] = c["case_id"]; c["case_id"] += f"_R{rep:02d}"; cases.append(c)
    random.Random(a.seed).shuffle(cases)
    (a.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    rows = []
    for i, c in enumerate(cases):
        rows.append({"order": i, "case_id": c["case_id"], "M": c["M"], "geometry": c["geometry"],
                     "total_assignments": c["total_assignments"], "active_experts": c["active_experts"],
                     "rank_assignments": json.dumps(c["rank_assignments"]),
                     "route_sha256": hashlib.sha256(np.asarray(c["routes"], dtype=np.int64).tobytes()).hexdigest()})
    with (a.output / "route_statistics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (a.output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "H6_sender_destination_geometry_balanced", "M_values": list(map(int, a.m_values.split(","))),
        "shapes": ["concentrated", "cyclic"], "repetitions": a.reps, "active": a.active,
        "top_k": 8, "ep": 4, "layer": a.layer, "seed": a.seed,
        "invariants": ["M", "total_assignments", "aggregate rank assignments", "activation"],
        "physical_gpus": [1, 2, 3, 4],
    }, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "cases": len(cases), "reps": a.reps}, indent=2))


if __name__ == "__main__": main()
