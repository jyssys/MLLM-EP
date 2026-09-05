"""Analyze H7 uniform/skew distribution-shape replay."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True); args = ap.parse_args()
    rows = []
    for path in sorted((args.result / "replay").glob("rank*_layer*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok": continue
        for obs in payload.get("observations", []):
            m = re.search(r"distribution_M(\d+)_F4_A16_(uniform|skew)", obs["case_id"])
            if not m: continue
            rows.append({"M": int(m.group(1)), "shape": m.group(2), "rank": int(payload["rank"]),
                         "expert_ms": float(obs["expert_stats"]["median_ms"]),
                         "dispatch_ms": float(obs["dispatch_stats"]["median_ms"]),
                         "combine_ms": float(obs["combine_stats"]["median_ms"]),
                         "wall_ms": float(obs["wall_stats"]["median_ms"]),
                         "iterations": int(obs.get("iterations", 0)), "warmups": int(obs.get("warmups", 0))})
    raw = pd.DataFrame(rows); raw.to_csv(args.result / "distribution_rank_timing_raw.csv", index=False)
    rank = raw.groupby(["M", "shape", "rank"], as_index=False).median(numeric_only=True)
    rank.to_csv(args.result / "distribution_rank_medians.csv", index=False)
    agg = rank.groupby(["M", "shape"], as_index=False).agg(
        critical_expert_ms=("expert_ms", "max"), critical_dispatch_ms=("dispatch_ms", "max"),
        critical_combine_ms=("combine_ms", "max"), critical_wall_ms=("wall_ms", "max"),
        rank_expert_mean=("expert_ms", "mean"), rank_wall_mean=("wall_ms", "mean"))
    agg.to_csv(args.result / "distribution_metrics.csv", index=False)
    pivot = agg.pivot(index="M", columns="shape", values="critical_expert_ms")
    wall = agg.pivot(index="M", columns="shape", values="critical_wall_ms")
    summary = {"status": "ok", "raw_rows": int(len(raw)), "repetitions": int(raw.groupby(["M", "shape"]).size().min() if not raw.empty else 0),
               "expert_skew_vs_uniform_pct": {str(int(m)): float((pivot.loc[m, "skew"] / pivot.loc[m, "uniform"] - 1) * 100) for m in pivot.index},
               "wall_skew_vs_uniform_pct": {str(int(m)): float((wall.loc[m, "skew"] / wall.loc[m, "uniform"] - 1) * 100) for m in wall.index},
               "invariants": "same M, top-k, fanout=4, active=16, total/rank assignments",
               "interpretation": "H7 distribution-shape diagnostic on real DeepEP/TritonExperts"}
    (args.result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
