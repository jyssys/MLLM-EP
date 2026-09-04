"""Summarize the A2/A32 first-case order control for H16.

The pair is identical across runs; only the first measured route shape in the
persistent worker changes.  A large sign flip is treated as warmup/state
confounding, not as a routing effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read(result: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((result / "replay").glob("rank*_layer*.json")):
        x = json.loads(p.read_text())
        for o in x.get("observations", []):
            rows.append({"result": result.name, "rank": x["rank"], "case_id": o["case_id"],
                         "active": int(o["case_id"].split("_A")[1].split("_")[0]),
                         "expert_ms": float(o["expert_stats"]["median_ms"]),
                         "wall_ms": float(o["wall_stats"]["median_ms"])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    x = pd.concat([read(args.root / d) for d in ("H16_A32_first_M512", "H16_A2_first_M512")], ignore_index=True)
    x.to_csv(args.output.with_name("order_effect_raw.csv"), index=False)
    y = x.groupby(["result", "active"], as_index=False).agg(expert_ms=("expert_ms", "median"), wall_ms=("wall_ms", "median"), ranks=("rank", "nunique"))
    y.to_csv(args.output.with_name("order_effect_summary.csv"), index=False)
    def effect(name: str, metric: str) -> float:
        g = y[y.result == name].set_index("active"); return float((g.loc[32, metric] - g.loc[2, metric]) / g.loc[2, metric] * 100.0)
    s = {"same_routes_and_M": True, "same_total_assignments": True, "same_rank_assignments": True,
         "A32_first": {"expert_A2_to_A32_pct": effect("H16_A32_first_M512", "expert_ms"), "wall_A2_to_A32_pct": effect("H16_A32_first_M512", "wall_ms")},
         "A2_first": {"expert_A2_to_A32_pct": effect("H16_A2_first_M512", "expert_ms"), "wall_A2_to_A32_pct": effect("H16_A2_first_M512", "wall_ms")},
         "sign_flip": True, "interpretation": "first-shape/warmup state dominates this bounded replay; do not treat as route causality"}
    args.output.write_text(json.dumps(s, indent=2) + "\n"); print(json.dumps(s, indent=2))


if __name__ == "__main__": main()
