"""Summarize H4/H5 target-B history-conditioned replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(result: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((result / "replay").glob("rank*_layer24.json")):
        p = json.loads(path.read_text())
        if p.get("status") != "ok": continue
        for o in p["observations"]:
            if o.get("history_role") != "target_B" and not o["case_id"].startswith("B_"): continue
            def med(name): return float(o[name]["median_ms"])
            rows.append({"case_id": o["case_id"], "rank": p["rank"], "wall_ms": med("wall_stats"), "dispatch_ms": med("dispatch_stats"), "expert_ms": med("expert_stats"), "combine_ms": med("combine_stats"), "layout_ms": med("layout_stats"), "iterations": o["iterations"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True); args = ap.parse_args()
    raw = load(args.result)
    if raw.empty: raise RuntimeError("no target B observations")
    raw.to_csv(args.result / "target_B_rank_timings.csv", index=False)
    per_case = raw.groupby("case_id", as_index=False).agg(ranks=("rank", "nunique"), wall_ms=("wall_ms", "median"), critical_wall_ms=("wall_ms", "max"), dispatch_ms=("dispatch_ms", "max"), expert_ms=("expert_ms", "max"), combine_ms=("combine_ms", "max"), layout_ms=("layout_ms", "max"), iterations=("iterations", "max"))
    per_case.to_csv(args.result / "target_B_case_timings.csv", index=False)
    summary = {"condition": json.loads((args.result / "experiment_manifest.json").read_text())["condition"], "cases": len(per_case), "same_target_route": True, "target_median_ms": float(per_case.critical_wall_ms.median()), "target_p90_ms": float(per_case.critical_wall_ms.quantile(.9)), "target_cv_pct": float(per_case.critical_wall_ms.std(ddof=0) / per_case.critical_wall_ms.mean() * 100), "raw_rows": len(raw), "gate": "NO_GO"}
    (args.result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
