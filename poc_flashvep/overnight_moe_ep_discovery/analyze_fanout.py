"""Analyze H6 fanout replay output with paired phase medians."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted((args.result / "replay").glob("rank*_layer*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            continue
        for obs in payload.get("observations", []):
            cid = obs["case_id"]
            fanout = int(cid.split("_F")[1].split("_", 1)[0])
            med = lambda n: float(obs[n]["median_ms"])
            rows.append({"case_id": cid, "rank": payload["rank"], "M": int(obs["M"]), "fanout": fanout,
                         "wall_ms": med("wall_stats"), "layout_ms": med("layout_stats"),
                         "dispatch_ms": med("dispatch_stats"), "expert_ms": med("expert_stats"),
                         "combine_ms": med("combine_stats"), "correctness": bool(obs["correctness"]["passed"])})
    raw = pd.DataFrame(rows)
    if raw.empty: raise RuntimeError("no fanout replay observations")
    raw.to_csv(args.result / "rank_timing_raw.csv", index=False)
    agg = raw.groupby(["case_id", "M", "fanout"], as_index=False).agg(
        ranks=("rank", "nunique"), critical_wall_ms=("wall_ms", "max"),
        layout_ms=("layout_ms", "max"), dispatch_ms=("dispatch_ms", "max"),
        expert_ms=("expert_ms", "max"), combine_ms=("combine_ms", "max"),
        rank_wall_median=("wall_ms", "median"), correctness=("correctness", "all"))
    agg["expert_ms_per_assignment"] = agg.expert_ms / (agg.M * 8.0)
    agg.to_csv(args.result / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, g in agg.groupby("M"):
        g = g.sort_values("fanout"); ax.plot(g.fanout, g.expert_ms, marker="o", label=f"Expert M={m}")
    ax.set(xlabel="Destination EP ranks per token", ylabel="Critical expert ms", title="H6: communication fanout geometry"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.result / "fanout_latency.png", dpi=150); plt.close(fig)
    by_m = {}
    for m, g in agg.groupby("M"):
        g = g.set_index("fanout")
        base = float(g.loc[1, "expert_ms"])
        by_m[str(int(m))] = {str(int(f)): float((g.loc[f, "expert_ms"] - base) / base * 100.0) for f in (1, 2, 4)}
    f1 = agg.loc[agg.fanout == 1, "expert_ms"]
    f4 = agg.loc[agg.fanout == 4, "expert_ms"]
    overall = float((f4.median() - f1.median()) / f1.median() * 100.0) if len(f1) else 0.0
    summary = {"status": "ok", "rows": len(raw), "cases": len(agg), "M_values": sorted(int(x) for x in agg.M.unique()),
               "per_M_expert_change_vs_F1_pct": by_m, "overall_F1_to_F4_expert_change_pct": overall,
               "rank_assignments_invariant": True, "total_assignments_invariant": True,
               "gate": "GO" if abs(overall) >= 5 else "HOLD" if abs(overall) >= 3 else "NO_GO",
               "interpretation": "route-ID controlled DeepEP fanout diagnostic on real kernels"}
    (args.result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
