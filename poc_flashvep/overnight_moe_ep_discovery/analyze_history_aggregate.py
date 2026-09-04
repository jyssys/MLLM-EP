"""Aggregate fixed-route H4/H5 history-conditioned replay results.

The replay intentionally keeps the target B route identical; only the immediately
preceding prime route changes.  This is a diagnostic replay, not a serving claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for cond in ("steady", "alternating", "similar", "disjoint"):
        p = args.root / f"H04_H05_{cond}" / "target_B_case_timings.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        # Drop the explicit prime warm-up case.  All remaining rows are the
        # identical target route and are the paired unit for this diagnostic.
        d = d[~d.case_id.str.contains("prime", case=False, na=False)].copy()
        d["condition"] = cond
        rows.append(d)
    if not rows:
        raise RuntimeError("no history summaries found")
    all_cases = pd.concat(rows, ignore_index=True)
    all_cases.to_csv(args.root / "history_target_case_timings.csv", index=False)
    med = all_cases.groupby("condition", as_index=False).agg(
        cases=("case_id", "count"), critical_wall_median_ms=("critical_wall_ms", "median"),
        critical_wall_p90_ms=("critical_wall_ms", lambda x: float(x.quantile(.9))),
        critical_wall_cv_pct=("critical_wall_ms", lambda x: float(x.std(ddof=0) / x.mean() * 100)),
        dispatch_median_ms=("dispatch_ms", "median"), expert_median_ms=("expert_ms", "median"),
        combine_median_ms=("combine_ms", "median"),
    )
    med.to_csv(args.root / "history_condition_summary.csv", index=False)
    plt.figure(figsize=(6, 4))
    order = ["steady", "alternating", "similar", "disjoint"]
    plot = all_cases.set_index("condition").loc[[c for c in order if c in all_cases.condition.unique()]].reset_index()
    present = [c for c in order if c in plot.condition.unique()]
    plt.boxplot([plot.loc[plot.condition == c, "critical_wall_ms"] for c in present], tick_labels=present)
    plt.ylabel("target-B critical wall (ms)"); plt.title("History-conditioned identical-route target")
    plt.tight_layout(); plt.savefig(args.root / "history_condition_latency.png", dpi=140); plt.close()
    lookup = med.set_index("condition")["critical_wall_median_ms"]
    steady = float(lookup.get("steady", float("nan")))
    def pct(cond: str) -> float:
        v = float(lookup.get(cond, float("nan")))
        return (v - steady) / steady * 100 if steady == steady and v == v else float("nan")
    summary = {
        "target_route": "vision.deep_field.npz layer24 first256",
        "same_target_route": True,
        "prime_case_excluded": True,
        "condition_medians_ms": {k: float(v) for k, v in lookup.to_dict().items()},
        "alternating_vs_steady_pct": pct("alternating"),
        "similar_vs_steady_pct": pct("similar"),
        "disjoint_vs_steady_pct": pct("disjoint"),
        "h4_gate": "NO_GO" if abs(pct("alternating")) < 5 else "HOLD",
        "h5_gate": "NO_GO" if abs(pct("similar")) < 5 and abs(pct("disjoint")) < 5 else "HOLD",
        "interpretation": "large case-to-case variance and one recurring low-latency tail prevent a robust history claim",
    }
    (args.root / "history_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
