"""Tail and regime analysis for the online runtime discovery traces.

The source hook records a rank-local full FusedMoE CUDA interval.  This
script only compares rows with the same source/phase/layer/M bucket and uses
the maximum of the four rank intervals as a conservative critical span.
It does not infer a dispatch/expert/combine breakdown that was not recorded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def load(path):
    with Path(path).open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def quantiles(x):
    a = np.asarray(x, dtype=float)
    return {"p50": float(np.quantile(a, .50)), "p90": float(np.quantile(a, .90)),
            "p95": float(np.quantile(a, .95)), "p99": float(np.quantile(a, .99)),
            "mean": float(np.mean(a)), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load(args.aggregate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Aggregate rows are already complete four-rank invocations.  Keep an
    # explicit high-tail table so the report can inspect matched regimes.
    groups = {}
    for r in rows:
        key = (r.get("source", ""), r.get("phase", ""), int(f(r, "layer")), int(f(r, "M")))
        groups.setdefault(key, []).append(r)
    tail_rows = []
    for (source, phase, layer, m), z in sorted(groups.items()):
        if len(z) < 20:
            continue
        zsort = sorted(z, key=lambda r: f(r, "critical_cuda_ms"))
        vals = np.asarray([f(r, "critical_cuda_ms") for r in zsort])
        q95 = float(np.quantile(vals, .95)); q50 = float(np.quantile(vals, .50))
        slow = [r for r in zsort if f(r, "critical_cuda_ms") >= q95]
        fast = [r for r in zsort if f(r, "critical_cuda_ms") <= q50]
        def mean(key, x):
            return float(np.mean([f(r, key) for r in x])) if x else 0.0
        def median(key, x):
            return float(np.median([f(r, key) for r in x])) if x else 0.0
        fast_med=median("critical_cuda_ms", fast); slow_med=median("critical_cuda_ms", slow)
        tail_rows.append({
            "source": source, "phase": phase, "layer": layer, "M": m,
            "n": len(z), "p50_ms": q50, "p95_ms": q95,
            # Median tail gap is the robust primary; retain the mean-based
            # value separately because a single scheduler outlier can be
            # orders of magnitude larger than the fixed-M regime.
            "tail_gap_pct": 100.0 * (slow_med-fast_med) / (fast_med + 1e-12),
            "tail_gap_mean_pct": 100.0 * (mean("critical_cuda_ms", slow) - mean("critical_cuda_ms", fast)) / (mean("critical_cuda_ms", fast) + 1e-12),
            "slow_median_ms": slow_med, "fast_median_ms": fast_med,
            "slow_rank_imbalance": mean("rank_imbalance", slow),
            "fast_rank_imbalance": mean("rank_imbalance", fast),
            "slow_fanout": mean("fanout_mean", slow), "fast_fanout": mean("fanout_mean", fast),
            "slow_active_experts": mean("active_experts", slow), "fast_active_experts": mean("active_experts", fast),
            "slow_total_assignments": mean("total_assignments", slow), "fast_total_assignments": mean("total_assignments", fast),
        })
    fields = sorted(tail_rows[0]) if tail_rows else ["source", "phase", "layer", "M"]
    with (out / "tail_analysis.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(tail_rows)

    summary = {"rows": len(rows), "tail_groups": len(tail_rows), "by_phase": {},
               "by_source_phase": {}, "top_tail_gaps": sorted(tail_rows, key=lambda x: x["tail_gap_pct"], reverse=True)[:20]}
    for phase in sorted({r.get("phase", "") for r in rows}):
        z = [f(r, "critical_cuda_ms") for r in rows if r.get("phase") == phase]
        summary["by_phase"][phase] = quantiles(z)
    for source, phase in sorted({(r.get("source", ""), r.get("phase", "")) for r in rows}):
        z = [f(r, "critical_cuda_ms") for r in rows if r.get("source") == source and r.get("phase") == phase]
        summary["by_source_phase"][f"{source}:{phase}"] = quantiles(z)
    # Simple rank-load association (descriptive only; timing is noisy).
    for phase in summary["by_phase"]:
        z = [r for r in rows if r.get("phase") == phase]
        if len(z) > 2:
            y = np.asarray([f(r, "critical_cuda_ms") for r in z]); x = np.asarray([f(r, "rank_imbalance") for r in z])
            summary["by_phase"][phase]["corr_critical_rank_imbalance"] = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
    (out / "tail_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "tail_groups": len(tail_rows), "top": summary["top_tail_gaps"][:3]}, indent=2))


if __name__ == "__main__":
    main()
