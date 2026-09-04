#!/usr/bin/env python3
"""Analyze measured DeepSeek-V2-Lite EP4 route/expert traces.

The first gate is deliberately preregistered: a Stage-0 natural run is a
failure when most layer invocations have expert max/mean <= 1.10.  Stage 1/2
oracle summaries are emitted only when that gate passes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(path.glob("rank*.jsonl")):
        ep = int(p.stem.replace("rank", ""))
        for line in p.read_text().splitlines():
            if line:
                x = json.loads(line); x["ep_rank_file"] = ep; rows.append(x)
    return rows


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = []
    for x in rows:
        if not bool(x.get("measured", False)):
            continue
        hist = np.asarray(x.get("expert_histogram", []), dtype=float)
        if hist.size == 0:
            continue
        # The DeepSeek EP metadata is local physical-expert assignment count.
        # Summing it is therefore the exact rank workload for this invocation.
        active = hist[hist > 0]
        total = float(hist.sum())
        mean = float(hist.mean()) if hist.size else 0.0
        out.append({
            "batch_id": x.get("batch_id"), "condition": x.get("condition"),
            "domain": x.get("domain"), "step": x.get("step", -1),
            "scheduler_iteration": x.get("scheduler_iteration", -1),
            "worker_dp_rank": int(x.get("worker_dp_rank", -1)),
            "ep_rank": int(x.get("ep_rank", x.get("ep_rank_file", -1))),
            "layer": int(x.get("layer", -1)),
            "total_assignments": total, "local_rows": int(x.get("dispatched_rows", 0)),
            "expert_ms": float(x.get("expert_ms", np.nan)),
            "dispatch_ms": float(x.get("dispatch_ms", np.nan)),
            "combine_ms": float(x.get("combine_ms", np.nan)),
            "active_experts_local": int((hist > 0).sum()),
            "expert_max_assignments_local": float(hist.max()),
            "expert_mean_assignments_local": mean,
            "expert_load_cv_local": float(hist.std() / mean) if mean else 0.0,
            "tiny_expert_fraction_local": float(np.mean(active <= 4)) if active.size else 0.0,
            "expert_histogram": json.dumps([int(v) for v in hist.tolist()]),
        })
    return pd.DataFrame(out)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    # Each request tag contains one (possibly chunked) prefill and subsequent
    # one-token decode calls.  Driver-local forward counters are not globally
    # synchronized, so select the largest-row call independently for each EP
    # rank.  This yields the four rank views of the same natural prefill.
    dedup_keys = ["batch_id", "condition", "step", "layer", "ep_rank"]
    idx = df.groupby(dedup_keys, dropna=False).local_rows.idxmax()
    df = df.loc[idx].copy()
    keys = ["batch_id", "condition", "step", "layer"]
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        loads = g.total_assignments.to_numpy(float)
        ex = g.expert_ms.to_numpy(float)
        dis = g.dispatch_ms.to_numpy(float)
        com = g.combine_ms.to_numpy(float)
        rank_mean = float(loads.mean())
        ex_mean = float(np.nanmean(ex)); dis_mean = float(np.nanmean(dis)); com_mean = float(np.nanmean(com))
        rows.append(dict(zip(keys, key)))
        rows[-1]["domain"] = "|".join(sorted(set(str(v) for v in g.domain)))
        rows[-1].update({
            "n_ranks": len(g), "max_local_rows": int(g.local_rows.max()),
            "total_assignments": float(loads.sum()),
            "rank_max_assignments": float(loads.max()), "rank_mean_assignments": rank_mean,
            "rank_ratio": float(loads.max() / max(rank_mean, 1e-9)),
            "rank_cv": float(loads.std() / max(rank_mean, 1e-9)),
            "hottest_rank": int(g.iloc[int(np.argmax(loads))].ep_rank),
            "expert_max_ms": float(np.nanmax(ex)), "expert_mean_ms": ex_mean,
            "expert_ratio": float(np.nanmax(ex) / max(ex_mean, 1e-9)),
            "dispatch_max_ms": float(np.nanmax(dis)), "dispatch_mean_ms": dis_mean,
            "dispatch_ratio": float(np.nanmax(dis) / max(dis_mean, 1e-9)),
            "combine_max_ms": float(np.nanmax(com)), "combine_mean_ms": com_mean,
            "combine_ratio": float(np.nanmax(com) / max(com_mean, 1e-9)),
            "critical_moe_ms": float(np.nanmax(dis) + np.nanmax(ex) + np.nanmax(com)),
        })
    return pd.DataFrame(rows)


def _temporal(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty: return pd.DataFrame()
    rows = []
    # The schedule repeats the same domain-pair condition across episode
    # repetitions.  Keeping the condition label makes the temporal diagnostic
    # a real repeated-workload check rather than a one-point-per-domain table.
    for condition, g in agg.sort_values(["step", "layer"]).groupby("condition", dropna=False):
        hot = g[g.layer == g.layer.min()].sort_values("step")
        vals = hot.expert_ratio.to_numpy(float)
        ranks = hot.hottest_rank.to_numpy()
        rows.append({"condition": condition, "domain": "|".join(sorted(set(str(v) for v in g.domain))),
                     "steps": len(vals),
                     "expert_ratio_autocorr": float(np.corrcoef(vals[:-1], vals[1:])[0,1]) if len(vals)>2 and np.std(vals)>0 else 0.0,
                     "hottest_rank_recurrence": float(np.mean(ranks[1:] == ranks[:-1])) if len(ranks)>1 else 0.0,
                     "mean_expert_ratio": float(np.nanmean(vals)) if len(vals) else np.nan})
    return pd.DataFrame(rows)


def _plot(agg: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if agg.empty: return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(agg.rank_ratio, agg.expert_ratio, c=agg.layer, s=10, alpha=.55, cmap="viridis")
    ax.axvline(1.5, color="k", ls="--", lw=.8); ax.axhline(1.15, color="r", ls="--", lw=.8)
    ax.set(xlabel="max/mean rank assignments", ylabel="max/mean expert CUDA time")
    fig.tight_layout(); fig.savefig(out / "routing_vs_expert_imbalance.png", dpi=160); plt.close(fig)
    z = agg.groupby("layer")[["rank_ratio", "expert_ratio", "critical_moe_ms"]].median()
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(z.index, z.rank_ratio, label="rank load")
    ax.plot(z.index, z.expert_ratio, label="expert time"); ax.set(xlabel="decoder layer", ylabel="ratio")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "per_layer_straggler.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4));
    for domain, g in agg.groupby("domain"):
        q = g.groupby("step").expert_ratio.median(); ax.plot(q.index, q.values, marker="o", label=domain)
    ax.set(xlabel="episode step", ylabel="median expert max/mean"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "temporal_expert_imbalance.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args(); result = args.result
    raw = _load(result / "raw_live"); local = _frame(raw); agg = _aggregate(local)
    local.to_csv(result / "local_expert_trace.csv", index=False)
    agg.to_csv(result / "invocation_metrics.csv", index=False)
    temporal = _temporal(agg); temporal.to_csv(result / "temporal_metrics.csv", index=False)
    _plot(agg, result / "figures")
    if agg.empty:
        gate = {"stage0": "NO_GO", "reason": "no routed expert rows captured", "n_invocations": 0}
    else:
        # Stage 0 is a prefill straggler gate.  Decode calls have only a few
        # rows and are retained in the raw/CSV diagnostics but excluded from
        # the primary natural-prefill decision.
        prefill = agg[agg.max_local_rows >= 16]
        ratios = prefill.expert_ratio.dropna().to_numpy(float)
        if len(ratios) == 0:
            ratios = agg.expert_ratio.dropna().to_numpy(float)
        # Preregistered user gate: natural routed-expert CUDA imbalance must
        # be meaningfully above noise.  A median >=1.15 and >=50% of measured
        # prefill invocations at/above 1.15 is the primary PASS criterion;
        # >=1.50 is retained as the heavy-case diagnostic.
        median_ratio = float(np.median(ratios))
        frac_115 = float(np.mean(ratios >= 1.15))
        stage0_pass = median_ratio >= 1.15 and frac_115 >= 0.50
        gate = {
            "stage0": "GO" if stage0_pass else "NO_GO",
            "n_invocations": int(len(ratios)), "prefill_invocations": int(len(prefill)), "expert_ratio_median": float(np.median(ratios)),
            "expert_ratio_p90": float(np.quantile(ratios, .90)), "expert_ratio_max": float(np.max(ratios)),
            "fraction_expert_ratio_gt_1_10": float(np.mean(ratios > 1.10)),
            "fraction_expert_ratio_ge_1_15": float(np.mean(ratios >= 1.15)),
            "fraction_expert_ratio_ge_1_25": float(np.mean(ratios >= 1.25)),
            "fraction_expert_ratio_ge_1_50": float(np.mean(ratios >= 1.50)),
            "stage0_gate_rule": "GO iff median expert CUDA max/mean >=1.15 and >=50% prefill invocations >=1.15; heavy >=1.50 reported",
        }
    (result / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__": main()
