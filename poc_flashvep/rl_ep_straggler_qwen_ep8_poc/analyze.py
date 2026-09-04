#!/usr/bin/env python3
"""Offline analysis for the real Qwen3-30B-A3B EP8 capture.

The input is produced by the validated, read-only Qwen MoE hook.  It contains
one local-expert histogram and three CUDA-event spans per EP rank and layer.
No route or timing is synthesized here: aggregation only joins the eight
rank views of the same wave/layer.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(result: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((result / "raw_live").glob("rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            value = json.loads(line)
            value["ep_rank_file"] = int(path.stem.replace("rank", ""))
            rows.append(value)
    return rows


def _condition(request_id: str) -> str:
    match = re.match(r"qwen_ep8_(.*)_rep[0-9]+$", request_id)
    return match.group(1) if match else request_id


def _local_frame(rows: list[dict[str, Any]], measured_only: bool = True) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for value in rows:
        if measured_only and not bool(value.get("measured", False)):
            continue
        histogram = np.asarray(value.get("expert_histogram", []), dtype=float)
        if histogram.size == 0:
            continue
        dispatch = value.get("dispatch", {}) or {}
        expert = value.get("expert", {}) or {}
        combine = value.get("combine", {}) or {}
        request_id = str(value.get("request_id", "unknown"))
        active = histogram[histogram > 0]
        total = float(histogram.sum())
        mean = float(histogram.mean())
        out.append({
            "request_id": request_id,
            "condition": _condition(request_id),
            "rep": int(value.get("iteration", -1)) // 6,
            "wave": int(value.get("wave", -1)),
            "layer": int(value.get("layer", -1)),
            "worker_dp_rank": int(value.get("worker_dp_rank", -1)),
            "ep_rank": int(value.get("ep_rank", value.get("ep_rank_file", -1))),
            "total_assignments": total,
            "local_rows": int(value.get("dispatched_rows", 0)),
            "expert_ms": float(expert.get("ms", np.nan)),
            "dispatch_ms": float(dispatch.get("ms", np.nan)),
            "combine_ms": float(combine.get("ms", np.nan)),
            "active_experts_local": int((histogram > 0).sum()),
            "expert_max_assignments_local": float(histogram.max()),
            "expert_mean_assignments_local": mean,
            "expert_load_cv_local": float(histogram.std() / mean) if mean else 0.0,
            "tiny_expert_fraction_local": float(np.mean(active <= 4)) if active.size else 0.0,
            "expert_histogram": json.dumps([int(v) for v in histogram.tolist()]),
        })
    return pd.DataFrame(out)


def _aggregate(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame()
    # A rank can observe a profile/chunk call with fewer rows around the same
    # request.  Keep the largest local-row call per rank; this is the real
    # prefill view and leaves every timing row auditable in local_expert_trace.
    key = ["request_id", "wave", "layer", "ep_rank"]
    idx = local.groupby(key, dropna=False)["local_rows"].idxmax()
    local = local.loc[idx].copy()
    rows: list[dict[str, Any]] = []
    for key_value, group in local.groupby(["request_id", "wave", "layer"], dropna=False):
        loads = group["total_assignments"].to_numpy(float)
        local_rows = group["local_rows"].to_numpy(float)
        expert = group["expert_ms"].to_numpy(float)
        dispatch = group["dispatch_ms"].to_numpy(float)
        combine = group["combine_ms"].to_numpy(float)
        stage_sum = dispatch + expert + combine
        rank_mean = float(loads.mean())
        exp_mean = float(np.nanmean(expert))
        dis_mean = float(np.nanmean(dispatch))
        com_mean = float(np.nanmean(combine))
        active_global = []
        for _, item in group.iterrows():
            hist = np.asarray(json.loads(item["expert_histogram"]), dtype=float)
            active_global.extend(hist[hist > 0].tolist())
        active_arr = np.asarray(active_global, dtype=float)
        request_id, wave, layer = key_value
        row: dict[str, Any] = {
            "request_id": request_id,
            "condition": _condition(str(request_id)),
            "wave": int(wave),
            "layer": int(layer),
            "n_ep_ranks": int(len(group)),
            "total_assignments": float(loads.sum()),
            "effective_tokens_top8": float(loads.sum() / 8.0),
            "max_local_rows": int(local_rows.max()),
            "sum_local_rows": int(local_rows.sum()),
            "rank_max_assignments": float(loads.max()),
            "rank_mean_assignments": rank_mean,
            "rank_ratio": float(loads.max() / max(rank_mean, 1e-9)),
            "rank_cv": float(loads.std() / max(rank_mean, 1e-9)),
            "hottest_rank": int(group.iloc[int(np.argmax(loads))]["ep_rank"]),
            "expert_max_ms": float(np.nanmax(expert)),
            "expert_mean_ms": exp_mean,
            "expert_ratio": float(np.nanmax(expert) / max(exp_mean, 1e-9)),
            "dispatch_max_ms": float(np.nanmax(dispatch)),
            "dispatch_mean_ms": dis_mean,
            "dispatch_ratio": float(np.nanmax(dispatch) / max(dis_mean, 1e-9)),
            "combine_max_ms": float(np.nanmax(combine)),
            "combine_mean_ms": com_mean,
            "combine_ratio": float(np.nanmax(combine) / max(com_mean, 1e-9)),
            "critical_path_max_ms": float(np.nanmax(stage_sum)),
            "critical_path_mean_ms": float(np.nanmean(stage_sum)),
            "critical_path_ratio": float(np.nanmax(stage_sum) / max(np.nanmean(stage_sum), 1e-9)),
            "critical_rank": int(group.iloc[int(np.nanargmax(stage_sum))]["ep_rank"]),
            "active_experts_global": int((active_arr > 0).sum()),
            "expert_assignment_max": float(active_arr.max()) if active_arr.size else 0.0,
            "expert_assignment_mean_active": float(active_arr.mean()) if active_arr.size else 0.0,
            "expert_assignment_cv_global": float(active_arr.std() / max(active_arr.mean(), 1e-9)) if active_arr.size else 0.0,
            "expert_assignment_hhi_global": float(np.square(active_arr / max(active_arr.sum(), 1e-9)).sum()) if active_arr.size else 0.0,
            "expert_histograms": json.dumps({str(int(r.ep_rank)): json.loads(r.expert_histogram)
                                               for _, r in group.iterrows()})
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _route_temporal(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Repeatability of hot experts from all waves, including warmup routes."""
    local = _local_frame(rows, measured_only=False)
    if local.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (condition, layer), group in local.groupby(["condition", "layer"], dropna=False):
        by_wave = []
        for wave, wave_group in group.groupby("wave"):
            counts: dict[int, int] = {}
            for _, item in wave_group.iterrows():
                hist = json.loads(item.expert_histogram)
                for local_id, count in enumerate(hist):
                    counts[int(item.ep_rank) * 16 + local_id] = int(count)
            if counts:
                by_wave.append((int(wave), max(counts, key=counts.get)))
        by_wave.sort()
        hot = [x[1] for x in by_wave]
        records.append({
            "condition": condition, "layer": int(layer), "n_waves": len(hot),
            "hot_expert_recurrence": float(np.mean(np.asarray(hot[1:]) == np.asarray(hot[:-1]))) if len(hot) > 1 else 0.0,
            "hot_expert_unique": int(len(set(hot))),
        })
    return pd.DataFrame(records)


def _plots(agg: pd.DataFrame, temporal: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if agg.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(agg.rank_ratio, agg.expert_ratio, c=agg.layer, s=11, alpha=.55, cmap="viridis")
    axes[0].axvline(1.5, color="k", ls="--", lw=.8)
    axes[0].axhline(1.25, color="r", ls="--", lw=.8)
    axes[0].set(xlabel="max/mean rank assignments", ylabel="max/mean expert CUDA")
    axes[1].scatter(agg.total_assignments, agg.expert_ratio, c=agg.layer, s=11, alpha=.55, cmap="viridis")
    axes[1].set(xlabel="total routed assignments", ylabel="max/mean expert CUDA")
    fig.tight_layout(); fig.savefig(out / "routing_vs_expert_imbalance.png", dpi=160); plt.close(fig)
    layer = agg.groupby("layer")[["rank_ratio", "expert_ratio", "dispatch_ratio", "combine_ratio", "critical_path_ratio"]].median()
    fig, ax = plt.subplots(figsize=(9, 4)); layer.plot(ax=ax)
    ax.set(xlabel="decoder layer", ylabel="median max/mean ratio"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "per_layer_straggler.png", dpi=160); plt.close(fig)
    condition = agg.groupby("condition")[["rank_ratio", "expert_ratio", "critical_path_max_ms"]].median()
    fig, ax = plt.subplots(figsize=(9, 4)); condition.plot.bar(ax=ax)
    ax.set_ylabel("median (ratio or ms)"); fig.tight_layout(); fig.savefig(out / "condition_summary.png", dpi=160); plt.close(fig)
    if not temporal.empty:
        x = temporal.groupby("condition").hot_expert_recurrence.median().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4)); x.plot.barh(ax=ax, color="#7f9eaf")
        ax.set_xlabel("adjacent-wave hottest-expert recurrence")
        fig.tight_layout(); fig.savefig(out / "hot_expert_persistence.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args(); result = args.result
    rows = _load(result)
    local = _local_frame(rows, measured_only=True)
    agg = _aggregate(local)
    temporal = _route_temporal(rows)
    local.to_csv(result / "local_expert_trace.csv", index=False)
    agg.to_csv(result / "invocation_metrics.csv", index=False)
    temporal.to_csv(result / "temporal_metrics.csv", index=False)
    _plots(agg, temporal, result / "figures")
    if agg.empty:
        gate = {"stage0": "NO_GO", "reason": "no measured routed expert rows captured", "n_invocations": 0}
    else:
        # A measured prefill is the largest routed call (>=16 rows) and all
        # captured Qwen stages are sparse MoE layers.  The user gate is fixed
        # before looking at this result.
        prefill = agg[agg.max_local_rows >= 16]
        ratios = prefill.expert_ratio.dropna().to_numpy(float)
        if len(ratios) == 0:
            ratios = agg.expert_ratio.dropna().to_numpy(float)
        median = float(np.median(ratios)); p90 = float(np.quantile(ratios, .90))
        frac115 = float(np.mean(ratios >= 1.15)); frac125 = float(np.mean(ratios >= 1.25))
        frac150 = float(np.mean(ratios >= 1.50))
        # GO is the preregistered stable 1.15 gate; STRONG_GO is retained for
        # the requested EP8 testbed claim and requires repeated >=1.25 plus
        # repeated heavy >=1.50 evidence.
        stage0 = "STRONG_GO" if median >= 1.25 and frac125 >= .50 and int((ratios >= 1.50).sum()) >= 2 else ("GO" if median >= 1.15 and frac115 >= .50 else "NO_GO")
        gate = {
            "stage0": stage0, "n_invocations": int(len(ratios)),
            "prefill_invocations": int(len(prefill)),
            "expert_cuda_max_mean_median": median,
            "expert_cuda_max_mean_p75": float(np.quantile(ratios, .75)),
            "expert_cuda_max_mean_p90": p90,
            "expert_cuda_max_mean_max": float(np.max(ratios)),
            "fraction_ge_1_15": frac115, "fraction_ge_1_25": frac125,
            "fraction_ge_1_50": frac150,
            "count_ge_1_15": int((ratios >= 1.15).sum()),
            "count_ge_1_25": int((ratios >= 1.25).sum()),
            "count_ge_1_50": int((ratios >= 1.50).sum()),
            "stage0_gate_rule": "GO iff median >=1.15 and >=50% >=1.15; STRONG_GO iff median >=1.25, >=50% >=1.25, and at least two >=1.50 invocations",
            "timing_note": "expert/dispatch/combine CUDA events are resolved once at final flush; per-rank values are compared as durations, never cross-device absolute timestamps",
        }
    (result / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
