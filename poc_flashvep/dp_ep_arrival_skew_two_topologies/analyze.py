#!/usr/bin/env python3
"""Aggregate DP-local arrival skew and EP wait proxies from live traces."""

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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _load_scheduler(result: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # One TP worker per DP rank is sufficient; two TP workers duplicate the
    # scheduler observation in TP2/DP2.
    seen_dp: set[int] = set()
    for path in sorted((result / "scheduler_trace").glob("*.jsonl")):
        if "dp" not in path.name:
            continue
        try:
            dp = int(path.name.split("scheduler_dp", 1)[1].split("_", 1)[0])
        except (ValueError, IndexError):
            continue
        if dp in seen_dp:
            continue
        seen_dp.add(dp)
        rows.extend(_jsonl(path))
    return pd.DataFrame(rows)


def _load_rank_rows(result: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((result / "raw_live").glob("rank[0-3].jsonl")):
        ep = int(path.stem[4:])
        # The eager vLLM MoE backend performs small autotuner probes on the
        # first encounter of a shape.  They are tagged with the same control
        # record but have only a handful of assignments (typically 16--20),
        # unlike real prefill chunks.  Keep the preregistered real-work range
        # and make the exclusion explicit rather than calling probes serving
        # iterations.
        out[ep] = [r for r in _jsonl(path)
                   if bool(r.get("measured"))
                   and int(r.get("total_assignments", 0)) >= 64]
    return out


def _load_arrival(result: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((result / "raw_live").glob("arrival_*.jsonl")):
        rows.extend(r for r in _jsonl(path) if bool(r.get("measured")))
    return rows


def _batch_scheduler(sched: pd.DataFrame, batch: str) -> dict[str, Any]:
    if sched.empty:
        return {"scheduled_tokens_sum": 0, "scheduled_tokens_max": 0, "scheduled_tokens_median": 0, "scheduled_requests_max": 0, "scheduler_rows": 0}
    g = sched[sched.batch_id.astype(str) == str(batch)]
    if g.empty:
        return {"scheduled_tokens_sum": 0, "scheduled_tokens_max": 0, "scheduled_tokens_median": 0, "scheduled_requests_max": 0, "scheduler_rows": 0}
    vals = g.total_num_scheduled_tokens.astype(float)
    positive = vals[vals > 0]
    return {
        "scheduled_tokens_sum": float(positive.sum()), "scheduled_tokens_max": float(positive.max()) if len(positive) else 0.0,
        "scheduled_tokens_median": float(positive.median()) if len(positive) else 0.0,
        "scheduled_requests_max": int(max((len(x) for x in g.num_scheduled_tokens), default=0)),
        "scheduler_rows": int(len(g)),
    }


def collect(result: Path) -> pd.DataFrame:
    meta = json.loads((result / "run_metadata.json").read_text())
    sched = _load_scheduler(result)
    rows_by_ep = _load_rank_rows(result)
    arrivals = _load_arrival(result)
    arr_index: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in arrivals:
        arr_index[(str(row.get("batch_id")), int(row.get("scheduler_iteration", -1)), int(row.get("layer", -1)), int(row.get("worker_dp_rank", -1)))] = row
    # Group raw expert rows by batch, local model-forward index, and layer.
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for ep, rows in rows_by_ep.items():
        for row in rows:
            grouped[(str(row.get("batch_id")), int(row.get("scheduler_iteration", -1)), int(row.get("layer", -1)))].append(row)
    records: list[dict[str, Any]] = []
    for (batch, iteration, layer), rr in sorted(grouped.items()):
        by_ep = {int(x.get("ep_rank", -1)): x for x in rr}
        if len(by_ep) != 4:
            continue
        dp_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in by_ep.values():
            dp_rows[int(row.get("worker_dp_rank", -1))].append(row)
        if len(dp_rows) not in (2, 4):
            continue
        pre = {dp: max(float(arr_index.get((batch, iteration, layer, dp), {}).get("pre_moe_cuda_ms", np.nan)) for _ in rws) for dp, rws in dp_rows.items()}
        ep_done = {dp: max(float(arr_index.get((batch, iteration, layer, dp), {}).get("layer_entry_to_ep_done_ms", np.nan)) for _ in rws) for dp, rws in dp_rows.items()}
        pre_vals = np.asarray([v for v in pre.values() if np.isfinite(v)], dtype=float)
        done_vals = np.asarray([v for v in ep_done.values() if np.isfinite(v)], dtype=float)
        if len(pre_vals) != len(dp_rows):
            continue
        loads = np.asarray([float(x.get("total_assignments", 0)) for x in by_ep.values()])
        expert = np.asarray([float(x.get("expert", {}).get("ms", np.nan)) for x in by_ep.values()])
        dispatch = np.asarray([float(x.get("dispatch", {}).get("ms", np.nan)) for x in by_ep.values()])
        combine = np.asarray([float(x.get("combine", {}).get("ms", np.nan)) for x in by_ep.values()])
        hist = np.concatenate([np.asarray(x.get("expert_histogram", []), dtype=float) for x in by_ep.values()])
        sb = _batch_scheduler(sched, batch)
        control = next(iter(by_ep.values()))
        # Older read-only hook records only the condition in each expert row.
        # Recover the preregistered workload labels from the stable batch-id
        # prefix/suffix without using any latency-derived selection.
        kind = control.get("workload_kind")
        if not kind:
            kind = "long" if batch.startswith("long_") else ("text" if batch.startswith("text_") else "vision")
        mode = control.get("mode")
        if not mode:
            mode = "heterogeneous" if "_heterogeneous_" in batch else "balanced"
        row = {
            "result": str(result), "topology": str(meta.get("topology")), "tp": int(meta.get("tp", -1)), "dp": int(meta.get("dp", -1)),
            "batch_id": batch, "scheduler_iteration": iteration, "layer": layer,
            "condition": control.get("condition"), "modality": control.get("modality"), "workload_kind": kind,
            "mode": mode, "max_num_batched_tokens": int(meta.get("max_num_batched_tokens", -1)),
            "concurrency": int(control.get("concurrency", 1)), "control_iteration": int(control.get("iteration", -1)),
            **sb, "pre_moe_max_ms": float(np.max(pre_vals)), "pre_moe_min_ms": float(np.min(pre_vals)),
            "arrival_skew_ms": float(np.max(pre_vals) - np.min(pre_vals)), "arrival_skew_ratio": float((np.max(pre_vals) - np.min(pre_vals)) / max(np.max(pre_vals), 1e-9)),
            "pre_moe_mean_ms": float(np.mean(pre_vals)), "pre_moe_std_ms": float(np.std(pre_vals)),
            "ep_done_max_ms": float(np.max(done_vals)) if len(done_vals) == len(dp_rows) else np.nan,
            "ep_done_min_ms": float(np.min(done_vals)) if len(done_vals) == len(dp_rows) else np.nan,
            "ep_wait_proxy_ms": float(np.max(done_vals) - np.min(done_vals)) if len(done_vals) == len(dp_rows) else np.nan,
            "sync_stall_fraction": float((np.max(done_vals) - np.min(done_vals)) / max(np.max(done_vals), 1e-9)) if len(done_vals) == len(dp_rows) else np.nan,
            "rank_load_max": float(loads.max()), "rank_load_mean": float(loads.mean()), "rank_load_ratio": float(loads.max() / max(loads.mean(), 1e-9)),
            "rank_load_cv": float(loads.std() / max(loads.mean(), 1e-9)), "hottest_rank": int(np.argmax(loads)),
            "expert_max_ms": float(np.nanmax(expert)), "expert_mean_ms": float(np.nanmean(expert)), "expert_ratio": float(np.nanmax(expert) / max(np.nanmean(expert), 1e-9)),
            "dispatch_max_ms": float(np.nanmax(dispatch)), "dispatch_mean_ms": float(np.nanmean(dispatch)), "dispatch_ratio": float(np.nanmax(dispatch) / max(np.nanmean(dispatch), 1e-9)),
            "combine_max_ms": float(np.nanmax(combine)), "combine_mean_ms": float(np.nanmean(combine)), "combine_ratio": float(np.nanmax(combine) / max(np.nanmean(combine), 1e-9)),
            "active_experts": int((hist > 0).sum()), "effective_experts": float(np.exp(-np.sum((hist[hist > 0] / max(hist.sum(), 1e-9)) * np.log(np.maximum(hist[hist > 0] / max(hist.sum(), 1e-9), 1e-12)))) if np.any(hist > 0) else 0),
            "dp_pre_moe_values": json.dumps({str(k): float(v) for k, v in pre.items()}),
        }
        records.append(row)
    return pd.DataFrame(records)


def figures(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    for topology, title in (("A", "TP2/DP2/EP4"), ("B", "TP1/DP4/EP4")):
        g = df[df.topology == topology]
        if g.empty:
            continue
        # 1/2: per-topology pre-MoE timeline by DP rank (duration proxy).
        fig, ax = plt.subplots(figsize=(8, 4))
        z = g[g.layer == 0].sort_values(["batch_id", "scheduler_iteration"])
        for i, (batch, b) in enumerate(z.groupby("batch_id")):
            vals = [json.loads(x)[str(dp)] for x in b.dp_pre_moe_values for dp in range(int(g.dp.iloc[0])) if str(dp) in json.loads(x)]
            if vals: ax.plot(np.arange(len(vals)), vals, ".", ms=3)
        ax.set(xlabel="invocation order", ylabel="DP-local pre-MoE duration (ms)", title=title)
        fig.tight_layout(); fig.savefig(out / f"plot{'1' if topology == 'A' else '2'}_dp_group_pre_moe_timeline.png", dpi=160); plt.close(fig)
    # 3: workload imbalance versus arrival skew.
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, g in df.groupby("topology"):
        ax.scatter(g.rank_load_ratio, g.arrival_skew_ratio, s=5, alpha=.3, label=f"{t}: TP{g.tp.iloc[0]}/DP{g.dp.iloc[0]}")
    ax.set(xlabel="max/mean EP assignment load", ylabel="DP arrival-skew ratio"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "plot3_dp_imbalance_vs_arrival_skew.png", dpi=160); plt.close(fig)
    # 4: arrival skew versus EP wait proxy.
    fig, ax = plt.subplots(figsize=(7, 4));
    for t, g in df.groupby("topology"): ax.scatter(g.arrival_skew_ms, g.ep_wait_proxy_ms, s=5, alpha=.3, label=t)
    ax.set(xlabel="arrival-skew duration proxy (ms)", ylabel="EP completion spread proxy (ms)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "plot4_arrival_skew_vs_ep_wait.png", dpi=160); plt.close(fig)
    # 5: topology summary.
    summary = df.groupby(["topology", "concurrency"])["sync_stall_fraction"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4)); summary.plot(ax=ax, marker="o"); ax.set(xlabel="submitted concurrency", ylabel="median stall fraction")
    fig.tight_layout(); fig.savefig(out / "plot5_dp2_vs_dp4_stall.png", dpi=160); plt.close(fig)
    # 6: MLLM versus text control.
    fig, ax = plt.subplots(figsize=(7, 4));
    for label, g in df.groupby("modality"): ax.plot(g.groupby("concurrency").sync_stall_fraction.median(), marker="o", label=label)
    ax.set(xlabel="submitted concurrency", ylabel="median stall fraction"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "plot6_mllm_vs_text.png", dpi=160); plt.close(fig)
    # 7: layer heatmap.
    piv = df.pivot_table(index="layer", columns="topology", values="sync_stall_fraction", aggfunc="median")
    fig, ax = plt.subplots(figsize=(7, 7)); im = ax.imshow(piv.to_numpy().T, aspect="auto", interpolation="nearest")
    ax.set(yticks=np.arange(len(piv.columns)), yticklabels=piv.columns, xlabel="decoder layer", ylabel="topology"); fig.colorbar(im, ax=ax, label="stall fraction")
    fig.tight_layout(); fig.savefig(out / "plot7_per_layer_stall_heatmap.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, nargs="+", required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for r in args.result:
        frame = collect(r); frame.to_csv(r / "arrival_invocation_metrics.csv", index=False); frames.append(frame); figures(frame, r / "figures")
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(); all_df.to_csv(args.output / "arrival_invocation_metrics_all.csv", index=False)
    if not all_df.empty:
        figures(all_df, args.output / "figures")
    summary = []
    if not all_df.empty:
        for (top, budget, cond, c), g in all_df.groupby(["topology", "max_num_batched_tokens", "condition", "concurrency"]):
            summary.append({"topology": top, "max_num_batched_tokens": int(budget), "condition": cond, "concurrency": int(c), "n": len(g), "scheduled_tokens_median": float(g.scheduled_tokens_median.median()), "arrival_skew_ms_median": float(g.arrival_skew_ms.median()), "arrival_skew_ratio_median": float(g.arrival_skew_ratio.median()), "ep_wait_proxy_ms_median": float(g.ep_wait_proxy_ms.median()), "stall_fraction_median": float(g.sync_stall_fraction.median()), "rank_ratio_median": float(g.rank_load_ratio.median()), "expert_ratio_median": float(g.expert_ratio.median()), "dispatch_ratio_median": float(g.dispatch_ratio.median()), "combine_ratio_median": float(g.combine_ratio.median()), "layers": int(g.layer.nunique())})
    pd.DataFrame(summary).to_csv(args.output / "arrival_skew_summary.csv", index=False)
    (args.output / "instrumentation_validation.json").write_text(json.dumps({"status": "NOT_RUN", "reason": "No artificial delay injected; production traces remain unperturbed."}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
