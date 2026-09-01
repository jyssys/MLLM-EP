#!/usr/bin/env python3
"""Aggregate the real scheduler/rank traces and render the forensic plots."""

from __future__ import annotations

import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _scheduler(result: Path) -> dict[str, list[dict[str, Any]]]:
    paths = sorted((result / "scheduler_trace").glob("*.jsonl"))
    if not paths:
        return {}
    # TP workers see the same scheduler output; retain one copy and deduplicate
    # sequence IDs.  The row is an actual scheduler iteration, not a model
    # tuning call.
    rows = _load_jsonl(paths[0])
    seen: set[int] = set()
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seq = int(row["sequence"])
        if seq in seen:
            continue
        seen.add(seq)
        out[str(row.get("batch_id"))].append(row)
    return out


def _raw(result: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    out: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted((result / "raw_live").glob("rank[0-3].jsonl")):
        for row in _load_jsonl(path):
            if not row.get("measured"):
                continue
            out[str(row["batch_id"])][int(row["ep_rank"])].append(row)
    return out


def _match_prefill(batch: str, sched: list[dict[str, Any]], ranks: dict[int, list[dict[str, Any]]]) -> list[tuple[dict[str, Any], dict[int, int]]]:
    """Match real positive scheduler iterations to layer-0 model rows.

    vLLM's eager MoE backend may call its tuning path during the first shape
    encounter.  Those rows are not scheduler iterations and have tiny M.  A
    positive scheduler output has a known total token count, so nearest-load
    matching makes the exclusion explicit and deterministic.
    """
    positives = [r for r in sched if int(r.get("total_num_scheduled_tokens", 0)) > 0]
    available: dict[int, list[dict[str, Any]]] = {}
    for ep, rows in ranks.items():
        available[ep] = sorted(
            [r for r in rows if int(r.get("layer", -1)) == 0],
            key=lambda r: int(r.get("scheduler_iteration", -1)),
        )
    # Match each positive scheduler output to the closest real prefill model
    # forward by its expected per-rank assignment count.  Tuning calls are much
    # smaller and therefore are never selected for these prompt sizes.
    selected: list[tuple[dict[str, Any], dict[int, int]]] = []
    available2 = {ep: sorted([r for r in rows if int(r.get("layer", -1)) == 0], key=lambda r: int(r.get("scheduler_iteration", -1))) for ep, rows in ranks.items()}
    for srow in positives:
        expected = max(1.0, float(srow["total_num_scheduled_tokens"]) * 8.0 / 4.0)
        chosen: dict[int, dict[str, Any]] = {}
        for ep, rows in available2.items():
            if not rows:
                chosen = {}
                break
            idx = min(range(len(rows)), key=lambda i: abs(float(rows[i]["total_assignments"]) - expected))
            chosen[ep] = rows.pop(idx)
        if chosen:
            selected.append((srow, {ep: int(row["scheduler_iteration"]) for ep, row in chosen.items()}))
    return selected


def collect(result: Path) -> pd.DataFrame:
    sched = _scheduler(result)
    raw = _raw(result)
    records: list[dict[str, Any]] = []
    for batch, ranks in raw.items():
        selections = _match_prefill(batch, sched.get(batch, []), ranks)
        for chunk_idx, (srow, selected) in enumerate(selections):
            for layer in range(48):
                rr = []
                for ep, rows in ranks.items():
                    found = [r for r in rows if int(r.get("layer", -1)) == layer and int(r.get("scheduler_iteration", -1)) == selected[ep]]
                    if not found:
                        rr = []
                        break
                    rr.append(found[0])
                if len(rr) != 4:
                    continue
                loads = np.asarray([float(r["total_assignments"]) for r in rr])
                ex = np.asarray([float(r["expert"]["ms"]) for r in rr])
                dis = np.asarray([float(r["dispatch"]["ms"]) for r in rr])
                com = np.asarray([float(r["combine"]["ms"]) for r in rr])
                hists = [np.asarray(r["expert_histogram"], dtype=float) for r in rr]
                h = np.concatenate(hists)
                records.append({
                    "result": str(result), "batch_id": batch,
                    "condition": rr[0].get("condition"), "modality": rr[0].get("modality"),
                    "concurrency": int(rr[0].get("concurrency", 1)),
                    "control_iteration": int(rr[0].get("iteration", -1)),
                    "chunk_index": chunk_idx, "layer": layer,
                    "scheduled_tokens": int(srow["total_num_scheduled_tokens"]),
                    "scheduled_requests": len(srow.get("num_scheduled_tokens", {})),
                    "scheduler_mode": (
                        "prefill" if (
                            int(srow["total_num_scheduled_tokens"]) > len(srow.get("num_scheduled_tokens", {}))
                            or bool(srow.get("scheduled_new_req_ids"))
                        ) else "decode"
                    ),
                    "N": float(loads.sum()), "rank_max": float(loads.max()),
                    "rank_mean": float(loads.mean()), "rank_ratio": float(loads.max() / max(loads.mean(), 1e-9)),
                    "rank_cv": float(loads.std() / max(loads.mean(), 1e-9)),
                    "hottest_rank": int(np.argmax(loads)),
                    "expert_max_ms": float(ex.max()), "expert_mean_ms": float(ex.mean()),
                    "expert_ratio": float(ex.max() / max(ex.mean(), 1e-9)),
                    "dispatch_max_ms": float(dis.max()), "dispatch_mean_ms": float(dis.mean()),
                    "dispatch_ratio": float(dis.max() / max(dis.mean(), 1e-9)),
                    "combine_max_ms": float(com.max()), "combine_mean_ms": float(com.mean()),
                    "combine_ratio": float(com.max() / max(com.mean(), 1e-9)),
                    "active_experts": int((h > 0).sum()),
                    "effective_experts": float(np.exp(-np.sum((h[h > 0] / h.sum()) * np.log(h[h > 0] / h.sum())))),
                    "tiny_expert_fraction": float(np.mean(h[h > 0] <= 4)) if np.any(h > 0) else 0.0,
                })
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["total_ms"] = frame["dispatch_max_ms"] + frame["expert_max_ms"] + frame["combine_max_ms"]
    return frame


def _driver(result: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(result.glob("driver.dp_rank*.json")):
        d = json.loads(path.read_text())
        for r in d.get("records", []):
            if r.get("measured") and int(r.get("driver_dp_rank", -1)) == 0:
                rows.append(r)
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, drivers: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    # 1: rank-load imbalance versus concurrency.
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, g in df.groupby("condition"):
        z = g.groupby("concurrency")["rank_ratio"].median()
        ax.plot(z.index, z.values, marker="o", label=label)
    ax.axhline(1.5, color="k", ls="--", lw=.8, label="gate 1.5")
    ax.set(xlabel="submitted concurrency", ylabel="median max/mean rank assignments")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "plot1_concurrency_rank_imbalance.png", dpi=160); plt.close(fig)
    # 2: expert critical-path imbalance.
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, g in df.groupby("condition"):
        z = g.groupby("concurrency")["expert_ratio"].median()
        ax.plot(z.index, z.values, marker="o", label=label)
    ax.axhline(1.10, color="k", ls="--", lw=.8, label="gate 1.10")
    ax.set(xlabel="submitted concurrency", ylabel="median max/mean expert CUDA time")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "plot2_concurrency_expert_imbalance.png", dpi=160); plt.close(fig)
    # 3: scheduled prefill tokens versus rank/expert ratio.
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, g in df.groupby("condition"):
        ax.scatter(g["scheduled_tokens"], g["expert_ratio"], s=8, alpha=.35, label=label)
    ax.set(xlabel="actual scheduled prefill tokens", ylabel="max/mean expert time")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "plot3_scheduled_tokens_vs_straggler.png", dpi=160); plt.close(fig)
    # 4: co-batched vision/text comparison where scheduled request count > 1.
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, g in df[df.scheduled_requests > 1].groupby("modality"):
        z = g.groupby("concurrency")["expert_max_ms"].median()
        ax.plot(z.index, z.values, marker="o", label=label)
    ax.set(xlabel="submitted concurrency", ylabel="median critical expert ms (co-batched)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "plot4_vision_vs_text_matched_serving.png", dpi=160); plt.close(fig)
    # 5: layer × rank hotness for the largest vision co-batch.
    g = df[(df.modality == "vision") & (df.scheduled_requests == df.scheduled_requests.max())]
    if not g.empty:
        piv = g.pivot_table(index="layer", columns="hottest_rank", values="expert_ratio", aggfunc="median")
        fig, ax = plt.subplots(figsize=(8, 5)); im = ax.imshow(piv.fillna(0).T, aspect="auto", interpolation="nearest")
        ax.set(xlabel="layer", ylabel="hottest rank"); fig.colorbar(im, ax=ax, label="expert max/mean")
        fig.tight_layout(); fig.savefig(out / "plot5_layer_hot_rank_heatmap.png", dpi=160); plt.close(fig)
    # 6: scheduler trace timeline for the largest measured co-batch.
    g = df[(df.modality == "vision") & (df.scheduled_requests == df.scheduled_requests.max())].sort_values(["control_iteration", "chunk_index", "layer"])
    if not g.empty:
        z = g[g.layer == 0]
        fig, ax1 = plt.subplots(figsize=(8, 4)); ax1.plot(np.arange(len(z)), z.scheduled_tokens, marker=".", label="scheduled tokens")
        ax1.set_ylabel("scheduled prefill tokens"); ax2 = ax1.twinx(); ax2.plot(np.arange(len(z)), z.rank_ratio, color="tab:red", marker=".", label="rank ratio")
        ax2.set_ylabel("max/mean rank load"); ax1.set_xlabel("positive scheduler iteration")
        fig.tight_layout(); fig.savefig(out / "plot6_scheduler_iteration_timeline.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, nargs="+", required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); frames = []
    for path in args.result:
        f = collect(path)
        f.to_csv(path / "analysis_invocation_metrics.csv", index=False)
        _plot(f, _driver(path), path / "figures")
        frames.append(f)
    all_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_frame.to_csv(args.output / "invocation_metrics_all.csv", index=False)
    summary = []
    if not all_frame.empty:
        for (res, cond, conc), g in all_frame.groupby(["result", "condition", "concurrency"]):
            summary.append({"result": res, "condition": cond, "concurrency": int(conc), "n": len(g), "scheduled_tokens_median": float(g.scheduled_tokens.median()), "scheduled_requests_median": float(g.scheduled_requests.median()), "rank_ratio_median": float(g.rank_ratio.median()), "rank_ratio_p95": float(g.rank_ratio.quantile(.95)), "expert_ratio_median": float(g.expert_ratio.median()), "expert_ratio_p95": float(g.expert_ratio.quantile(.95)), "expert_max_ms_median": float(g.expert_max_ms.median()), "dispatch_max_ms_median": float(g.dispatch_max_ms.median()), "combine_max_ms_median": float(g.combine_max_ms.median()), "tail15_fraction": float(np.mean(g.expert_ratio >= 1.15)), "critical_rank": int(g.hottest_rank.mode().iloc[0])})
    pd.DataFrame(summary).to_csv(args.output / "serving_summary.csv", index=False)
    print(args.output)


if __name__ == "__main__":
    main()
