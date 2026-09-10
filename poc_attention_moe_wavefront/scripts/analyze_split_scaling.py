#!/usr/bin/env python3
"""Build fragment-aware empirical split curves and O1--O4 diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poc_attention_moe_wavefront.wavefront.oracle import DEFAULT_FRACTIONS, ideal_wave_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    run = args.root / "split_scaling"
    schedule = pd.DataFrame(json.loads((run / "schedule.json").read_text()))
    schedule = schedule[schedule.phase == "measured"].copy()
    raw_rows = []
    for path in sorted((run / "raw").glob("rank[0-9].json")):
        payload = json.loads(path.read_text())
        if payload["visible_devices"] != "4,5,6,7":
            raise AssertionError(payload["visible_devices"])
        for row in payload["stage_records"]:
            raw_rows.append({"ep_rank": int(payload["ep_rank"]), **row})
    if len({row["ep_rank"] for row in raw_rows}) != 4:
        raise RuntimeError("four-rank traces are required")
    raw = pd.DataFrame(raw_rows)
    trace = raw.groupby(
        ["wave", "request_id", "iteration", "layer", "segment", "stage"],
        as_index=False,
    ).duration_ms.max()
    trace = trace.merge(
        schedule[["wave", "workload_id", "prefix_tokens", "tail_tokens", "prompt_tokens", "split_fraction", "repetition"]],
        on="wave",
        validate="many_to_one",
    )
    trace.to_csv(args.root / "empirical_split_stage_trace.csv", index=False)

    median = trace.groupby(
        ["workload_id", "request_id", "prompt_tokens", "prefix_tokens", "tail_tokens", "split_fraction", "layer", "segment", "stage"],
        as_index=False,
    ).duration_ms.median()
    piv = median.pivot_table(
        index=["workload_id", "request_id", "prompt_tokens", "prefix_tokens", "tail_tokens", "split_fraction", "layer"],
        columns=["segment", "stage"], values="duration_ms",
    ).reset_index()
    required = [("prefix", "attention"), ("tail", "attention"), ("prefix", "moe_total"), ("tail", "moe_total")]
    if not all(column in piv.columns for column in required):
        raise AssertionError(piv.columns.tolist())
    flat = pd.DataFrame({column[0]: piv[column] for column in piv.columns[:7]})
    flat["attention_head_ms"] = piv[("prefix", "attention")].to_numpy()
    flat["attention_tail_ms"] = piv[("tail", "attention")].to_numpy()
    flat["moe_head_ms"] = piv[("prefix", "moe_total")].to_numpy()
    flat["moe_tail_ms"] = piv[("tail", "moe_total")].to_numpy()
    flat["fragment_attention_ms"] = flat.attention_head_ms + flat.attention_tail_ms
    flat["fragment_moe_ms"] = flat.moe_head_ms + flat.moe_tail_ms
    flat["fragment_wave_ms"] = (
        flat.attention_head_ms
        + np.maximum(flat.attention_tail_ms, flat.moe_head_ms)
        + flat.moe_tail_ms
    )
    base = pd.read_csv(args.root / "baseline_layer_medians.csv")
    flat = flat.merge(
        base[["workload_id", "layer", "attention_total", "moe_total"]],
        on=["workload_id", "layer"], validate="many_to_one",
    )
    flat["base_affected_ms"] = flat.attention_total + flat.moe_total
    flat["attention_fragment_ratio"] = flat.fragment_attention_ms / flat.attention_total
    flat["moe_fragment_ratio"] = flat.fragment_moe_ms / flat.moe_total
    flat["fragment_wave_gain_pct"] = 100 * (1 - flat.fragment_wave_ms / flat.base_affected_ms)
    flat.to_csv(args.root / "empirical_split_curves.csv", index=False)

    manifest = json.loads((args.root / "baseline_v3/workload_manifest.json").read_text())
    clean = pd.read_csv(args.root / "oracle_summary.csv").set_index("workload_id")
    summaries = []
    global_fractions = list(DEFAULT_FRACTIONS)

    def nearest(local: pd.DataFrame, fraction: float) -> pd.DataFrame:
        return local.assign(distance=(local.split_fraction - fraction).abs()).sort_values(
            ["layer", "distance", "prefix_tokens"]
        ).groupby("layer", as_index=False).first()

    # O4 is selected on a shared normalized fraction across all requests.
    global_fraction = min(
        global_fractions,
        key=lambda fraction: sum(nearest(local, fraction).fragment_wave_ms.sum() for _, local in flat.groupby("workload_id")),
    )
    for workload_id, local in flat.groupby("workload_id"):
        base_ms = float(local.groupby("layer").base_affected_ms.first().sum())
        ttft = float(clean.loc[workload_id, "clean_ttft_ms"])
        o1_rows = local.loc[local.groupby("layer").fragment_wave_ms.idxmin()]
        o1 = float(o1_rows.fragment_wave_ms.sum())
        fractions = sorted(local.split_fraction.unique())
        o2_fraction = min(fractions, key=lambda fraction: float(nearest(local, fraction).fragment_wave_ms.sum()))
        o2 = float(nearest(local, o2_fraction).fragment_wave_ms.sum())
        boundaries = [int(value) for value in manifest[workload_id]["modality_boundaries"]]
        if boundaries:
            boundary_costs = []
            for boundary in boundaries:
                rows = local[local.prefix_tokens == boundary]
                if len(rows) == 48:
                    boundary_costs.append((boundary, float(rows.fragment_wave_ms.sum())))
            o3_boundary, o3 = min(boundary_costs, key=lambda pair: pair[1])
        else:
            o3_boundary, o3 = None, math.nan
        o4 = float(nearest(local, global_fraction).fragment_wave_ms.sum())
        row = {
            "workload_id": workload_id,
            "fragment_O1_ms": o1, "fragment_O2_ms": o2,
            "fragment_O3_ms": o3, "fragment_O4_ms": o4,
            "fragment_O2_fraction": o2_fraction,
            "fragment_O3_boundary": o3_boundary,
            "fragment_O4_fraction": global_fraction,
            "base_affected_ms": base_ms, "clean_ttft_ms": ttft,
            "median_attention_fragment_ratio": float(local.attention_fragment_ratio.median()),
            "median_moe_fragment_ratio": float(local.moe_fragment_ratio.median()),
        }
        for oracle in ("O1", "O2", "O3", "O4"):
            value = row[f"fragment_{oracle}_ms"]
            row[f"fragment_{oracle}_ttft_gain_pct"] = math.nan if math.isnan(value) else 100 * (base_ms - value) / ttft
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.root / "empirical_oracle_summary.csv", index=False)
    multimodal = summary[summary.workload_id != "W1_text"]
    payload = {
        "scope": "fragment-aware exact sequential split kernels, then zero-contention overlap",
        "splits_measured": int(len(schedule)),
        "repetitions_per_split": int(schedule.repetition.nunique()),
        "median_attention_fragment_ratio": float(flat.attention_fragment_ratio.median()),
        "median_moe_fragment_ratio": float(flat.moe_fragment_ratio.median()),
        "median_multimodal_fragment_O1_ttft_gain_pct": float(multimodal.fragment_O1_ttft_gain_pct.median()),
        "median_multimodal_fragment_O2_ttft_gain_pct": float(multimodal.fragment_O2_ttft_gain_pct.median()),
        "median_multimodal_fragment_O3_ttft_gain_pct": float(multimodal.fragment_O3_ttft_gain_pct.median()),
        "median_multimodal_fragment_O4_ttft_gain_pct": float(multimodal.fragment_O4_ttft_gain_pct.median()),
        "global_fragment_O4_fraction": float(global_fraction),
        "linear_scaling_validated": bool(
            abs(float(flat.attention_fragment_ratio.median()) - 1) <= .05
            and abs(float(flat.moe_fragment_ratio.median()) - 1) <= .05
        ),
    }
    (args.root / "empirical_scaling_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    figures = args.root / "figures"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for workload_id, local in flat.groupby("workload_id"):
        curve = local.groupby("split_fraction").fragment_wave_gain_pct.median()
        ax.plot(curve.index, curve.values, marker=".", label=workload_id)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xlabel("Split fraction"); ax.set_ylabel("Fragment-aware affected-region gain (%)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(figures / "09_empirical_split_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    summary.set_index("workload_id")[["fragment_O1_ttft_gain_pct", "fragment_O2_ttft_gain_pct", "fragment_O3_ttft_gain_pct", "fragment_O4_ttft_gain_pct"]].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Fragment-aware TTFT oracle (%)")
    fig.tight_layout(); fig.savefig(figures / "10_empirical_O1_O4.png", dpi=180); plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
