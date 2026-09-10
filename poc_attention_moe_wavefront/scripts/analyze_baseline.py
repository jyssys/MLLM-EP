#!/usr/bin/env python3
"""Validate stock traces and compute the optimistic Phase-0 O1–O4 gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from poc_attention_moe_wavefront.wavefront.oracle import candidate_splits, ideal_wave_ms


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def load_drivers(run: Path) -> pd.DataFrame:
    payload = json.loads((run / "driver.dp_rank0.json").read_text())
    if not payload["ok"]:
        raise RuntimeError(payload["traceback"])
    return pd.DataFrame(payload["records"])


def load_stage(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((run / "raw").glob("rank*.json")):
        payload = json.loads(path.read_text())
        if payload["visible_devices"] != "4,5,6,7":
            raise AssertionError(payload["visible_devices"])
        rows.extend(payload["rows"])
    if not rows:
        raise RuntimeError(f"no stage trace in {run}")
    frame = pd.DataFrame(rows)
    # Logical invocation latency is the slowest same-device rank duration.
    keys = ["run_id", "wave", "request_id", "workload_id", "iteration", "layer", "stage", "tokens"]
    return frame.groupby(keys, as_index=False).duration_ms.max()


def paired_tax(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    rows = []
    for workload, local in frame[~frame.warmup].groupby("workload_id"):
        clean = local[~local.instrumented].set_index("iteration")
        profiled = local[local.instrumented].set_index("iteration")
        common = clean.index.intersection(profiled.index)
        ratios = [100 * (profiled.loc[i].first_token_latency_s / clean.loc[i].first_token_latency_s - 1) for i in common]
        rows.append({"workload_id": workload, "median_overhead_pct": statistics.median(ratios),
                     "p90_overhead_pct": percentile(ratios, .9), "pairs": len(ratios)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    manifests = [json.loads((run / "workload_manifest.json").read_text()) for run in args.runs]
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise AssertionError("workload manifests differ")
    manifest = manifests[0]
    drivers = []
    stages = []
    for run in args.runs:
        driver = load_drivers(run)
        driver["source_run"] = run.name
        drivers.append(driver)
        if any((run / "raw").glob("rank*.json")):
            stage = load_stage(run)
            stage["source_run"] = run.name
            stages.append(stage)
    driver = pd.concat(drivers, ignore_index=True)
    stage = pd.concat(stages, ignore_index=True)
    clean = driver[(~driver.warmup) & (~driver.instrumented)].copy()
    clean["ttft_ms"] = 1000 * clean.first_token_latency_s
    clean.to_csv(args.root / "clean_request_metrics.csv", index=False)
    stage.to_csv(args.root / "baseline_stage_trace.csv", index=False)

    critical = stage[stage.layer >= 0].groupby(
        ["workload_id", "layer", "stage"], as_index=False
    ).duration_ms.median()
    piv = critical.pivot(index=["workload_id", "layer"], columns="stage", values="duration_ms").reset_index()
    required = {"attention_total", "dispatch", "expert", "combine", "moe_total", "layer_total"}
    if not required.issubset(piv.columns) or any(len(local) != 48 for _, local in piv.groupby("workload_id")):
        raise AssertionError((piv.columns.tolist(), piv.groupby("workload_id").size().to_dict()))
    piv.to_csv(args.root / "baseline_layer_medians.csv", index=False)

    split_rows = []
    for _, row in piv.iterrows():
        workload = row.workload_id
        tokens = int(manifest[workload]["processor_prompt_tokens"])
        boundaries = manifest[workload]["modality_boundaries"]
        for split in candidate_splits(tokens, boundaries):
            fraction = split / tokens
            # This is deliberately optimistic: logical token-progress fractions
            # of the already measured unsplit kernels, with zero extra launches,
            # copies, packing, or materialization.  It is an upper-bound oracle,
            # not a claim that current FlashAttention exposes tokens linearly.
            ah = fraction * row.attention_total
            at = (1 - fraction) * row.attention_total
            eh = fraction * row.moe_total
            et = (1 - fraction) * row.moe_total
            wave = ideal_wave_ms(ah, at, eh, et)
            split_rows.append({
                "workload_id": workload, "layer": int(row.layer), "tokens": tokens,
                "split": split, "split_fraction": fraction,
                "is_modality_boundary": split in boundaries,
                "attention_all_ms": row.attention_total, "moe_all_ms": row.moe_total,
                "attention_head_ms": ah, "attention_tail_ms": at,
                "moe_head_ms": eh, "moe_tail_ms": et,
                "base_ms": row.attention_total + row.moe_total,
                "ideal_wave_ms": wave,
                "ideal_improvement_pct": 100 * (1 - wave / (row.attention_total + row.moe_total)),
            })
    split = pd.DataFrame(split_rows)
    split.to_csv(args.root / "split_oracle.csv", index=False)

    workloads = sorted(manifest)
    fractions = sorted(split.split_fraction.round(6).unique())
    o4_fraction = min(fractions, key=lambda f: split.assign(d=(split.split_fraction-f).abs()).sort_values("d").groupby(["workload_id", "layer"]).first().ideal_wave_ms.sum())
    summary_rows = []
    choices = []
    for workload in workloads:
        local = split[split.workload_id == workload]
        base = float(local.groupby("layer").base_ms.first().sum())
        ttft = float(clean[clean.workload_id == workload].ttft_ms.median())
        o1_rows = local.loc[local.groupby("layer").ideal_wave_ms.idxmin()]
        o1 = float(o1_rows.ideal_wave_ms.sum())
        request_fractions = sorted(local.split_fraction.round(6).unique())
        def cost_at(fraction: float) -> tuple[float, pd.DataFrame]:
            selected = local.assign(distance=(local.split_fraction-fraction).abs()).sort_values(["layer", "distance", "split"]).groupby("layer", as_index=False).first()
            return float(selected.ideal_wave_ms.sum()), selected
        o2_fraction = min(request_fractions, key=lambda f: cost_at(f)[0])
        o2, o2_rows = cost_at(o2_fraction)
        o4, o4_rows = cost_at(float(o4_fraction))
        boundaries = manifest[workload]["modality_boundaries"]
        if boundaries:
            boundary_costs = [(boundary, cost_at(boundary / int(manifest[workload]["processor_prompt_tokens"]))) for boundary in boundaries]
            o3_boundary, (o3, o3_rows) = min(boundary_costs, key=lambda item: item[1][0])
        else:
            o3_boundary, o3, o3_rows = None, math.nan, None
        row = {
            "workload_id": workload, "kind": manifest[workload]["kind"],
            "tokens": int(manifest[workload]["processor_prompt_tokens"]),
            "vision_tokens": int(manifest[workload]["processor_vision_tokens"]),
            "vision_fraction": manifest[workload]["processor_vision_tokens"] / manifest[workload]["processor_prompt_tokens"],
            "clean_ttft_ms": ttft, "affected_base_ms": base,
            "attention_ms": float(local.groupby("layer").attention_all_ms.first().sum()),
            "moe_ms": float(local.groupby("layer").moe_all_ms.first().sum()),
            "O1_ms": o1, "O2_ms": o2, "O3_ms": o3, "O4_ms": o4,
            "O2_fraction": float(o2_fraction), "O3_boundary": o3_boundary,
            "O4_fraction": float(o4_fraction),
        }
        for oracle in ("O1", "O2", "O3", "O4"):
            value = row[f"{oracle}_ms"]
            row[f"{oracle}_ttft_gain_pct"] = math.nan if math.isnan(value) else 100 * min(ttft, base-value) / ttft
        summary_rows.append(row)
        boundary = o3_boundary
        for _, value in o1_rows.iterrows():
            choices.append({"workload_id": workload, "layer": int(value.layer), "optimal_split": int(value.split),
                            "optimal_fraction": float(value.split_fraction), "modality_boundary": boundary,
                            "normalized_distance": math.nan if boundary is None else abs(value.split-boundary)/value.tokens})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.root / "oracle_summary.csv", index=False)
    pd.DataFrame(choices).to_csv(args.root / "optimal_split_analysis.csv", index=False)
    tax = []
    for run in args.runs:
        tax.extend({"source_run": run.name, **row} for row in paired_tax(load_drivers(run)))
    pd.DataFrame(tax).to_csv(args.root / "instrumentation_overhead.csv", index=False)

    target = summary[summary.vision_tokens > 0]
    median_o1 = float(target.O1_ttft_gain_pct.median())
    gate = "NO-GO" if median_o1 < 8 else ("HOLD" if median_o1 < 15 else "CONTINUE_SPLIT_TAX")
    payload = {
        "modeling_scope": "optimistic proportional-progress, zero split/copy/launch tax upper bound",
        "median_multimodal_O1_amdahl_ttft_gain_pct": median_o1,
        "median_multimodal_O2_amdahl_ttft_gain_pct": float(target.O2_ttft_gain_pct.median()),
        "median_multimodal_O3_amdahl_ttft_gain_pct": float(target.O3_ttft_gain_pct.median()),
        "median_multimodal_O4_amdahl_ttft_gain_pct": float(target.O4_ttft_gain_pct.median()),
        "median_instrumentation_overhead_pct": float(pd.DataFrame(tax).median_overhead_pct.median()),
        "gate": gate,
        "naive_split_allowed": gate == "CONTINUE_SPLIT_TAX",
        "global_static_fraction": float(o4_fraction),
    }
    (args.root / "stage_c_gate.json").write_text(json.dumps(payload, indent=2) + "\n")

    figures = args.root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    components = piv.groupby("workload_id")[["attention_total", "dispatch", "expert", "combine", "moe_total"]].sum()
    stage_plot = components[["attention_total", "dispatch", "expert", "combine"]].copy()
    stage_plot["MoE other"] = components.moe_total - components[["dispatch", "expert", "combine"]].sum(axis=1)
    stage_plot["TTFT other"] = summary.set_index("workload_id").clean_ttft_ms - components.attention_total - components.moe_total
    stage_plot.columns = ["Attention", "Dispatch", "Expert", "Combine", "MoE other", "TTFT other"]
    stage_plot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Clean TTFT and attributed critical-rank CUDA time (ms)")
    fig.tight_layout(); fig.savefig(figures / "01_attention_moe_breakdown.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    summary.set_index("workload_id")[["O1_ttft_gain_pct", "O2_ttft_gain_pct", "O3_ttft_gain_pct", "O4_ttft_gain_pct"]].plot(kind="bar", ax=ax)
    ax.axhline(8, color="red", linestyle="--"); ax.axhline(15, color="green", linestyle="--")
    ax.set_ylabel("Amdahl-adjusted TTFT reduction (%)")
    fig.tight_layout(); fig.savefig(figures / "02_O1_O2_O3_O4.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for workload, local in split.groupby("workload_id"):
        curve = local.groupby("split_fraction").ideal_improvement_pct.median()
        ax.plot(curve.index, curve.values, marker=".", label=workload)
    ax.set_xlabel("Split fraction"); ax.set_ylabel("Affected-region ideal reduction (%)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "03_split_fraction_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    choice = pd.DataFrame(choices).dropna()
    choice_plot = choice.merge(summary[["workload_id", "tokens"]], on="workload_id")
    ax.scatter(choice_plot.modality_boundary.to_numpy() / choice_plot.tokens.to_numpy(),
               choice_plot.optimal_fraction.to_numpy(), alpha=.6)
    ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("Modality boundary fraction"); ax.set_ylabel("O1 split fraction")
    fig.tight_layout(); fig.savefig(figures / "04_optimum_vs_boundary.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(summary.vision_fraction, summary.O1_ttft_gain_pct)
    for _, row in summary.iterrows(): ax.annotate(row.workload_id, (row.vision_fraction, row.O1_ttft_gain_pct), fontsize=7)
    ax.set_xlabel("Vision-token fraction"); ax.set_ylabel("O1 TTFT oracle (%)")
    fig.tight_layout(); fig.savefig(figures / "05_vision_fraction_oracle.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for workload, local in pd.DataFrame(choices).groupby("workload_id"):
        ax.plot(local.layer, local.optimal_fraction, marker=".", label=workload)
    ax.set_xlabel("Layer"); ax.set_ylabel("O1 optimal split fraction"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "04b_layerwise_optimal_split.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    best = split.loc[split.groupby(["workload_id", "layer"]).ideal_wave_ms.idxmin()].copy()
    for workload, local in best.groupby("workload_id"):
        ax.plot(local.layer, local.ideal_improvement_pct, marker=".", label=workload)
    ax.set_xlabel("Layer"); ax.set_ylabel("Affected-region O1 gain (%)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "05b_layerwise_oracle_gain.png", dpi=180); plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
