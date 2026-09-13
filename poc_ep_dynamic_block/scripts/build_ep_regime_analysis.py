#!/usr/bin/env python3
"""Create the AR-ness x EP physical-regime tables and figures.

Clean request time and observer-heavy component traces remain deliberately
separate.  Rank-local trace rows were already collapsed with critical-rank
latencies by ``analyze_block_traces.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.rank().corr(y.rank()))


def aggregate_waves(waves: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "task", "sample_count", "block_length", "mini_batch_size",
        "generation", "threshold", "target_total_length", "repeat",
    ]
    metrics = [
        "physical_rows", "live_ratio", "_backfill", "model_forward_ms",
        "active_experts", "rows_per_active_expert_mean",
        "rows_per_active_expert_p50", "tiny_expert_fraction_le4",
        "rank_load_cv", "rank_fanout_mean", "remote_pairs",
        "remote_fraction", "dispatch_bytes_bf16", "attention_critical_ms",
        "router_critical_ms", "dispatch_critical_ms", "expert_critical_ms",
        "combine_critical_ms", "shared_critical_ms", "mlp_critical_ms",
    ]
    metrics.remove("live_ratio")
    metrics.remove("physical_rows")
    metrics.remove("_backfill")
    rows = []
    for identity, group in waves.groupby(keys, dropna=False):
        row = dict(zip(keys, identity))
        row["waves"] = len(group)
        row["physical_rows_median"] = group.physical_rows.median()
        row["physical_rows_p90"] = group.physical_rows.quantile(.9)
        row["decision_live_ratio_median"] = group.live_ratio.median()
        row["dead_work_fraction_median"] = 1 - group.live_ratio.median()
        for metric in metrics:
            if metric in group:
                row[f"{metric}_median"] = group[metric].median()
                row[f"{metric}_mean"] = group[metric].mean()
        stage = [
            group.attention_critical_ms.median(),
            group.dispatch_critical_ms.median(),
            group.expert_critical_ms.median(),
            group.combine_critical_ms.median(),
        ]
        total = sum(stage)
        for name, value in zip(("attention", "dispatch", "expert", "combine"), stage):
            row[f"sampled_component_{name}_share"] = value / total if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def line_panels(df: pd.DataFrame, y: str, ylabel: str, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    for axis, task in zip(axes, ("gsm8k", "humaneval")):
        data = df[(df.task == task) & (df.sample_count == 8) & (df.mini_batch_size == 8)]
        data = data.sort_values("block_length")
        if len(data):
            axis.plot(data.block_length, data[y], marker="o")
        axis.set_title(task)
        axis.set_xlabel("block size B")
        axis.grid(alpha=.25)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    waves = pd.read_csv(ROOT / "BLOCK_WAVE_METRICS.csv")
    runs = pd.read_csv(ROOT / "TRACE_RUN_SUMMARY.csv")
    raw_clean = pd.read_csv(ROOT / "STATIC_BLOCK_SWEEP.csv")
    regime = aggregate_waves(waves)
    regime["configured_wave_rows"] = (
        regime.block_length * regime.mini_batch_size
    )
    regime["wave_fill_ratio_median"] = (
        regime.physical_rows_median / regime.configured_wave_rows
    )
    regime["token_expert_pairs_median"] = regime.physical_rows_median * 8
    regime.to_csv(ROOT / "EP_REGIME_MAP.csv", index=False)

    matched = regime[
        (regime.sample_count == 32)
        & (regime.target_total_length == 256)
        & (regime.configured_wave_rows == 256)
    ].copy()
    matched["matched_M_target"] = 256
    matched.to_csv(ROOT / "MATCHED_M_ANALYSIS.csv", index=False)

    effort = raw_clean[
        raw_clean.regime.isin(["effort", "effortnoeos", "matchedeffort"])
    ].copy()
    effort["early_stop"] = ~effort.regime.isin(["effortnoeos", "matchedeffort"])
    effort["harness_fidelity"] = np.where(
        effort.regime == "effort", "pre-fix threshold overwritten", "valid"
    )
    effort.to_csv(ROOT / "DECODER_EFFORT_CONTROLS.csv", index=False)

    clean_candidates = raw_clean[
        (raw_clean.n == 8)
        & (raw_clean.mini == 8)
        & (raw_clean.target_total > 0)
        & (raw_clean.threshold_effective == .9)
        & (raw_clean.regime == "effort")
    ].copy()
    clean_candidates = clean_candidates.rename(
        columns={
            "block": "block_length",
            "mini": "mini_batch_size",
            "bct_s": "clean_bct_s",
        }
    )
    observer = runs.merge(
        clean_candidates[
            ["task", "n", "block_length", "mini_batch_size", "clean_bct_s"]
        ],
        left_on=["task", "sample_count", "block_length", "mini_batch_size"],
        right_on=["task", "n", "block_length", "mini_batch_size"],
        how="left",
    )
    observer["observer_tax_percent"] = 100 * (
        observer.trace_wall_s / observer.clean_bct_s - 1
    )
    observer.to_csv(ROOT / "OBSERVER_TAX.csv", index=False)

    corr_rows = []
    for task in ("gsm8k", "humaneval"):
        data = regime[(regime.task == task) & (regime.sample_count == 8) & (regime.mini_batch_size == 8)]
        for metric in (
            "model_forward_ms_median", "active_experts_median",
            "rows_per_active_expert_mean_median", "tiny_expert_fraction_le4_median",
            "rank_load_cv_median", "rank_fanout_mean_median",
            "remote_fraction_median", "dispatch_critical_ms_median",
            "expert_critical_ms_median", "combine_critical_ms_median",
            "decision_live_ratio_median",
        ):
            corr_rows.append({
                "task": task,
                "control": "fixed_mini8_n8",
                "x": "block_length",
                "y": metric,
                "spearman_rho": spearman(data.block_length, data[metric]),
                "points": len(data),
            })
    pd.DataFrame(corr_rows).to_csv(ROOT / "BLOCK_REGIME_CORRELATIONS.csv", index=False)

    plot_specs = (
        ("model_forward_ms_median", "instrumented model-forward wall (ms)", "03_block_size_vs_trace_forward.png"),
        ("dispatch_bytes_bf16_median", "BF16 dispatch bytes", "05_block_size_vs_remote_bytes.png"),
        ("token_expert_pairs_median", "token-expert pairs", "06_block_size_vs_token_expert_pairs.png"),
        ("rows_per_active_expert_mean_median", "rows / active expert", "07_block_size_vs_rows_per_expert.png"),
        ("active_experts_median", "active experts", "08_block_size_vs_active_experts.png"),
        ("rank_fanout_mean_median", "mean destination-rank fanout", "09_block_size_vs_rank_fanout.png"),
        ("rank_load_cv_median", "rank-load CV", "10_block_size_vs_rank_load_cv.png"),
        ("decision_live_ratio_median", "decision-live / physical rows", "12_block_size_vs_live_ratio.png"),
        ("dead_work_fraction_median", "hypothetical removable row fraction", "14_block_size_vs_dead_work.png"),
    )
    for y, ylabel, filename in plot_specs:
        line_panels(regime, y, ylabel, filename)

    if len(matched):
        fig, axis = plt.subplots(figsize=(6.5, 3.8))
        data = matched.sort_values("block_length")
        axis.plot(data.block_length, data.model_forward_ms_median, marker="o")
        axis.set_xlabel("block size B (configured M=256)")
        axis.set_ylabel("instrumented forward wall (ms)")
        axis.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(FIG / "14_matched_M_EP_comparison.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True, sharey=True)
    for axis, task in zip(axes, ("gsm8k", "humaneval")):
        data = regime[(regime.task == task) & (regime.sample_count == 8) & (regime.mini_batch_size == 8)].sort_values("block_length")
        for stage in ("attention", "dispatch", "expert", "combine"):
            axis.plot(data.block_length, 100 * data[f"sampled_component_{stage}_share"], marker="o", label=stage)
        axis.set_title(task)
        axis.set_xlabel("block size B")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("sampled component share (%)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "11_block_size_vs_component_share.png", dpi=180)
    plt.close(fig)

    print(
        f"wrote {len(regime)} regime rows, {len(matched)} matched-M rows, "
        f"{len(observer)} observer-tax rows"
    )


if __name__ == "__main__":
    main()
