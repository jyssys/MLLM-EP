#!/usr/bin/env python3
"""Create compact, auditable tables and the figures required by the PoC spec."""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DERIVED = RESULTS / "final_analysis"
PLOTS = DERIVED / "plots"
MOE_SHARE_OF_TTFT = 0.6219106193174727


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_quality() -> pd.DataFrame:
    records: list[dict] = []
    for path_string in glob.glob(str(RESULTS / "*quality*20260911/quality_*.jsonl")):
        path = Path(path_string)
        root = path.parent.name
        for row in read_jsonl(path):
            records.append({"experiment": root, **row})
    frame = pd.DataFrame(records)
    grouped = []
    for (experiment, policy), part in frame.groupby(["experiment", "policy"]):
        grouped.append({
            "experiment": experiment,
            "policy": policy,
            "requests": len(part),
            "datasets": ";".join(sorted(part.dataset.unique())),
            "assignment_drop_pct": 100 * part.assignment_drop_fraction.median(),
            "max_rank_reduction_pct": 100 * part.max_rank_reduction_median.median(),
            "mean_kl_per_answer_token": part.kl_per_answer_token.mean(),
            "median_kl_per_answer_token": part.kl_per_answer_token.median(),
            "short_greedy_exact_pct": 100 * part.short_greedy_exact.mean(),
            "benchmark_correct": int(part.benchmark_correct.sum()),
            "reference_benchmark_correct": int(part.reference_benchmark_correct.sum()),
            "benchmark_delta_pp": 100 * (part.benchmark_correct.mean() -
                                          part.reference_benchmark_correct.mean()),
        })
    result = pd.DataFrame(grouped).sort_values(["experiment", "policy"])
    result.to_csv(DERIVED / "quality_summary.csv", index=False)
    frame.to_csv(DERIVED / "quality_rows.csv", index=False)
    return result


def summarize_replay() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    patterns = ["deepep_replay_global_*.jsonl", "deepep_replay_trivial_*.jsonl",
                "deepep_replay_exactbudget_*.jsonl"]
    for pattern in patterns:
        for path_string in glob.glob(str(RESULTS / pattern)):
            path = Path(path_string)
            rows = pd.DataFrame(read_jsonl(path))
            if rows.empty:
                continue
            stock = rows[rows.policy == "stock"].moe_ms.median()
            for policy, part in rows.groupby("policy"):
                first = part.iloc[0]
                median_moe = part.moe_ms.median()
                records.append({
                    "experiment": path.stem,
                    "sample": first["sample"], "layer": int(first["layer"]),
                    "policy": policy, "repetitions": len(part),
                    "vision_assignment_drop_pct": 100 * first["vision_assignment_drop"],
                    "all_assignment_drop_pct": 100 * first["all_assignment_drop"],
                    "max_rank_reduction_pct": 100 * first["max_rank_reduction"],
                    "dispatch_ms": part.dispatch_ms.median(),
                    "expert_ms": part.expert_ms.median(),
                    "combine_ms": part.combine_ms.median(),
                    "moe_ms": median_moe,
                    "moe_reduction_pct": 100 * (1 - median_moe / stock),
                    "projected_ttft_reduction_pct": 100 * (1 - median_moe / stock) *
                                                    MOE_SHARE_OF_TTFT,
                    "worst_cosine": part.worst_cosine.median(),
                    "worst_rel_l2_pct": 100 * part.worst_rel_l2.median(),
                })
    result = pd.DataFrame(records).sort_values(["experiment", "policy"])
    result.to_csv(DERIVED / "replay_summary.csv", index=False)
    exact = result[result.experiment.str.contains("exactbudget")]
    l24 = exact if not exact.empty else result[result.layer == 24]
    aggregate = l24.groupby("policy", as_index=False).agg(
        samples=("sample", "nunique"),
        vision_assignment_drop_pct=("vision_assignment_drop_pct", "median"),
        max_rank_reduction_pct=("max_rank_reduction_pct", "median"),
        moe_reduction_pct=("moe_reduction_pct", "median"),
        projected_ttft_reduction_pct=("projected_ttft_reduction_pct", "median"),
        worst_rel_l2_pct=("worst_rel_l2_pct", "median"),
    ).sort_values("projected_ttft_reduction_pct")
    aggregate.to_csv(DERIVED / "replay_l24_aggregate.csv", index=False)
    return result, aggregate


def save_plot(name: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=170)
    plt.close()


def required_plots(quality: pd.DataFrame, replay: pd.DataFrame,
                   aggregate: pd.DataFrame) -> None:
    sensitivity_paths = glob.glob(str(RESULTS / "semantic_sensitivity_*20260911/sensitivity_*.jsonl"))
    sensitivity = pd.DataFrame(row for path in sensitivity_paths
                               for row in read_jsonl(Path(path)))
    sensitivity = sensitivity[sensitivity.group >= 0]
    plt.hist(np.log10(sensitivity.kl_per_answer_token.clip(lower=1e-9)), bins=50)
    plt.xlabel("log10 answer-token KL from one spatial-group perturbation")
    plt.ylabel("count")
    save_plot("01_token_semantic_risk.png")

    router = pd.read_csv(ROOT.parent / "poc_vision_heterogeneous_topk/results/final_analysis/"
                         "router_mass_distribution.csv")
    means = router.groupby("modality")[[f"top{k}_mass_median" for k in range(1, 9)]].median()
    for modality, row in means.iterrows():
        plt.plot(range(1, 9), row, marker="o", label=modality)
    plt.xlabel("retained top-k"); plt.ylabel("median cumulative router mass"); plt.legend()
    save_plot("02_vision_text_router_mass.png")

    heldout = quality[quality.experiment.str.contains("heldout32")]
    for experiment, part in heldout.groupby("experiment"):
        plt.scatter(part.assignment_drop_pct, part.benchmark_delta_pp, label=experiment)
    plt.axhline(0, color="black", lw=.7); plt.xlabel("vision assignment reduction (%)")
    plt.ylabel("benchmark delta (pp)"); plt.legend(fontsize=6)
    save_plot("03_quality_vs_assignment.png")

    quality_rows = pd.read_csv(DERIVED / "quality_rows.csv")
    chosen = quality_rows[(quality_rows.experiment == "proxy_quality_heldout32_20260911") &
                          (quality_rows.policy == "semantic_full_0.3")]
    counts = np.zeros(8)
    for value in chosen.chosen_k_counts.dropna():
        parsed = json.loads(value) if isinstance(value, str) else value
        counts += np.asarray(parsed)
    plt.bar(range(1, 9), counts); plt.xlabel("chosen k"); plt.ylabel("token-layer decisions")
    save_plot("04_chosen_k_histogram.png")

    risk = sensitivity.groupby("group").kl_per_answer_token.median()
    schedules = json.loads((RESULTS / "combined_calibration_gqa8_chartqa8_20260911/"
                            "oracle_schedules_dp.json").read_text())
    k = schedules["global_policies"]["semantic_full_0.3"]["group_k"]
    plt.scatter([risk.get(i, np.nan) for i in range(16)], k)
    plt.xscale("log"); plt.xlabel("group semantic risk (median KL)"); plt.ylabel("chosen k")
    save_plot("05_k_vs_importance.png")

    row = pd.DataFrame(read_jsonl(Path(glob.glob(str(RESULTS /
        "deepep_replay_trivial_camera_e448_l24_*.jsonl"))[0]))).iloc[0]
    before = np.asarray(row.rank_load_before)
    plt.bar(range(4), before); plt.xlabel("EP rank"); plt.ylabel("assignments")
    save_plot("06_rank_load_before.png")

    semantic_row = pd.DataFrame(read_jsonl(Path(glob.glob(str(RESULTS /
        "deepep_replay_trivial_camera_e448_l24_*.jsonl"))[0])))
    semantic_row = semantic_row[semantic_row.policy == "global_semantic_full_0.3"].iloc[0]
    plt.bar(range(4), semantic_row.rank_load_after); plt.xlabel("EP rank"); plt.ylabel("assignments")
    save_plot("07_rank_load_stage1.png")

    refined_row = pd.DataFrame(read_jsonl(Path(glob.glob(str(RESULTS /
        "deepep_replay_trivial_camera_e448_l24_*.jsonl"))[0])))
    refined_row = refined_row[refined_row.policy ==
                              "global_semantic_full_0.3_ep_refined_s0.05"].iloc[0]
    plt.bar(range(4), refined_row.rank_load_after); plt.xlabel("EP rank"); plt.ylabel("assignments")
    save_plot("08_rank_load_stage2.png")

    compare = aggregate[aggregate.policy.str.contains("global_semantic_full")]
    plt.bar(compare.policy.str.replace("global_semantic_", "", regex=False),
            compare.max_rank_reduction_pct)
    plt.xticks(rotation=30, ha="right"); plt.ylabel("max-rank reduction (%)")
    save_plot("09_same_budget_maxrank.png")

    for policy, part in replay.groupby("policy"):
        if policy == "stock" or "global_semantic_full" in policy:
            plt.scatter(part.vision_assignment_drop_pct, part.moe_reduction_pct, label=policy)
    plt.xlabel("vision assignment reduction (%)"); plt.ylabel("MoE latency reduction (%)")
    plt.legend(fontsize=6); save_plot("10_moe_latency_vs_assignment.png")

    plt.scatter(heldout.assignment_drop_pct, heldout.benchmark_delta_pp,
                c=heldout.max_rank_reduction_pct, cmap="viridis")
    plt.xlabel("vision assignment reduction (%)"); plt.ylabel("benchmark delta (pp)")
    plt.colorbar(label="max-rank reduction (%)")
    save_plot("11_ttft_quality_pareto.png")

    grids = heldout[heldout.policy.str.contains(r"semantic_(?:full|coarse)", regex=True)].copy()
    for grid, part in grids.groupby(grids.policy.str.extract(r"semantic_(full|coarse)")[0]):
        plt.plot(part.assignment_drop_pct, part.mean_kl_per_answer_token, "o", label=grid)
    plt.yscale("log"); plt.xlabel("vision assignment reduction (%)"); plt.ylabel("mean answer KL")
    plt.legend(); save_plot("12_coarse_vs_full.png")

    selectors = heldout[heldout.policy.str.contains("semantic_full|router_mass|contribution")]
    plt.scatter(selectors.assignment_drop_pct, selectors.mean_kl_per_answer_token)
    for _, value in selectors.iterrows():
        plt.annotate(value.policy, (value.assignment_drop_pct, value.mean_kl_per_answer_token),
                     fontsize=5)
    plt.yscale("log"); plt.xlabel("vision assignment reduction (%)"); plt.ylabel("mean answer KL")
    save_plot("13_selector_gap.png")

    logical = pd.read_csv(RESULTS / "capture_oracles_full_20260911/allocation_summary.csv")
    logical = logical[(logical.risk_kind == "router") &
                      (logical.policy.isin(["semantic_full", "ep_refined_s0.05"]))]
    for policy, part in logical.groupby("policy"):
        plt.plot(100 * part.target_fraction, 100 * part.max_rank_reduction_median,
                 marker="o", label=f"logical {policy}")
    actual = aggregate[aggregate.policy.str.contains("global_semantic_full")]
    plt.scatter(actual.vision_assignment_drop_pct, actual.moe_reduction_pct,
                label="actual DeepEP MoE", marker="x")
    plt.xlabel("vision assignment reduction (%)"); plt.ylabel("reduction (%)"); plt.legend(fontsize=6)
    save_plot("14_logical_vs_gpu.png")


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    quality = summarize_quality()
    replay, aggregate = summarize_replay()
    required_plots(quality, replay, aggregate)
    elapsed = 0.0
    for path in glob.glob(str(RESULTS / "*/completed_*.json")):
        elapsed += json.loads(Path(path).read_text()).get("elapsed_seconds", 0)
    replay_setup_lower_bound = 0.0
    for path in glob.glob(str(RESULTS / "deepep_replay_*.summary.json")):
        replay_setup_lower_bound += json.loads(Path(path).read_text()).get("elapsed_seconds", 0)
    gpu_log = pd.read_csv(ROOT / "GPU_TIME_LOG.csv")
    total_gpu_wall = float(gpu_log[gpu_log.experiment == "TOTAL_MEANINGFUL_LIVE"].wall_seconds.iloc[0])
    (DERIVED / "summary.json").write_text(json.dumps({
        "quality_gpu_seconds_sum": elapsed,
        "deep_ep_measured_loop_seconds_lower_bound": replay_setup_lower_bound,
        "meaningful_four_gpu_wall_seconds": total_gpu_wall,
        "meaningful_four_gpu_hours": 4 * total_gpu_wall / 3600,
        "moe_share_of_ttft_used": MOE_SHARE_OF_TTFT,
        "quality_table": str((DERIVED / "quality_summary.csv").resolve()),
        "replay_table": str((DERIVED / "replay_summary.csv").resolve()),
        "evidence_boundary": "quality times are per-GPU process sums; replay summary time excludes setup",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
