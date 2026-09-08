"""CPU-only descriptive figures and a bounded MoDES monotonicity diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--task", type=Path, required=True)
    args = ap.parse_args()
    out = args.results / "analysis/cpu_release_checkpoint"
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    quality = pd.read_csv(args.task / "QUALITY_EFFICIENCY_PARETO.csv")
    selected = quality[(quality.screen == "sere_chart128_phase") & (quality.policy != "vanilla")]
    selected = pd.concat([selected, quality[(quality.screen == "sere_chart128_b16_partial") &
                                           (quality.policy == "sere_s2_r05")]])
    fig, ax = plt.subplots(figsize=(8, 4))
    y = selected.paired_quality_delta_pp.to_numpy()
    ax.bar(np.arange(len(y)), y, color=["#c9605b", "#c9605b", "#7c9eaf", "#7c9eaf"])
    ax.errorbar(np.arange(len(y)), y, yerr=[y-selected.quality_ci_low_pp.to_numpy(),
                selected.quality_ci_high_pp.to_numpy()-y], fmt="none", color="black", capsize=4)
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(range(4), ["B1 all", "B1 decode", "B1 prefill", "B16 all"])
    ax.set_ylabel("Paired ChartQA accuracy difference (pp)")
    ax.set_title("Exploratory SERE: phase and batch controls\n128 images; official-norm confirmatory quality is pending")
    fig.tight_layout();fig.savefig(plots / "sere_exploratory_phase_batch.png", dpi=150);plt.close(fig)

    plans = pd.read_csv(out / "libra_matched_modality_plans.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(100*plans.topk_recall_vision_minus_text,
               100*plans.local_fraction_delta_vision_minus_text, s=10, alpha=.3)
    ax.axhline(0, color="gray", lw=.8);ax.axvline(0, color="gray", lw=.8)
    ax.set_xlabel("Vision - Text prediction recall (pp)")
    ax.set_ylabel("Vision - Text oracle locality gain (pp)")
    ax.set_title("Matched real-route CPU planner diagnostic\nNot a GPU latency or E2E oracle")
    fig.tight_layout();fig.savefig(plots / "libra_matched_recall_vs_plan.png", dpi=150);plt.close(fig)

    points = pd.read_csv(out / "modes_measured_points.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    c = ax.scatter(100*points.skip_fraction, points.kl, c=np.log10(points.tau_text), s=14, alpha=.7)
    ax.set_yscale("log")
    ax.set_xlabel("Calibration assignment skip fraction (%)")
    ax.set_ylabel("Official answer-position KL")
    ax.set_title("MoDES measured calibration points (search interrupted)\nNo held-out quality or serving-speed claim")
    fig.colorbar(c, ax=ax, label="log10 text threshold")
    fig.tight_layout();fig.savefig(plots / "modes_partial_calibration.png", dpi=150);plt.close(fig)

    # Test only actually observed adjacent points at a fixed other threshold.
    # Missing grid points are not filled by interpolation or invented GPU runs.
    comparisons = []
    for fixed, varying in (("tau_text", "tau_vision"), ("tau_vision", "tau_text")):
        for value, group in points.groupby(fixed):
            group = group.sort_values(varying)
            records = group.to_dict("records")
            for lo, hi in zip(records, records[1:]):
                comparisons.append({"fixed_threshold": fixed, "fixed_value": value,
                    "increased_threshold": varying, "lower_value": lo[varying], "higher_value": hi[varying],
                    "skip_change_pp": 100*(hi["skip_fraction"]-lo["skip_fraction"]),
                    "kl_change": hi["kl"]-lo["kl"],
                    "scope": "OBSERVED_GRID_POINTS_ONLY_CHANGED_DOWNSTREAM_ROUTES_ALLOWED"})
    comp = pd.DataFrame(comparisons)
    comp.to_csv(out / "modes_observed_monotonicity.csv", index=False)
    summary = {"compared_pairs": len(comp),
        "any_skip_decrease": int((comp.skip_change_pp < 0).sum()),
        "decrease_ge_0_1pp": int((comp.skip_change_pp <= -.1).sum()),
        "decrease_ge_1pp": int((comp.skip_change_pp <= -1).sum()),
        "largest_skip_decrease_pp": max(0, -float(comp.skip_change_pp.min())),
        "interpretation": "Small deterministic downstream-route changes are not themselves material search failure; no unobserved grid optimum is inferred."}
    (out / "modes_observed_monotonicity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
