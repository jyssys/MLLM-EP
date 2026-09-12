#!/usr/bin/env python3
"""Join causal sensitivity, representation stability, and low-overhead cost."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASES = ["early", "middle", "late"]


def load_single(path: str, dataset: str, modes: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / path)
    frame = frame[frame["mode"].isin(modes)].copy()
    frame = frame[~frame["layers"].fillna("").astype(str).str.contains(",")]
    frame["dataset"] = dataset
    frame["layer"] = frame["layers"].astype(int)
    return frame


frames = [
    load_single("CAUSAL_ROUTED_BYPASS_GSM8K.csv", "gsm8k", {"routed_bypass"}),
    load_single("CAUSAL_ROUTED_BYPASS_HUMANEVAL.csv", "humaneval", {"routed_bypass"}),
    load_single(
        "CAUSAL_DEEP_DIAGNOSTICS_GSM8K.csv",
        "gsm8k",
        {"layer_bypass", "routed_only"},
    ),
    load_single(
        "CAUSAL_VALIDATION_HUMANEVAL.csv",
        "humaneval",
        {"layer_bypass", "routed_only"},
    ),
    load_single("CAUSAL_STALE_ROUTED_GSM8K.csv", "gsm8k", {"stale_routed"}),
]
causal = pd.concat(frames, ignore_index=True)
causal = causal.drop_duplicates(["dataset", "mode", "layer", "phase"], keep="last")

baseline_nfe = {"gsm8k": 66.0, "humaneval": 86.0}
causal["top1_flip"] = causal["live_top1_flip_fraction"].fillna(1.0)
causal["accepted_change"] = causal["accepted_set_change_fraction"].fillna(1.0)
causal["final_sequence_divergence"] = 1.0 - causal["final_sequence_exact"] / 32.0
causal["benchmark_score_flip_fraction"] = causal["per_sample_score_flips"] / 32.0
causal["nfe_relative_change"] = causal.apply(
    lambda row: abs(float(row["nfe"]) - baseline_nfe[row["dataset"]])
    / baseline_nfe[row["dataset"]],
    axis=1,
)
causal["decision_sensitivity_score"] = causal[
    [
        "top1_flip",
        "accepted_change",
        "final_sequence_divergence",
        "benchmark_score_flip_fraction",
        "nfe_relative_change",
    ]
].mean(axis=1)

cost = pd.read_csv(ROOT / "LOW_OVERHEAD_LAYER_PHASE_COST.csv")
cost_columns = [
    "dataset",
    "layer",
    "phase",
    "clean_request_median_ms",
    "whole_layer_normalized_optimistic_e2e_percent",
    "routed_moe_request_e2e_upper_percent",
    "shared_sum_ms",
]
causal = causal.merge(cost[cost_columns], on=["dataset", "layer", "phase"], how="left")
causal["shared_request_e2e_upper_percent"] = (
    100.0 * causal["shared_sum_ms"] / causal["clean_request_median_ms"]
)
causal["physical_cost_e2e_upper_percent"] = np.select(
    [
        causal["mode"] == "layer_bypass",
        causal["mode"].isin(["routed_bypass", "stale_routed"]),
        causal["mode"] == "routed_only",
    ],
    [
        causal["whole_layer_normalized_optimistic_e2e_percent"],
        causal["routed_moe_request_e2e_upper_percent"],
        causal["shared_request_e2e_upper_percent"],
    ],
    default=np.nan,
)
stability = pd.read_csv(ROOT / "LAYER_PHASE_STABILITY.csv")
stability = stability[
    [
        "dataset",
        "layer",
        "phase",
        "live_ratio_median",
        "relative_l2_update_median",
        "output_cosine_median",
        "attention_update_norm_median",
        "moe_update_norm_median",
        "routed_output_norm_median",
        "shared_output_norm_median",
    ]
]
causal = causal.merge(stability, on=["dataset", "layer", "phase"], how="left")
causal["post_epoch_live_weighted_cost_percent"] = (
    causal["physical_cost_e2e_upper_percent"] * causal["live_ratio_median"]
)
causal["cost_per_decision_sensitivity"] = causal[
    "physical_cost_e2e_upper_percent"
] / causal["decision_sensitivity_score"].clip(lower=1e-6)
causal.to_csv(ROOT / "DECISION_SENSITIVITY_COST.csv", index=False)

figures = ROOT / "figures"
figures.mkdir(exist_ok=True)


def heatmaps(frame: pd.DataFrame, field: str, title: str, filename: str) -> None:
    datasets = sorted(frame["dataset"].unique())
    figure, axes = plt.subplots(
        1, len(datasets), figsize=(5.8 * len(datasets), 7), squeeze=False
    )
    for axis, dataset in zip(axes[0], datasets):
        matrix = np.full((32, 3), np.nan)
        for _, row in frame[frame["dataset"] == dataset].iterrows():
            matrix[int(row["layer"]), PHASES.index(row["phase"])] = float(row[field])
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
        axis.set_title(dataset)
        axis.set_xticks(range(3), PHASES)
        axis.set_xlabel("refinement phase")
        axis.set_ylabel("layer")
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(figures / filename, dpi=180)
    plt.close(figure)


routed = causal[causal["mode"] == "routed_bypass"].copy()
heatmaps(
    routed,
    "decision_sensitivity_score",
    "Causal routed-MoE decision sensitivity",
    "08_layer_phase_decision_sensitivity_map.png",
)
heatmaps(
    routed,
    "top1_flip",
    "MoE-only bypass: live-token top-1 flip fraction",
    "10_moe_only_sensitivity_map.png",
)

figure, axis = plt.subplots(figsize=(8, 6))
for dataset, marker in (("gsm8k", "o"), ("humaneval", "s")):
    rows = routed[routed["dataset"] == dataset]
    axis.scatter(
        rows["physical_cost_e2e_upper_percent"],
        rows["decision_sensitivity_score"],
        label=dataset,
        marker=marker,
        alpha=0.75,
    )
axis.set_xlabel("routed-MoE clean request E2E upper cost (%)")
axis.set_ylabel("causal decision sensitivity score")
axis.set_title("Decision sensitivity versus physical latency cost")
axis.grid(alpha=0.25)
axis.legend()
figure.tight_layout()
figure.savefig(figures / "09_decision_sensitivity_vs_latency_scatter.png", dpi=180)
plt.close(figure)

shared = causal[(causal["dataset"] == "gsm8k") & (causal["mode"] == "routed_only")]
comparison = routed[routed["dataset"] == "gsm8k"].merge(
    shared,
    on=["dataset", "layer", "phase"],
    suffixes=("_routed_removed", "_shared_removed"),
)
figure, axis = plt.subplots(figsize=(7, 6))
for phase, marker in zip(PHASES, ("o", "s", "^")):
    rows = comparison[comparison["phase"] == phase]
    axis.scatter(
        rows["decision_sensitivity_score_routed_removed"],
        rows["decision_sensitivity_score_shared_removed"],
        label=phase,
        marker=marker,
        s=55,
    )
axis.plot([0, 0.5], [0, 0.5], linestyle="--", color="black", linewidth=1)
axis.set_xlabel("sensitivity when routed contribution is removed")
axis.set_ylabel("sensitivity when shared contribution is removed")
axis.set_title("Shared versus routed expert decision contribution")
axis.grid(alpha=0.25)
axis.legend()
figure.tight_layout()
figure.savefig(figures / "11_shared_vs_routed_contribution_map.png", dpi=180)
plt.close(figure)

correlations = []
features = [
    "relative_l2_update_median",
    "output_cosine_median",
    "attention_update_norm_median",
    "moe_update_norm_median",
]
for dataset, rows in routed.groupby("dataset"):
    for feature in features:
        correlations.append(
            {
                "dataset": dataset,
                "feature": feature,
                "spearman_with_causal_sensitivity": rows[feature].corr(
                    rows["decision_sensitivity_score"], method="spearman"
                ),
                "pearson_with_causal_sensitivity": rows[feature].corr(
                    rows["decision_sensitivity_score"], method="pearson"
                ),
                "cells": len(rows),
            }
        )
with (ROOT / "STABILITY_DECISION_CORRELATION.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(correlations[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(correlations)

print(
    json.dumps(
        {
            "causal_cells": len(causal),
            "routed_cells": len(routed),
            "shared_comparisons": len(comparison),
        }
    )
)
