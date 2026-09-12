#!/usr/bin/env python3
"""Compute interaction-aware layer/MoE necessity oracles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_NFE = {"gsm8k": 66, "humaneval": 86}
cost = pd.read_csv(ROOT / "LOW_OVERHEAD_LAYER_PHASE_COST.csv")
stability = pd.read_csv(ROOT / "LAYER_PHASE_STABILITY.csv")
cost = cost.merge(
    stability[["dataset", "layer", "phase", "live_ratio_median"]],
    on=["dataset", "layer", "phase"],
    how="left",
)
cost["shared_request_e2e_upper_percent"] = (
    100.0 * cost["shared_sum_ms"] / cost["clean_request_median_ms"]
)


def parse_layers(value) -> list[int]:
    text = str(value)
    return [int(float(item)) for item in text.split(",") if item and item != "nan"]


def policy_cost(dataset: str, mode: str, phase: str, layers: list[int]):
    rows = cost[
        (cost["dataset"] == dataset)
        & (cost["phase"] == phase)
        & (cost["layer"].isin(layers))
    ]
    if mode == "layer_bypass":
        field = "whole_layer_normalized_optimistic_e2e_percent"
    elif mode in {"routed_bypass", "stale_routed"}:
        field = "routed_moe_request_e2e_upper_percent"
    elif mode == "routed_only":
        field = "shared_request_e2e_upper_percent"
    else:
        raise ValueError(mode)
    direct = float(rows[field].sum())
    post_epoch = float((rows[field] * rows["live_ratio_median"]).sum())
    return direct, post_epoch


def annotate(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = frame[frame["mode"] != "none"].copy()
    frame["dataset"] = dataset
    values = frame.apply(
        lambda row: policy_cost(
            dataset, row["mode"], row["phase"], parse_layers(row["layers"])
        ),
        axis=1,
    )
    frame["direct_e2e_optimistic_percent"] = [value[0] for value in values]
    frame["post_epoch_live_weighted_percent"] = [value[1] for value in values]
    frame["current_step_exact"] = (
        frame["live_top1_flip_fraction"].fillna(1.0).abs() < 1e-12
    ) & (frame["accepted_set_change_fraction"].fillna(1.0).abs() < 1e-12)
    frame["benchmark_vector_safe"] = (
        (frame["per_sample_score_flips"] == 0)
        & (frame["benchmark_delta_vs_baseline"] == 0)
    )
    frame["final_trajectory_exact"] = (
        (frame["final_sequence_exact"] == 32)
        & (frame["per_sample_score_flips"] == 0)
        & (frame["nfe"] == BASELINE_NFE[dataset])
    )
    return frame


single_frames = []
for dataset, path, modes in (
    ("gsm8k", "CAUSAL_ROUTED_BYPASS_GSM8K.csv", {"routed_bypass"}),
    ("humaneval", "CAUSAL_ROUTED_BYPASS_HUMANEVAL.csv", {"routed_bypass"}),
    ("gsm8k", "CAUSAL_STALE_ROUTED_GSM8K.csv", {"stale_routed"}),
    ("gsm8k", "CAUSAL_DEEP_DIAGNOSTICS_GSM8K.csv", {"layer_bypass", "routed_only"}),
    ("humaneval", "CAUSAL_VALIDATION_HUMANEVAL.csv", {"layer_bypass", "routed_only"}),
):
    frame = pd.read_csv(ROOT / path)
    frame = frame[frame["mode"].isin(modes)]
    frame = frame[~frame["layers"].fillna("").astype(str).str.contains(",")]
    single_frames.append(annotate(frame, dataset))
single = pd.concat(single_frames, ignore_index=True).drop_duplicates(
    ["dataset", "mode", "phase", "layers"], keep="last"
)
single.to_csv(ROOT / "SINGLE_LAYER_ORACLE.csv", index=False)


def multi_rows(filename: str, dataset: str, require_comma: bool) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / filename)
    if require_comma:
        frame = frame[frame["layers"].fillna("").astype(str).str.contains(",")]
    return annotate(frame, dataset)


groups = pd.concat(
    [
        multi_rows("CAUSAL_DEEP_DIAGNOSTICS_GSM8K.csv", "gsm8k", True),
        multi_rows("CAUSAL_VALIDATION_HUMANEVAL.csv", "humaneval", True),
    ],
    ignore_index=True,
)
groups.to_csv(ROOT / "GROUP_ORACLE.csv", index=False)

greedy = pd.concat(
    [
        multi_rows("CAUSAL_GREEDY_GSM8K.csv", "gsm8k", False),
        multi_rows("CAUSAL_GREEDY_HUMANEVAL.csv", "humaneval", False),
    ],
    ignore_index=True,
)
greedy.to_csv(ROOT / "MULTI_LAYER_GREEDY_ORACLE.csv", index=False)

summary = []
for (dataset, mode), rows in single.groupby(["dataset", "mode"]):
    for criterion in (
        "current_step_exact",
        "benchmark_vector_safe",
        "final_trajectory_exact",
    ):
        safe = rows[rows[criterion]]
        summary.append(
            {
                "oracle": "O1_single_layer_independent",
                "dataset": dataset,
                "mode": mode,
                "phase": "all",
                "criterion": criterion,
                "tested_policies": len(rows),
                "safe_policies": len(safe),
                "optimistic_e2e_percent": safe["direct_e2e_optimistic_percent"].sum(),
                "post_epoch_live_weighted_percent": safe[
                    "post_epoch_live_weighted_percent"
                ].sum(),
                "evidence_boundary": "interaction-unsafe sum of disjoint cells",
            }
        )

for oracle, frame in (("O2_greedy_multi_layer", greedy), ("O3_contiguous_group", groups)):
    for (dataset, mode, phase), rows in frame.groupby(["dataset", "mode", "phase"]):
        for criterion in ("benchmark_vector_safe", "final_trajectory_exact"):
            safe = rows[rows[criterion]]
            best = safe.sort_values("direct_e2e_optimistic_percent").tail(1)
            summary.append(
                {
                    "oracle": oracle,
                    "dataset": dataset,
                    "mode": mode,
                    "phase": phase,
                    "criterion": criterion,
                    "tested_policies": len(rows),
                    "safe_policies": len(safe),
                    "optimistic_e2e_percent": 0.0
                    if best.empty
                    else best["direct_e2e_optimistic_percent"].iloc[0],
                    "post_epoch_live_weighted_percent": 0.0
                    if best.empty
                    else best["post_epoch_live_weighted_percent"].iloc[0],
                    "evidence_boundary": "largest actually tested within-phase joint policy",
                }
            )

pd.DataFrame(summary).to_csv(ROOT / "FINAL_TRAJECTORY_ORACLE.csv", index=False)

cross = []
for oracle, frame in (("O2_greedy_multi_layer", greedy), ("O3_contiguous_group", groups)):
    keys = ["policy", "mode", "phase", "layers"]
    gsm = frame[frame["dataset"] == "gsm8k"]
    human = frame[frame["dataset"] == "humaneval"]
    joined = gsm.merge(human, on=keys, suffixes=("_gsm8k", "_humaneval"))
    for criterion in ("benchmark_vector_safe", "final_trajectory_exact"):
        safe = joined[
            joined[f"{criterion}_gsm8k"] & joined[f"{criterion}_humaneval"]
        ]
        for _, row in safe.iterrows():
            cross.append(
                {
                    "oracle": oracle,
                    "criterion": criterion,
                    "policy": row["policy"],
                    "mode": row["mode"],
                    "phase": row["phase"],
                    "layers": row["layers"],
                    "gsm8k_optimistic_e2e_percent": row[
                        "direct_e2e_optimistic_percent_gsm8k"
                    ],
                    "humaneval_optimistic_e2e_percent": row[
                        "direct_e2e_optimistic_percent_humaneval"
                    ],
                    "median_optimistic_e2e_percent": 0.5
                    * (
                        row["direct_e2e_optimistic_percent_gsm8k"]
                        + row["direct_e2e_optimistic_percent_humaneval"]
                    ),
                    "gsm8k_post_epoch_percent": row[
                        "post_epoch_live_weighted_percent_gsm8k"
                    ],
                    "humaneval_post_epoch_percent": row[
                        "post_epoch_live_weighted_percent_humaneval"
                    ],
                    "gsm8k_final_sequence_exact": row["final_sequence_exact_gsm8k"],
                    "humaneval_final_sequence_exact": row[
                        "final_sequence_exact_humaneval"
                    ],
                }
            )

cross_frame = pd.DataFrame(cross)
cross_frame.to_csv(ROOT / "CROSS_TASK_ORACLE.csv", index=False)
print(
    json.dumps(
        {
            "single_rows": len(single),
            "group_rows": len(groups),
            "greedy_rows": len(greedy),
            "cross_task_safe_rows": len(cross_frame),
        }
    )
)
