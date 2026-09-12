#!/usr/bin/env python3
"""Build conservative controls and decision tables from discovery artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_concatenated_json(path: Path) -> list[dict]:
    text = path.read_text()
    decoder = json.JSONDecoder()
    rows = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        row, position = decoder.raw_decode(text, position)
        rows.append(row)
    return rows


def quality_reproducibility() -> list[dict]:
    rows = []
    for dataset in ("gsm8k", "humaneval"):
        for repeat in (1, 2, 3):
            prediction = next((ROOT / f"results/clean/{dataset}/r{repeat}").glob("*.jsonl"))
            records = read_concatenated_json(prediction)
            answer_blob = "\n".join(record["answer"] for record in records).encode()
            quality = json.loads((ROOT / f"results/quality/{dataset}_r{repeat}.json").read_text())
            rows.append({
                "dataset": dataset,
                "repeat": repeat,
                "requests": len(records),
                "answer_sha256": hashlib.sha256(answer_blob).hexdigest(),
                "correct": quality["correct"],
                "total": quality["total"],
                "accuracy": quality["accuracy"],
                "note": "exact deterministic output across clean restarts; no method changes model semantics",
            })
    return rows


def fixed_m_controls(layers: pd.DataFrame) -> list[dict]:
    rows = []
    metrics = (
        "live_ratio", "full_active_experts", "full_rows_p50", "full_tiny_le4",
        "expert_critical_ms", "moe_critical_ms", "rank_load_cv", "rank_fanout_mean",
    )
    for dataset in sorted(layers.dataset.unique()):
        subset = layers[(layers.dataset == dataset) & (layers.physical_rows == 1024)]
        wave_liveness = subset.groupby("wave").live_ratio.first().sort_values()
        if len(wave_liveness) < 2:
            continue
        selected = (("lowest_live", int(wave_liveness.index[0])), ("highest_live", int(wave_liveness.index[-1])))
        summaries = {}
        for label, wave in selected:
            point = subset[subset.wave == wave]
            out = {
                "dataset": dataset,
                "physical_rows": 1024,
                "stratum": label,
                "wave": wave,
                "layers": len(point),
            }
            for metric in metrics:
                out[f"{metric}_median"] = float(point[metric].median())
            rows.append(out)
            summaries[label] = out
        low, high = summaries["lowest_live"], summaries["highest_live"]
        rows.append({
            "dataset": dataset,
            "physical_rows": 1024,
            "stratum": "low_minus_high_percent",
            "wave": "",
            "layers": 31,
            **{
                f"{metric}_median": 100 * (low[f"{metric}_median"] - high[f"{metric}_median"]) / max(abs(high[f"{metric}_median"]), 1e-12)
                for metric in metrics
            },
        })
    return rows


def component_upper_bounds(layers: pd.DataFrame, clean: pd.DataFrame) -> list[dict]:
    clean_median = clean.groupby("dataset").wall_seconds.median().to_dict()
    stages = ("router", "dispatch", "expert", "combine", "shared", "gather", "moe")
    rows = []
    for dataset in sorted(layers.dataset.unique()):
        point = layers[layers.dataset == dataset]
        wall_ms = 1000 * clean_median[dataset]
        for stage in stages:
            total = float(point[f"{stage}_critical_ms"].sum())
            rows.append({
                "dataset": dataset,
                "component": stage,
                "summed_critical_rank_cuda_event_ms": total,
                "clean_e2e_upper_bound_percent": 100 * total / wall_ms,
                "evidence_boundary": "observer-heavy component attribution divided by clean E2E; not jointly removable",
            })
    return rows


def candidate_table(oracles: pd.DataFrame) -> list[dict]:
    by_candidate = defaultdict(dict)
    for row in oracles.to_dict("records"):
        by_candidate[row["candidate"]][row["dataset"]] = float(row["clean_e2e_percent"])

    def values(candidate: str) -> tuple[float, float]:
        return by_candidate[candidate]["gsm8k"], by_candidate[candidate]["humaneval"]

    current = values("perfect_current_fragmentation_removal_p99_trimmed")
    compaction = values("hypothetical_live_row_compaction_expert_only_p99_trimmed")
    residual = values("post_compaction_fragmentation_residual_p99_trimmed")
    feasible = values("feasible_tiny_shape_specialization_50pct_capture_p99_trimmed")
    return [
        {
            "candidate": "F1_cross_wave_expert_major_coalescing",
            "measured_signal": "shape beats pair count as expert-latency predictor",
            "gsm8k_perfect_e2e_percent": current[0],
            "humaneval_perfect_e2e_percent": current[1],
            "feasible_e2e_percent": "< perfect bound; waiting and packing not included",
            "quality_risk": "none if exact",
            "prior_art_risk": "high: expert-major batching/grouped GEMM/queues",
            "decision": "KILL_<5%",
        },
        {
            "candidate": "F2_tiny_expert_specialized_execution",
            "measured_signal": "post-compaction <=4-row fraction reaches 68.5%/79.4% late",
            "gsm8k_perfect_e2e_percent": residual[0],
            "humaneval_perfect_e2e_percent": residual[1],
            "feasible_e2e_percent": f"{feasible[0]:.3f}/{feasible[1]:.3f} at assumed 50% capture",
            "quality_risk": "none if numerically exact",
            "prior_art_risk": "high: fused/grouped/tiny-GEMM kernel work",
            "decision": "KILL_<5%",
        },
        {
            "candidate": "F3_fragmentation_aware_wave_composition",
            "measured_signal": "same-work latency spread exists but large cases are runtime tails",
            "gsm8k_perfect_e2e_percent": current[0],
            "humaneval_perfect_e2e_percent": current[1],
            "feasible_e2e_percent": "strictly below perfect; queueing excluded",
            "quality_risk": "none if exact",
            "prior_art_risk": "high; prior RAWS ready-set oracle only 0.650%",
            "decision": "KILL_<5%",
        },
        {
            "candidate": "P1_shape_aware_cost_model",
            "measured_signal": "p99-trim shape model reduces expert RMSE by 45.2%",
            "gsm8k_perfect_e2e_percent": "no independent action oracle",
            "humaneval_perfect_e2e_percent": "no independent action oracle",
            "feasible_e2e_percent": "not established",
            "quality_risk": "none",
            "prior_art_risk": "medium/high: standard MoE cost modeling",
            "decision": "CHARACTERIZATION_ONLY",
        },
        {
            "candidate": "R1_route_plan_delta_reuse",
            "measured_signal": "rank-load cosine ~0.999 while exact top-k set is low",
            "gsm8k_perfect_e2e_percent": "router upper bound 6.43",
            "humaneval_perfect_e2e_percent": "router upper bound 7.22",
            "feasible_e2e_percent": "lower; exact route and hidden state volatile",
            "quality_risk": "high for route/output reuse",
            "prior_art_risk": "high: Epoch recomputes gate values; TEAM temporal routing",
            "decision": "KILL_NO_EXACT_REUSE",
        },
        {
            "candidate": "C1_confidence_conditioned_expert_budget",
            "measured_signal": "hypothesized confidence/router paradox absent",
            "gsm8k_perfect_e2e_percent": "not computed after causal premise failed",
            "humaneval_perfect_e2e_percent": "not computed after causal premise failed",
            "feasible_e2e_percent": "not applicable",
            "quality_risk": "high/approximate",
            "prior_art_risk": "directly adjacent to REFLEX/TEAM/DES",
            "decision": "KILL_PREMISE_AND_PRIOR_ART",
        },
        {
            "candidate": "Epoch_like_live_row_compaction_context_only",
            "measured_signal": "liveness collapse",
            "gsm8k_perfect_e2e_percent": compaction[0],
            "humaneval_perfect_e2e_percent": compaction[1],
            "feasible_e2e_percent": "not measured; expert-only sensitivity",
            "quality_risk": "depends on liveness/refresh contract",
            "prior_art_risk": "direct collision with Epoch",
            "decision": "EXCLUDED_NOT_NEW",
        },
    ]


def figures(layers: pd.DataFrame, oracles: pd.DataFrame) -> None:
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for dataset, marker in (("gsm8k", "o"), ("humaneval", "s")):
        point = layers[layers.dataset == dataset]
        ax.scatter(point.token_confidence_live_mean, point.router_entropy_live_mean, s=5, alpha=.18, marker=marker, label=dataset)
    ax.set_xlabel("live-token confidence")
    ax.set_ylabel("router entropy")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "05b_router_entropy_vs_confidence.png", dpi=180)
    plt.close(fig)

    names = (
        "perfect_current_fragmentation_removal_p99_trimmed",
        "post_compaction_fragmentation_residual_p99_trimmed",
    )
    labels = ("current\nfragmentation", "post-compaction\nresidual")
    x = np.arange(len(names))
    width = .36
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for offset, dataset in ((-.18, "gsm8k"), (.18, "humaneval")):
        values = [float(oracles[(oracles.dataset == dataset) & (oracles.candidate == name)].clean_e2e_percent.iloc[0]) for name in names]
        ax.bar(x + offset, values, width, label=dataset)
    ax.axhline(8, color="tab:red", linestyle="--", linewidth=1, label="implementation gate")
    ax.set_xticks(x, labels)
    ax.set_ylabel("clean request E2E oracle (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "13_fragmentation_removal_oracle.png", dpi=180)
    plt.close(fig)


def main() -> None:
    layers = pd.read_csv(ROOT / "LAYER_WAVE_METRICS.csv")
    clean = pd.read_csv(ROOT / "BASELINE_CLEAN.csv")
    oracles = pd.read_csv(ROOT / "CANDIDATE_ORACLES.csv")
    write_csv(ROOT / "QUALITY_REPRODUCIBILITY.csv", quality_reproducibility())
    write_csv(ROOT / "FIXED_M_CONTROLS.csv", fixed_m_controls(layers))
    write_csv(ROOT / "COMPONENT_E2E_UPPER_BOUNDS.csv", component_upper_bounds(layers, clean))
    write_csv(ROOT / "METHOD_CANDIDATES.csv", candidate_table(oracles))
    figures(layers, oracles)


if __name__ == "__main__":
    main()
