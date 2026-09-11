#!/usr/bin/env python3
"""Aggregate TEAM positive-control/EP2 measurements and render report figures.

All request-level speedups come from clean runs.  CUDA-event traces are used
only for stage attribution.  Candidate values are explicitly perfect-oracle
upper bounds unless the row is marked ``measured``.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "team_positive_20260911_211456"
RAW = RESULT / "raw"
ANALYSIS = RESULT / "analysis"
PLOTS = RESULT / "plots"


def load(name: str) -> dict:
    return json.loads((RAW / name).read_text())


def mean_latency(payload: dict) -> float:
    return statistics.mean(row["elapsed_s"] for row in payload["records"])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_bar(path: Path, labels, values, ylabel: str, title: str, colors=None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(labels, values, color=colors or ["#4c78a8"] * len(values))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def stage_sums(payload: dict) -> dict[str, float]:
    keys = ["router_ms", "prepare_ms", "dispatch_ms", "expert_ms", "combine_ms", "moe_ms"]
    return {key: sum(float(row.get(key, 0.0)) for row in payload["stage_rows"]) for key in keys}


def work_sums(payload: dict) -> dict[str, int]:
    keys = [
        "physical_rows", "computed_rows", "assignments", "local_assignments",
        "remote_assignments_from_source", "dispatch_bytes_hidden",
        "combine_bytes_hidden", "active_local_experts",
    ]
    return {key: sum(int(row.get(key, 0)) for row in payload["stage_rows"]) for key in keys}


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    # Stage A: three independent clean restart pairs.
    stage_a = []
    stage_a_speedups = []
    for restart in (1, 2, 3):
        baseline = load(f"stageA_clean_baseline_r{restart}.json")
        team = load(f"stageA_clean_team_r{restart}.json")
        b = mean_latency(baseline)
        t = mean_latency(team)
        speedup = b / t
        stage_a_speedups.append(speedup)
        stage_a.append({
            "restart": restart,
            "requests": len(baseline["records"]),
            "baseline_mean_s": b,
            "team_mean_s": t,
            "speedup_x": speedup,
        })
    write_csv(ANALYSIS / "stage_a_runs.csv", stage_a)

    r1_b = load("stageA_clean_baseline_r1.json")
    r1_t = load("stageA_clean_team_r1.json")
    task_rows = []
    for task in ("gsm8k", "humaneval"):
        b = statistics.mean(x["elapsed_s"] for x in r1_b["records"] if x["task"] == task)
        t = statistics.mean(x["elapsed_s"] for x in r1_t["records"] if x["task"] == task)
        task_rows.append({"task": task, "baseline_mean_s": b, "team_mean_s": t, "speedup_x": b / t})
    write_csv(ANALYSIS / "stage_a_task_speedup.csv", task_rows)

    trace_a_b = load("stageA_trace_baseline.json")
    trace_a_t = load("stageA_trace_team.json")
    stage_a_work = []
    for payload in (trace_a_b, trace_a_t):
        work = payload["trace_summary"]
        stage_a_work.append({"mode": payload["mode"], **work})
    write_csv(ANALYSIS / "stage_a_work.csv", stage_a_work)

    save_bar(
        PLOTS / "01_stage_a_clean_latency.png",
        ["Baseline", "TEAM"],
        [mean_latency(r1_b), mean_latency(r1_t)],
        "Mean request latency (s)",
        "Stage A clean bounded subset",
        ["#9ca3af", "#2563eb"],
    )
    save_bar(
        PLOTS / "02_stage_a_speedup_by_task.png",
        [x["task"] for x in task_rows] + ["median restart"],
        [x["speedup_x"] for x in task_rows] + [statistics.median(stage_a_speedups)],
        "Speedup (baseline / TEAM)",
        "TEAM clean speedup",
        ["#4c78a8", "#59a14f", "#f28e2b"],
    )

    work_metrics = ["model_forwards", "moe_layer_calls", "token_expert_pairs", "mean_active_experts_per_layer_call"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))
    for ax, key in zip(axes.ravel(), work_metrics):
        values = [trace_a_b["trace_summary"][key], trace_a_t["trace_summary"][key]]
        ax.bar(["Baseline", "TEAM"], values, color=["#9ca3af", "#2563eb"])
        ax.set_title(key.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Structural trace: work direction (observer-heavy)")
    fig.tight_layout()
    fig.savefig(PLOTS / "03_stage_a_work_reduction.png", dpi=180)
    plt.close(fig)

    quality = [
        {"setting": "128-token GSM8K", "baseline": 3 / 4, "team": 3 / 4},
        {"setting": "128-token HumanEval", "baseline": 2 / 4, "team": 1 / 4},
        {"setting": "256-token boundary pair", "baseline": 2 / 2, "team": 2 / 2},
    ]
    write_csv(ANALYSIS / "stage_a_quality.csv", quality)
    x = np.arange(len(quality))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - 0.18, [q["baseline"] * 100 for q in quality], 0.36, label="Baseline", color="#9ca3af")
    ax.bar(x + 0.18, [q["team"] * 100 for q in quality], 0.36, label="TEAM", color="#2563eb")
    ax.set_xticks(x, [q["setting"] for q in quality], rotation=10, ha="right")
    ax.set_ylabel("Bounded score (%)")
    ax.set_title("Quality sanity; 128-token HumanEval delta is truncation-sensitive")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS / "04_stage_a_quality.png", dpi=180)
    plt.close(fig)

    # Stage B: exact Python-reference true EP2, then fused trivial-fix control.
    stage_b = []
    ep2_speedups = []
    for restart in (2, 3, 4):
        prefix = "stageB_warm" if restart == 4 else "stageB_clean"
        baseline = load(f"{prefix}_baseline_ep2_r{restart}_rank0.json")
        team = load(f"{prefix}_team_ep2_r{restart}_rank0.json")
        b = mean_latency(baseline)
        t = mean_latency(team)
        speedup = b / t
        ep2_speedups.append(speedup)
        stage_b.append({
            "restart": restart,
            "warmup": int(restart == 4),
            "baseline_s": b,
            "team_s": t,
            "speedup_x": speedup,
            "backend": "python_reference",
        })
    fused_b = load("stageC_clean_baseline_ep2_fused_rank0.json")
    fused_t = load("stageC_clean_team_ep2_fused_rank0.json")
    fused_speedup = mean_latency(fused_b) / mean_latency(fused_t)
    stage_b.append({
        "restart": "fused_control",
        "warmup": 1,
        "baseline_s": mean_latency(fused_b),
        "team_s": mean_latency(fused_t),
        "speedup_x": fused_speedup,
        "backend": "vllm_fused",
    })
    write_csv(ANALYSIS / "stage_b_runs.csv", stage_b)

    layer_validation_rows = []
    for backend, name in (
        ("python_reference", "stageB_layer_validation_python_rank0.json"),
        ("vllm_fused", "stageC_layer_validation_fused_rank0.json"),
    ):
        validation = load(name)["layer_validation"]
        layer_validation_rows.append({"backend": backend, **validation})
    write_csv(ANALYSIS / "layer_correctness.csv", layer_validation_rows)
    save_bar(
        PLOTS / "05_stage_b_speedup_substrate.png",
        ["Single GPU\nmedian", "EP2 reference\nmedian", "EP2 fused\ncontrol"],
        [statistics.median(stage_a_speedups), statistics.median(ep2_speedups), fused_speedup],
        "TEAM speedup (x)",
        "TEAM benefit survives EP2; substrate removes much of the gap",
        ["#4c78a8", "#59a14f", "#f28e2b"],
    )

    fig, ax = plt.subplots(figsize=(9, 1.8))
    ownership = np.array([[0] * 64 + [1] * 64])
    ax.imshow(ownership, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks([0, 31, 63, 64, 95, 127])
    ax.set_xlabel("Global expert id")
    ax.set_title("True EP2 ownership: rank 0 = experts 0–63; rank 1 = 64–127")
    fig.tight_layout()
    fig.savefig(PLOTS / "06_ep2_ownership.png", dpi=180)
    plt.close(fig)

    trace_b = load("stageC_trace_baseline_ep2_rank0.json")
    trace_t = load("stageC_trace_team_ep2_rank0.json")
    work_rows = []
    for payload in (trace_b, trace_t):
        sums = work_sums(payload)
        work_rows.append({"mode": payload["mode"], "calls": len(payload["stage_rows"]), **sums})
    write_csv(ANALYSIS / "ep2_work.csv", work_rows)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, key in zip(axes, ["calls", "assignments", "active_local_experts"]):
        ax.bar(["Baseline", "TEAM"], [work_rows[0][key], work_rows[1][key]], color=["#9ca3af", "#2563eb"])
        ax.set_title(key.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("True EP2 structural work (same traced request)")
    fig.tight_layout()
    fig.savefig(PLOTS / "07_ep2_work.png", dpi=180)
    plt.close(fig)

    # Stage C: matched Python-reference before/after and production-like fused TEAM residual.
    py_stages = []
    for payload in (trace_b, trace_t):
        sums = stage_sums(payload)
        denom = payload["request_wall_s"] * 1000
        for stage in ("router", "prepare", "dispatch", "expert", "combine"):
            value = sums[f"{stage}_ms"]
            py_stages.append({"mode": payload["mode"], "stage": stage, "ms": value, "request_share_pct": value / denom * 100})
    write_csv(ANALYSIS / "ep2_python_stage_breakdown.csv", py_stages)

    stages = ["router", "prepare", "dispatch", "expert", "combine"]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bvals = [next(r["ms"] for r in py_stages if r["mode"] == "baseline" and r["stage"] == s) for s in stages]
    tvals = [next(r["ms"] for r in py_stages if r["mode"] == "team" and r["stage"] == s) for s in stages]
    ax.bar(x - 0.18, bvals, 0.36, label="Baseline", color="#9ca3af")
    ax.bar(x + 0.18, tvals, 0.36, label="TEAM", color="#2563eb")
    ax.set_xticks(x, stages)
    ax.set_ylabel("Summed CUDA-event time (ms)")
    ax.set_title("Reference EP2 stage breakdown (observer-heavy)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS / "08_ep2_stage_before_after.png", dpi=180)
    plt.close(fig)

    fused_trace = load("stageC_trace_team_ep2_fused_shapes_rank0.json")
    fused_clean_ms = mean_latency(fused_t) * 1000
    fused_sum = stage_sums(fused_trace)
    attention_ms = fused_trace["module_event_summary"]["attention"]["sum_ms"]
    decoder_ms = fused_trace["module_event_summary"]["decoder_layer"]["sum_ms"]
    lm_head_ms = fused_trace["module_event_summary"]["lm_head"]["sum_ms"]
    moe_components = sum(fused_sum[f"{stage}_ms"] for stage in stages)
    moe_unattributed_ms = max(0.0, fused_sum["moe_ms"] - moe_components)
    decoder_other_ms = max(0.0, decoder_ms - attention_ms - fused_sum["moe_ms"])
    outside_decoder_ms = max(0.0, fused_clean_ms - decoder_ms - lm_head_ms)
    residual = [
        {"stage": stage, "ms": fused_sum[f"{stage}_ms"], "clean_request_share_pct": fused_sum[f"{stage}_ms"] / fused_clean_ms * 100}
        for stage in stages
    ] + [
        {"stage": "moe_unattributed", "ms": moe_unattributed_ms, "clean_request_share_pct": moe_unattributed_ms / fused_clean_ms * 100},
        {"stage": "attention", "ms": attention_ms, "clean_request_share_pct": attention_ms / fused_clean_ms * 100},
        {"stage": "other_decoder", "ms": decoder_other_ms, "clean_request_share_pct": decoder_other_ms / fused_clean_ms * 100},
        {"stage": "lm_head", "ms": lm_head_ms, "clean_request_share_pct": lm_head_ms / fused_clean_ms * 100},
        {"stage": "outside_decoder", "ms": outside_decoder_ms, "clean_request_share_pct": outside_decoder_ms / fused_clean_ms * 100},
    ]
    write_csv(ANALYSIS / "team_fused_residual_breakdown.csv", residual)
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    labels = [r["stage"].replace("_", " ") for r in residual]
    values = [r["ms"] for r in residual]
    colors = ["#4c78a8"] * 6 + ["#59a14f", "#f28e2b", "#e15759", "#b07aa1"]
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Attributed time (ms)")
    ax.set_title("TEAM EP2 residual after fused-expert trivial-fix control")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "09_team_fused_residual.png", dpi=180)
    plt.close(fig)

    comm_rows = []
    for payload in (trace_b, trace_t):
        work = work_sums(payload)
        hidden_bytes_per_assignment = 2048 * 2
        one_way_remote = work["remote_assignments_from_source"] * hidden_bytes_per_assignment
        comm_rows.append({
            "mode": payload["mode"],
            "moe_calls": len(payload["stage_rows"]),
            "a2a_collectives": 2 * len(payload["stage_rows"]),
            "remote_assignments": work["remote_assignments_from_source"],
            "dispatch_remote_bytes": one_way_remote,
            "combine_remote_bytes": one_way_remote,
            "bidirectional_remote_bytes": one_way_remote * 2,
        })
    write_csv(ANALYSIS / "ep2_communication.csv", comm_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].bar(["Baseline", "TEAM"], [x["moe_calls"] for x in comm_rows], color=["#9ca3af", "#2563eb"])
    axes[0].set_title("Communication call count")
    axes[0].set_ylabel("MoE A2A round trips")
    axes[1].bar(["Baseline", "TEAM"], [x["bidirectional_remote_bytes"] / 1e9 for x in comm_rows], color=["#9ca3af", "#2563eb"])
    axes[1].set_title("Remote hidden payload")
    axes[1].set_ylabel("Dispatch + combine (GB)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("TEAM trades fewer collectives for larger speculative payload")
    fig.tight_layout()
    fig.savefig(PLOTS / "10_ep2_communication.png", dpi=180)
    plt.close(fig)

    shape_rows = []
    for key, entry in fused_trace["module_event_summary"].items():
        if key.startswith("decoder_layer_rows_"):
            rows = int(key.rsplit("_", 1)[1])
            shape_rows.append({"physical_rows": rows, **entry})
    shape_rows.sort(key=lambda row: row["physical_rows"])
    write_csv(ANALYSIS / "expert_kernel_shapes.csv", shape_rows)
    save_bar(
        PLOTS / "11_expert_kernel_shapes.png",
        [f"M={r['physical_rows']}\n({r['count']} calls)" for r in shape_rows],
        [r["mean_ms"] for r in shape_rows],
        "Mean full decoder-layer time (ms)",
        "TEAM branch shapes on fused EP2 substrate",
        ["#59a14f", "#f28e2b", "#e15759"],
    )

    # Coarse event timeline: summed stage occupancy in dependency order.
    fig, ax = plt.subplots(figsize=(9, 2.2))
    left = 0.0
    colors = ["#4c78a8", "#72b7b2", "#f28e2b", "#e15759", "#b07aa1"]
    for stage, color in zip(stages, colors):
        value = fused_sum[f"{stage}_ms"]
        ax.barh(["TEAM fused"], [value], left=left, color=color, label=stage)
        left += value
    ax.set_xlabel("Summed same-device CUDA-event time (ms)")
    ax.set_title("MoE critical sequence; no overlap inferred from summed events")
    ax.legend(ncol=5, bbox_to_anchor=(0.5, -0.35), loc="upper center")
    fig.tight_layout()
    fig.savefig(PLOTS / "12_team_fused_timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    dup = load("stageC_trace_team_ep2_branchdup_rank0.json")
    dup_rows = dup["stage_rows"]
    assignments = sum(row["assignments"] for row in dup_rows)
    duplicate_metrics = {
        "assignments": assignments,
        "same_position_route_duplicates": sum(row.get("cross_branch_same_position_route_duplicates", 0) for row in dup_rows),
        "exact_input_duplicates": sum(row.get("cross_branch_exact_input_duplicates", 0) for row in dup_rows),
        "near_input_duplicates_1pct": sum(row.get("cross_branch_near_input_duplicates_1pct", 0) for row in dup_rows),
        "near_input_duplicates_5pct": sum(row.get("cross_branch_near_input_duplicates_5pct", 0) for row in dup_rows),
    }
    duplicate_rows = [
        {"metric": key, "count": value, "fraction_pct": value / assignments * 100}
        for key, value in duplicate_metrics.items() if key != "assignments"
    ]
    write_csv(ANALYSIS / "branch_duplicates.csv", duplicate_rows)

    m32 = next(row for row in shape_rows if row["physical_rows"] == 32)
    m128 = next(row for row in shape_rows if row["physical_rows"] == 128)
    speculative_branch_saving_ms = m128["count"] * (m128["mean_ms"] - m32["mean_ms"])
    route_dup_fraction = duplicate_metrics["same_position_route_duplicates"] / assignments
    route_plan_ms = route_dup_fraction * fused_sum["prepare_ms"]
    candidates = [
        {
            "candidate": "existing fused expert packing",
            "evidence": "measured",
            "source_ms": (mean_latency(load("stageB_warm_team_ep2_r4_rank0.json")) - mean_latency(fused_t)) * 1000,
            "perfect_oracle_e2e_pct": (mean_latency(load("stageB_warm_team_ep2_r4_rank0.json")) - mean_latency(fused_t)) / mean_latency(load("stageB_warm_team_ep2_r4_rank0.json")) * 100,
            "independent_novel_candidate": "no",
            "kill_reason": "vLLM fused_experts already recovers it; reference Python loop artifact",
        },
        {
            "candidate": "perfect early speculative-branch commitment",
            "evidence": "perfect_oracle",
            "source_ms": speculative_branch_saving_ms,
            "perfect_oracle_e2e_pct": speculative_branch_saving_ms / fused_clean_ms * 100,
            "independent_novel_candidate": "no",
            "kill_reason": "below 10%; any predictor/verification reduces it; overlaps speculative dLLM scheduling",
        },
        {
            "candidate": "same-route metadata delta reuse",
            "evidence": "generous_perfect_oracle",
            "source_ms": route_plan_ms,
            "perfect_oracle_e2e_pct": route_plan_ms / fused_clean_ms * 100,
            "independent_novel_candidate": "no",
            "kill_reason": "below 5%; 51% route duplication has only 0.03% exact hidden-input duplication",
        },
        {
            "candidate": "all router/gating elimination",
            "evidence": "physically_unrealistic_upper_bound",
            "source_ms": fused_sum["router_ms"],
            "perfect_oracle_e2e_pct": fused_sum["router_ms"] / fused_clean_ms * 100,
            "independent_novel_candidate": "no",
            "kill_reason": "fresh hidden state changes; exact route cannot be assumed; Epoch intentionally recomputes gates",
        },
        {
            "candidate": "all EP communication elimination",
            "evidence": "physically_unrealistic_upper_bound",
            "source_ms": fused_sum["dispatch_ms"] + fused_sum["combine_ms"],
            "perfect_oracle_e2e_pct": (fused_sum["dispatch_ms"] + fused_sum["combine_ms"]) / fused_clean_ms * 100,
            "independent_novel_candidate": "no",
            "kill_reason": "remote experts require transport; optimized EP/fusion is direct prior art",
        },
    ]
    write_csv(ANALYSIS / "candidate_oracles.csv", candidates)
    save_bar(
        PLOTS / "13_candidate_oracles.png",
        ["Fused\n(existing)", "Early\ncommit", "Route-plan\nreuse", "Router\nfree", "EP comm\nfree"],
        [r["perfect_oracle_e2e_pct"] for r in candidates],
        "Request E2E effect / upper bound (%)",
        "Residual candidate attack: measured fix vs perfect oracles",
        ["#59a14f", "#f28e2b", "#9ca3af", "#9ca3af", "#9ca3af"],
    )

    risk = np.array([
        [0, 0, 5, 5],  # fused: high evidence/headroom but no novelty/nontriviality
        [2, 3, 2, 2],  # early commitment
        [1, 4, 2, 3],  # route reuse
        [1, 5, 1, 4],  # router-free
        [1, 5, 1, 4],  # comm-free
    ])
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    image = ax.imshow(risk, cmap="YlOrRd", vmin=0, vmax=5, aspect="auto")
    ax.set_xticks(range(4), ["Novelty risk", "Correctness risk", "Evidence", "Triviality risk"])
    ax.set_yticks(range(5), ["Existing fused", "Early commit", "Route reuse", "Router free", "Comm free"])
    for i in range(risk.shape[0]):
        for j in range(risk.shape[1]):
            ax.text(j, i, str(risk[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Risk/evidence score (0–5; evidence higher is better)")
    ax.set_title("Candidate novelty/risk matrix")
    fig.tight_layout()
    fig.savefig(PLOTS / "14_candidate_risk.png", dpi=180)
    plt.close(fig)

    summary = {
        "stage_a": {
            "restart_speedups_x": stage_a_speedups,
            "median_speedup_x": statistics.median(stage_a_speedups),
            "r1_speedup_x": stage_a_speedups[0],
            "trace_nfe_reduction_pct": (1 - trace_a_t["trace_summary"]["model_forwards"] / trace_a_b["trace_summary"]["model_forwards"]) * 100,
            "trace_active_expert_event_reduction_pct": (1 - trace_a_t["trace_summary"]["mean_active_experts_per_layer_call"] * trace_a_t["trace_summary"]["moe_layer_calls"] / (trace_a_b["trace_summary"]["mean_active_experts_per_layer_call"] * trace_a_b["trace_summary"]["moe_layer_calls"])) * 100,
            "trace_assignment_change_pct": (trace_a_t["trace_summary"]["token_expert_pairs"] / trace_a_b["trace_summary"]["token_expert_pairs"] - 1) * 100,
            "observer_tax_baseline_pct": (mean_latency(trace_a_b) / mean_latency(load("stageA_smoke_baseline.json")) - 1) * 100,
            "observer_tax_team_pct": (mean_latency(trace_a_t) / mean_latency(load("stageA_smoke_team.json")) - 1) * 100,
        },
        "stage_b": {
            "reference_restart_speedups_x": ep2_speedups,
            "reference_median_speedup_x": statistics.median(ep2_speedups),
            "warm_speedup_x": ep2_speedups[-1],
            "fused_speedup_x": fused_speedup,
            "python_team_s": mean_latency(load("stageB_warm_team_ep2_r4_rank0.json")),
            "fused_team_s": mean_latency(fused_t),
            "fused_vs_python_team_reduction_pct": (1 - mean_latency(fused_t) / mean_latency(load("stageB_warm_team_ep2_r4_rank0.json"))) * 100,
            "layer_correctness": layer_validation_rows,
        },
        "stage_c": {
            "fused_clean_request_ms": fused_clean_ms,
            "fused_trace_request_ms": mean_latency(fused_trace) * 1000,
            "trace_vs_clean_pct": (mean_latency(fused_trace) / mean_latency(fused_t) - 1) * 100,
            "route_duplicate_pct": route_dup_fraction * 100,
            "exact_input_duplicate_pct": duplicate_metrics["exact_input_duplicates"] / assignments * 100,
            "near_input_1pct_pct": duplicate_metrics["near_input_duplicates_1pct"] / assignments * 100,
            "near_input_5pct_pct": duplicate_metrics["near_input_duplicates_5pct"] / assignments * 100,
            "dominant_stage": "attention",
            "dominant_stage_share_pct": attention_ms / fused_clean_ms * 100,
            "moe_stage_share_pct": fused_sum["moe_ms"] / fused_clean_ms * 100,
        },
        "decision": {
            "label": "POSITIVE-CONTROL-ONLY",
            "best_independent_candidate": "perfect early speculative-branch commitment",
            "best_independent_perfect_oracle_e2e_pct": speculative_branch_saving_ms / fused_clean_ms * 100,
            "ep4_validation_justified": False,
        },
    }
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
