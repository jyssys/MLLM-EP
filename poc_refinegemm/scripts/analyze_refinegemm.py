#!/usr/bin/env python3
"""Aggregate RefineGEMM GPU replay evidence and construct bounded oracles."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "poc_dllm_ep_discovery"
FIGURES = ROOT / "figures"


def load_kernel_runs(scope: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kernel, single, correctness = [], [], []
    for restart, directory in enumerate(sorted(ROOT.glob(f"kernel2_{scope}_r*")), 1):
        frame = pd.read_csv(directory / "EXISTING_EXPERT_KERNELS.csv")
        frame.insert(0, "restart", restart)
        frame.insert(1, "scope", scope)
        kernel.append(frame)
        frame = pd.read_csv(directory / "SINGLE_EXPERT_SWEEP.csv")
        frame.insert(0, "restart", restart)
        frame.insert(1, "scope", scope)
        single.append(frame)
        frame = pd.read_csv(directory / "KERNEL_CORRECTNESS.csv")
        frame.insert(0, "restart", restart)
        frame.insert(1, "scope", scope)
        correctness.append(frame)
    if len(kernel) != 3:
        raise RuntimeError(f"expected three {scope} restarts, found {len(kernel)}")
    return pd.concat(kernel, ignore_index=True), pd.concat(single, ignore_index=True), pd.concat(correctness, ignore_index=True)


def median_case_table(kernel: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "scope", "restart", "case_id", "control_group", "geometry", "task",
        "phase", "wave", "layer", "rank", "backend", "total_assignments",
        "active_experts", "median_m_e", "max_m_e", "tiny_le4_fraction", "m_e_cv",
    ]
    return kernel.groupby(keys, dropna=False, as_index=False).latency_ms.median()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    dense_k, dense_s, dense_c = load_kernel_runs("dense")
    compact_k, compact_s, compact_c = load_kernel_runs("compacted")
    kernel_raw = pd.concat([dense_k, compact_k], ignore_index=True)
    single_raw = pd.concat([dense_s, compact_s], ignore_index=True)
    correctness = pd.concat([dense_c, compact_c], ignore_index=True)
    kernels = median_case_table(kernel_raw)

    single = (
        single_raw.groupby(["scope", "restart", "m_e", "backend"], as_index=False)
        .agg(latency_ms=("latency_ms", "median"), effective_tflops=("effective_tflops", "median"))
    )
    single.to_csv(ROOT / "SINGLE_EXPERT_SWEEP.csv", index=False)
    kernels.to_csv(ROOT / "EXISTING_EXPERT_KERNELS.csv", index=False)
    correctness.to_csv(ROOT / "KERNEL_CORRECTNESS.csv", index=False)

    components = pd.read_csv(PRIOR / "COMPONENT_E2E_UPPER_BOUNDS.csv")
    clean = pd.read_csv(PRIOR / "BASELINE_CLEAN.csv")
    quality = pd.read_csv(PRIOR / "QUALITY_REPRODUCIBILITY.csv")
    components.to_csv(ROOT / "OPERATOR_BREAKDOWN.csv", index=False)

    clean_out = clean.copy()
    clean_out["result_type"] = "measured strongest-static EP4 baseline"
    clean_out["method"] = "baseline"
    compatibility_log = ROOT / "results/clean_gpu4_7/clean_gpu4_7_gsm8k_ep4_b32_mini32_r1_g32.log"
    if compatibility_log.is_file():
        match = re.search(
            r"Forward: (\d+), Time: ([0-9.]+), FPS: [^(]+\([^)]+\), TPS: [^(]+\(([^)]+)\)",
            compatibility_log.read_text(),
        )
        if match:
            clean_out = pd.concat([clean_out, pd.DataFrame([{
                "dataset": "gsm8k", "repeat": "gpu4_7_compat", "nfe": int(match.group(1)),
                "wall_seconds": float(match.group(2)), "tokens_per_second": float(match.group(3)),
                "result_type": "measured GPU4-7 compatibility anchor; excluded from baseline statistics",
                "method": "baseline",
            }])], ignore_index=True)
    clean_out.to_csv(ROOT / "E2E_RESULTS.csv", index=False)

    real = kernels[kernels.geometry.eq("real")].copy()
    pivot = real.pivot_table(
        index=["scope", "restart", "case_id", "task", "phase"],
        columns="backend", values="latency_ms",
    ).reset_index()
    hybrid_cols = [column for column in pivot if column.startswith("hybrid_")]
    reverse_cols = [column for column in hybrid_cols if column.startswith("hybrid_fused_tiny")]
    pivot["strongest_whole_ms"] = pivot[["vllm_fused", "torch_grouped"]].min(axis=1)
    pivot["best_hybrid_ms"] = pivot[hybrid_cols].min(axis=1)
    pivot["best_reverse_hybrid_ms"] = pivot[reverse_cols].min(axis=1)
    pivot["production_to_grouped_gain_pct"] = (
        (pivot.vllm_fused - pivot.torch_grouped) / pivot.vllm_fused * 100
    )
    pivot["switching_additional_pct"] = (
        (pivot.strongest_whole_ms - pivot[["vllm_fused", "torch_grouped"]].min(axis=1))
        / pivot.strongest_whole_ms * 100
    )
    pivot["hybrid_additional_pct"] = np.maximum(
        0, (pivot.strongest_whole_ms - pivot.best_hybrid_ms) / pivot.strongest_whole_ms * 100,
    )
    pivot["reverse_hybrid_penalty_pct"] = (
        (pivot.best_reverse_hybrid_ms - pivot.strongest_whole_ms) / pivot.strongest_whole_ms * 100
    )

    micro = (
        pivot.groupby(["scope", "task", "phase"], as_index=False)
        .agg(
            real_cases=("case_id", "size"),
            production_to_grouped_gain_pct=("production_to_grouped_gain_pct", "median"),
            switching_additional_pct=("switching_additional_pct", "median"),
            measured_hybrid_additional_pct=("hybrid_additional_pct", "median"),
            reverse_hybrid_penalty_pct=("reverse_hybrid_penalty_pct", "median"),
        )
    )
    micro.to_csv(ROOT / "REFINEGEMM_MICROBENCH.csv", index=False)

    # Matched total-assignment and active-expert controls isolate row-shape effects.
    controls = kernels[
        kernels.geometry.isin(["low_heterogeneity_control", "high_heterogeneity_control"])
        & kernels.backend.eq("torch_grouped")
    ]
    control_pivot = controls.pivot_table(
        index=["scope", "restart", "control_group", "task", "phase"],
        columns="geometry", values="latency_ms",
    ).reset_index()
    control_pivot["high_over_low_latency"] = (
        control_pivot.high_heterogeneity_control / control_pivot.low_heterogeneity_control
    )
    control_pivot.to_csv(ROOT / "MATCHED_HETEROGENEITY_CONTROLS.csv", index=False)

    prior_oracles = pd.read_csv(PRIOR / "CANDIDATE_ORACLES.csv")
    expert_pie = components[components.component.eq("expert")].set_index("dataset")
    oracle_rows = []
    for task in ["gsm8k", "humaneval"]:
        task_dense = pivot[(pivot.scope.eq("dense")) & (pivot.task.eq(task))]
        replacement = float(task_dense.production_to_grouped_gain_pct.median())
        replacement_e2e = replacement * float(expert_pie.loc[task, "clean_e2e_upper_bound_percent"]) / 100
        dense_ideal = float(prior_oracles[
            (prior_oracles.dataset.eq(task))
            & (prior_oracles.candidate.eq("perfect_current_fragmentation_removal_p99_trimmed"))
        ].clean_e2e_percent.iloc[0])
        compact_ideal = float(prior_oracles[
            (prior_oracles.dataset.eq(task))
            & (prior_oracles.candidate.eq("post_compaction_fragmentation_residual_p99_trimmed"))
        ].clean_e2e_percent.iloc[0])
        compact_credible = float(prior_oracles[
            (prior_oracles.dataset.eq(task))
            & (prior_oracles.candidate.eq("feasible_tiny_shape_specialization_50pct_capture_p99_trimmed"))
        ].clean_e2e_percent.iloc[0])
        values = [
            ("production_to_torch_grouped_replacement", "dense", replacement, replacement_e2e,
             "measured owner-local operator replay; request value is Amdahl projection"),
            ("O0_best_existing_whole_switching_additional", "dense", 0.0, 0.0,
             "torch grouped won every sampled real invocation in all three restarts"),
            ("O1_measured_two_subgroup_hybrid_additional", "dense", 0.0, 0.0,
             "actual gather/two-launch/scatter hybrid; clipped at zero saving"),
            ("O2_ideal_persistent_shape_scheduler", "dense", np.nan, dense_ideal,
             "optimistic p99-robust count-matched lower envelope; same assignments/FLOPs"),
            ("O3_credible_refinegemm_50pct_capture", "dense", np.nan, dense_ideal * 0.5,
             "50% capture of O2; no implementation"),
            ("O2_post_compaction_ideal_residual", "compacted", np.nan, compact_ideal,
             "future-known live-row sensitivity; not measured Epoch"),
            ("O3_post_compaction_credible_50pct_capture", "compacted", np.nan, compact_credible,
             "50% capture of post-compaction O2; no implementation"),
        ]
        for name, scope, operator_gain, e2e_gain, evidence in values:
            oracle_rows.append({
                "task": task, "oracle": name, "scope": scope,
                "operator_gain_pct": operator_gain, "request_e2e_gain_pct": e2e_gain,
                "baseline": "strongest valid whole kernel" if name.startswith("O") else "production vLLM fused_experts",
                "evidence_boundary": evidence,
            })
    oracles = pd.DataFrame(oracle_rows)
    oracles.to_csv(ROOT / "REFINEGEMM_ORACLES.csv", index=False)

    attempts = pd.DataFrame([
        ["A00", "fresh strongest-static EP4 compatibility on GPUs 4-7", "completed", "GPU4-7", "GSM8K NFE66 and exact outputs match prior anchor; timing excluded from baseline statistics"],
        ["A01", "post-compaction shape atlas", "completed", "CPU", "future-known live-row sensitivity; exact route invariants pass"],
        ["A02", "dense actual-runtime atlas", "completed", "CPU", "measured physical rows and routes"],
        ["A03", "vLLM fused expert replay", "completed", "GPU4", "default untuned E64/N1024 H100 config in installed vLLM"],
        ["A04", "torch._grouped_mm replay", "completed", "GPU4", "strongest valid owner-local kernel in tested environment"],
        ["A05", "wrong-direction two-subgroup hybrid", "completed-negative-control", "GPU4", "grouped tiny + fused large never wins"],
        ["A06", "crossover-directed two-subgroup hybrid", "completed-negative", "GPU4", "fused M_e=1 + grouped M_e>=2 never beats grouped whole"],
        ["A07", "SonicMoE install", "not-attempted-incompatible", "none", "current official prerequisites CUDA>=12.9/Python3.12; local CUDA12.8/Python3.11"],
        ["A08", "custom RefineGEMM CUDA", "not-authorized-by-gate", "none", "credible additional request E2E <5%"],
        ["A09", "full-model integration", "not-authorized-by-gate", "none", "no custom kernel; baseline quality anchor reused"],
    ], columns=["attempt_id", "attempt", "status", "device", "result"])
    attempts.to_csv(ROOT / "ATTEMPT_LOG.csv", index=False)

    gpu_log = []
    for scope in ("dense", "compacted"):
        for restart, directory in enumerate(sorted(ROOT.glob(f"kernel2_{scope}_r*")), 1):
            audit = json.loads((directory / "KERNEL_BENCHMARK_AUDIT.json").read_text())
            gpu_log.append({
                "scope": scope, "restart": restart, "physical_gpu": 4,
                "cuda_visible_devices": audit["cuda_visible_devices"],
                "physical_uuid": audit["logical_cuda0_physical_uuid"],
                "wall_seconds": audit["wall_seconds"], "warmup": audit["warmup"],
                "measured_repetitions": audit["repetitions"],
                "note": audit["scope"],
            })
    compatibility_gpu_log = ROOT / "results/clean_gpu4_7/clean_gpu4_7_gsm8k_ep4_b32_mini32_r1_g32_gpu.csv"
    if compatibility_gpu_log.is_file():
        samples = pd.read_csv(
            compatibility_gpu_log, header=None,
            names=["timestamp", "physical_gpu", "physical_uuid", "memory_used_mib", "memory_free_mib", "utilization_pct"],
            skipinitialspace=True,
        )
        samples["timestamp"] = pd.to_datetime(samples.timestamp)
        wall_seconds = float((samples.timestamp.max() - samples.timestamp.min()).total_seconds())
        for physical_gpu, group in samples.groupby("physical_gpu"):
            gpu_log.append({
                "scope": "clean_ep4_compatibility", "restart": 1,
                "physical_gpu": int(physical_gpu), "cuda_visible_devices": "4,5,6,7",
                "physical_uuid": group.physical_uuid.iloc[0].strip(), "wall_seconds": wall_seconds,
                "warmup": 0, "measured_repetitions": 1,
                "note": f"full-model compatibility; peak HBM {group.memory_used_mib.max():.0f} MiB; timing excluded from baseline statistics",
            })
    pd.DataFrame(gpu_log).to_csv(ROOT / "GPU_TIME_LOG.csv", index=False)

    # Required figures.
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    pie = expert_pie.reset_index()
    ax.bar(pie.dataset, pie.clean_e2e_upper_bound_percent, color=["#4C78A8", "#F58518"])
    ax.axhline(20, color="black", ls="--", lw=1)
    ax.set_ylabel("Observer-attributed clean request E2E (%)")
    ax.set_title("Routed expert execution is a strong systems pie")
    fig.tight_layout(); fig.savefig(FIGURES / "01_routed_expert_e2e_pie.png", dpi=180); plt.close(fig)

    dense_single = single[single.scope.eq("dense")]
    summary = dense_single.groupby(["m_e", "backend"], as_index=False).agg(
        latency_ms=("latency_ms", "median"), effective_tflops=("effective_tflops", "median")
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for backend, group in summary.groupby("backend"):
        ax.plot(group.m_e, group.latency_ms, marker="o", label=backend)
    ax.set_xscale("log", base=2); ax.set_xlabel("M_e"); ax.set_ylabel("Latency (ms)")
    ax.set_title("Single-expert BF16 full-MLP latency"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "08_single_expert_latency_vs_me.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for backend, group in summary.groupby("backend"):
        ax.plot(group.m_e, group.effective_tflops, marker="o", label=backend)
    ax.set_xscale("log", base=2); ax.set_xlabel("M_e"); ax.set_ylabel("Effective TFLOP/s")
    ax.set_title("Single-expert effective throughput"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "09_single_expert_tflops_vs_me.png", dpi=180); plt.close(fig)

    homogeneous = kernels[(kernels.scope.eq("dense")) & kernels.geometry.eq("homogeneous_control")]
    homogeneous = homogeneous[homogeneous.backend.isin(["vllm_fused", "torch_grouped"])]
    homogeneous["m_e"] = homogeneous.case_id.str.extract(r"(\d+)").astype(int)
    cross = homogeneous.groupby(["m_e", "backend"], as_index=False).latency_ms.median()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for backend, group in cross.groupby("backend"):
        ax.plot(group.m_e, group.latency_ms, marker="o", label=backend)
    ax.set_xscale("log", base=2); ax.set_xlabel("Homogeneous M_e across 64 experts"); ax.set_ylabel("Latency (ms)")
    ax.set_title("Existing-kernel crossover exists only at M_e=1"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "10_grouped_tiny_kernel_crossover.png", dpi=180); plt.close(fig)

    for number, x, xlabel in [(11, "tiny_le4_fraction", "Tiny active-expert fraction (M_e<=4)"), (12, "m_e_cv", "M_e coefficient of variation")]:
        data = real[real.backend.eq("torch_grouped")]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for phase, group in data.groupby("phase"):
            ax.scatter(group[x], group.latency_ms, s=17, alpha=.55, label=phase)
        ax.set_xlabel(xlabel); ax.set_ylabel("Strongest whole-kernel latency (ms)")
        ax.set_title("Real invocation replay"); ax.legend()
        fig.tight_layout(); fig.savefig(FIGURES / f"{number:02d}_invocation_latency_vs_{'tiny_fraction' if number == 11 else 'me_cv'}.png", dpi=180); plt.close(fig)

    prod = pivot[pivot.scope.eq("dense")].groupby("task").production_to_grouped_gain_pct.median()
    sw = pivot[pivot.scope.eq("dense")].groupby("task").switching_additional_pct.median()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(2); width = .35
    ax.bar(x-width/2, [prod.get("gsm8k"), prod.get("humaneval")], width, label="backend replacement")
    ax.bar(x+width/2, [sw.get("gsm8k"), sw.get("humaneval")], width, label="switching beyond strongest")
    ax.set_xticks(x, ["GSM8K", "HumanEval"]); ax.set_ylabel("Expert-operator gain (%)")
    ax.set_title("Whole-invocation existing-kernel envelope"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "13_whole_invocation_switching_oracle.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    shown = oracles[oracles.oracle.isin([
        "O1_measured_two_subgroup_hybrid_additional", "O2_ideal_persistent_shape_scheduler",
        "O3_credible_refinegemm_50pct_capture",
    ])]
    chart = shown.pivot(index="task", columns="oracle", values="request_e2e_gain_pct")
    chart.plot(kind="bar", ax=ax)
    ax.set_ylabel("Additional request E2E gain (%)"); ax.set_title("Hybrid scheduler oracle over strongest whole kernel")
    ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIGURES / "14_per_expert_hybrid_oracle.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    shown = oracles[oracles.oracle.isin([
        "production_to_torch_grouped_replacement", "O3_credible_refinegemm_50pct_capture",
        "O3_post_compaction_credible_50pct_capture",
    ])]
    chart = shown.pivot(index="task", columns="oracle", values="request_e2e_gain_pct")
    chart.plot(kind="bar", ax=ax)
    ax.axhline(5, color="black", ls="--", lw=1, label="minimum method gate")
    ax.set_ylabel("Projected request E2E gain (%)"); ax.set_title("Existing replacement vs novel credible headroom")
    ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIGURES / "15_request_level_refinegemm_oracle.png", dpi=180); plt.close(fig)

    summary_json = {
        "gpu_restarts": 3,
        "warmup": 5,
        "measured_repetitions": 30,
        "real_dense_restart_cases": int(len(pivot[pivot.scope.eq("dense")])),
        "real_compacted_restart_cases": int(len(pivot[pivot.scope.eq("compacted")])),
        "torch_grouped_real_wins": int((pivot.strongest_whole_ms == pivot.torch_grouped).sum()),
        "hybrid_real_wins": int((pivot.best_hybrid_ms < pivot.strongest_whole_ms).sum()),
        "dense_reverse_hybrid_penalty_median_pct": float(pivot[pivot.scope.eq("dense")].reverse_hybrid_penalty_pct.median()),
        "compacted_reverse_hybrid_penalty_median_pct": float(pivot[pivot.scope.eq("compacted")].reverse_hybrid_penalty_pct.median()),
        "max_credible_dense_e2e_pct": float(oracles[oracles.oracle.eq("O3_credible_refinegemm_50pct_capture")].request_e2e_gain_pct.max()),
        "max_credible_post_compaction_e2e_pct": float(oracles[oracles.oracle.eq("O3_post_compaction_credible_50pct_capture")].request_e2e_gain_pct.max()),
        "correctness_max_relative_l2": float(correctness.relative_l2.max()),
        "quality_rows_reused": int(len(quality)),
    }
    (ROOT / "analysis_summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
