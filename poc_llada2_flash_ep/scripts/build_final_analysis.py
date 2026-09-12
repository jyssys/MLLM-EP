#!/usr/bin/env python3
"""Build final tables and figures from clean and observer-heavy measurements.

Observer-heavy CUDA-event traces are used only for stage proportions, route
geometry, and analytical upper bounds. Clean dInfer runs supply wall time and
throughput. The script deliberately does not multiply rank rows into request
latency.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TRACE = ROOT / "results" / "trace"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180)
    plt.close()


def log_time(path: Path) -> float:
    matches = re.findall(r"Forward:\s*\d+,\s*Time:\s*([0-9.]+)", path.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"no clean time in {path}")
    return float(matches[-1])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(ROOT / "PAIRED_SCALING.csv")
    batches = pd.read_csv(ROOT / "BATCH_SCALING.csv")
    micro = pd.read_csv(ROOT / "MICROBATCH_SCALING.csv")
    quality = pd.read_csv(ROOT / "QUALITY_RESULTS.csv")

    m8_dir = TRACE / "ep4" / "b16" / "micro8_g32_full" / "analysis"
    m16_dir = TRACE / "ep4" / "b16" / "micro16_g32_full" / "analysis"
    m8_logical = pd.read_csv(m8_dir / "logical_invocations.csv")
    m16_logical = pd.read_csv(m16_dir / "logical_invocations.csv")
    m8_stage = pd.read_csv(m8_dir / "stage_summary.csv")
    m16_stage = pd.read_csv(m16_dir / "stage_summary.csv")
    temporal = pd.read_csv(
        TRACE / "ep4" / "b2" / "temporal_b2_g128" / "analysis" / "temporal_summary.csv"
    )
    phase = pd.read_csv(
        TRACE / "ep4" / "b2" / "temporal_b2_g128" / "analysis" / "phase_summary.csv"
    )
    denoise = json.loads(
        (TRACE / "ep4" / "b16" / "blockonly_micro8_g32" / "analysis" / "denoise.summary.json").read_text()
    )
    block = pd.read_csv(
        TRACE / "ep4" / "b16" / "blockonly_micro8_g32" / "analysis" / "block_stage_summary.csv"
    )
    block_all = block[block.phase == "all"].set_index("stage")
    mlp_share = float(block_all.loc["mlp", "share_of_observed_block_sum"])
    attention_share = float(block_all.loc["attention", "share_of_observed_block_sum"])

    m8_all = m8_stage[m8_stage.phase == "all"].set_index("stage")
    inner_total = float(m8_all.sum_critical_ms.sum())
    inner_share = (m8_all.sum_critical_ms / inner_total).to_dict()
    balance_fraction = float(
        np.average(
            1.0 - m8_logical.rank_load_mean / m8_logical.rank_load_max,
            weights=m8_logical.expert_critical_ms,
        )
    )
    route_lag1 = temporal[
        (temporal.lag == 1) & (temporal.phase == "all") & (temporal.layer_group == "all")
    ].iloc[0]
    stable_branch = float(route_lag1.topk_expert_overlap_fraction)
    exact_route = float(route_lag1.topk_set_agreement)

    comm = pd.read_csv(ROOT / "DEEPEP_COMM_SWEEP.csv")
    m256_comm = float(comm.loc[comm.global_physical_m == 256, "communication_p50_ms"].iloc[0])
    fixed_floor = float(comm.communication_p50_ms.min())
    payload_sensitive_fraction = max(0.0, 1.0 - fixed_floor / m256_comm)

    # These are optimistic request-level mappings: block MLP share times the
    # relevant inner-MoE component. They are upper bounds, not measured gains.
    packing_expert_reduction = max(
        0.0,
        1.0
        - float(m16_stage[(m16_stage.phase == "all") & (m16_stage.stage == "expert")].mean_critical_ms.iloc[0])
        / float(m8_stage[(m8_stage.phase == "all") & (m8_stage.stage == "expert")].mean_critical_ms.iloc[0]),
    )
    packing_e2e = 100.0 * mlp_share * inner_share["expert"] * packing_expert_reduction
    load_balance_e2e = 100.0 * mlp_share * inner_share["expert"] * balance_fraction
    full_comm_share = mlp_share * (inner_share["dispatch"] + inner_share["combine"])
    replica_perfect = 100.0 * full_comm_share * stable_branch
    replica_feasible_upper = replica_perfect * payload_sensitive_fraction
    route_plan_perfect = 100.0 * mlp_share * inner_share["router"]
    route_plan_feasible = route_plan_perfect * exact_route
    live_removed = 1.0 - float(denoise["all"]["live_to_physical_ratio_mean"])
    freshlane_expert_only = 100.0 * mlp_share * inner_share["expert"] * live_removed
    freshlane_full_routed = 100.0 * mlp_share * sum(
        inner_share[key] for key in ("router", "dispatch", "expert", "combine")
    ) * live_removed
    overlap_perfect = 100.0 * min(attention_share, full_comm_share)

    ep_m4 = float(micro[(micro.topology == "ep4") & (micro.model_forward_microbatch == 4)].wall_seconds_median.iloc[0])
    ep_m8 = float(micro[(micro.topology == "ep4") & (micro.model_forward_microbatch == 8)].wall_seconds_median.iloc[0])
    trivial_batch_gain = 100.0 * (ep_m4 - ep_m8) / ep_m4

    candidate_rows = [
        {
            "candidate": "A_denoising_expert_work_packing",
            "regime": "EP4 submitted=16 physical-M about 256-512",
            "measured_source": "observer-heavy micro8 vs micro16 expert stage",
            "perfect_e2e_pct": round(packing_e2e, 3),
            "feasible_e2e_pct": round(packing_e2e, 3),
            "throughput_pct": "",
            "quality_risk": "none if exact packing",
            "novelty_risk": "high: generic grouped-GEMM batching",
            "implementation_complexity": "medium",
            "status": "KILL_HEADROOM_LT5",
        },
        {
            "candidate": "B_refinement_microbatch_coalescing",
            "regime": "EP4 submitted=16 default mini4 to best static mini8",
            "measured_source": "clean wall time",
            "perfect_e2e_pct": round(trivial_batch_gain, 3),
            "feasible_e2e_pct": round(trivial_batch_gain, 3),
            "throughput_pct": round(100.0 * (238.41473208366676 / 125.4693045438217 - 1.0), 3),
            "quality_risk": "none",
            "novelty_risk": "direct collision: existing dInfer mini_batch_size",
            "implementation_complexity": "none",
            "status": "TRIVIAL_EXISTING_KNOB_NOT_RESEARCH",
        },
        {
            "candidate": "C_block_scoped_replication",
            "regime": "EP4 micro8",
            "measured_source": "lag1 branch overlap + DeepEP payload sweep",
            "perfect_e2e_pct": round(replica_perfect, 3),
            "feasible_e2e_pct": round(replica_feasible_upper, 3),
            "throughput_pct": "",
            "quality_risk": "none for exact weights",
            "novelty_risk": "high: predictive expert replication",
            "implementation_complexity": "high",
            "status": "KILL_FEASIBLE_LT5",
        },
        {
            "candidate": "D_denoising_load_shaping",
            "regime": "EP4 micro8",
            "measured_source": "perfect proportional rank-balance lower bound",
            "perfect_e2e_pct": round(load_balance_e2e, 3),
            "feasible_e2e_pct": round(load_balance_e2e, 3),
            "throughput_pct": "",
            "quality_risk": "routing changes unless replication",
            "novelty_risk": "high: EPLB/load-aware routing",
            "implementation_complexity": "medium-high",
            "status": "KILL_HEADROOM_LT8",
        },
        {
            "candidate": "E_phase_adaptive_topology",
            "regime": "TP4 vs dense-TP4+routed-EP4",
            "measured_source": "clean crossover but incompatible resident layouts",
            "perfect_e2e_pct": 0.0,
            "feasible_e2e_pct": 0.0,
            "throughput_pct": "",
            "quality_risk": "none",
            "novelty_risk": "medium-high: dynamic TP/EP",
            "implementation_complexity": "very high; dual layout does not fit",
            "status": "KILL_NO_FEASIBLE_SWITCH_CONTRACT",
        },
        {
            "candidate": "F_fresh_work_compaction",
            "regime": "EP4 micro8 all denoising phases",
            "measured_source": "physical/live rows; full routed pipeline oracle",
            "perfect_e2e_pct": round(freshlane_full_routed, 3),
            "feasible_e2e_pct": 0.0,
            "throughput_pct": "",
            "quality_risk": "exact liveness required",
            "novelty_risk": "direct collision with Epoch Expert Atlas/Liveness/FreshLane",
            "implementation_complexity": "high",
            "status": "EXCLUDED_PRIOR_ART_COLLISION",
        },
        {
            "candidate": "G_route_dispatch_plan_reuse",
            "regime": "EP4 batch2 gen128",
            "measured_source": "lag1 exact route-set agreement",
            "perfect_e2e_pct": round(route_plan_perfect, 3),
            "feasible_e2e_pct": round(route_plan_feasible, 3),
            "throughput_pct": "",
            "quality_risk": "stale plan if route changes",
            "novelty_risk": "high: metadata caching/prefetch",
            "implementation_complexity": "medium",
            "status": "KILL_FEASIBLE_LT5",
        },
        {
            "candidate": "H_attention_EP_communication_overlap",
            "regime": "EP4 micro8",
            "measured_source": "zero-contention min(attention,dispatch+combine) bound",
            "perfect_e2e_pct": round(overlap_perfect, 3),
            "feasible_e2e_pct": 0.0,
            "throughput_pct": "",
            "quality_risk": "none if cross-request",
            "novelty_risk": "direct collision with generic DeepEP/TBO serving overlap",
            "implementation_complexity": "high",
            "status": "EXCLUDED_PRIOR_ART_AND_NO_LIVE_CAUSAL_POC",
        },
        {
            "candidate": "I_J_request_locality_support_grouping",
            "regime": "EP4 temporal trace",
            "measured_source": "rank-load stability but no independent wait-free request oracle",
            "perfect_e2e_pct": 0.0,
            "feasible_e2e_pct": 0.0,
            "throughput_pct": "",
            "quality_risk": "none",
            "novelty_risk": "high: affinity batching/scheduling",
            "implementation_complexity": "medium",
            "status": "NOT_ACTIVATED_BY_BOTTLENECK",
        },
        {
            "candidate": "K_phase_precision_adaptation",
            "regime": "not run",
            "measured_source": "no quality-valid FP8 path in bounded substrate",
            "perfect_e2e_pct": 0.0,
            "feasible_e2e_pct": 0.0,
            "throughput_pct": "",
            "quality_risk": "unmeasured/high",
            "novelty_risk": "high: phase mixed precision",
            "implementation_complexity": "medium",
            "status": "NOT_ACTIVATED_NO_QUALITY_VALID_PATH",
        },
    ]
    write_csv(ROOT / "CANDIDATE_ORACLES.csv", candidate_rows)

    observer_rows = []
    clean_m8 = float(micro[(micro.topology == "ep4") & (micro.model_forward_microbatch == 8)].wall_seconds_median.iloc[0])
    clean_m16 = float(micro[(micro.topology == "ep4") & (micro.model_forward_microbatch == 16)].wall_seconds_median.iloc[0])
    for label, clean, trace_log in (
        ("ep4_micro8_block_only", clean_m8, ROOT / "logs" / "trace_ep4_b16_blockonly_micro8_g32.log"),
        ("ep4_micro16_block_only", clean_m16, ROOT / "logs" / "trace_ep4_b16_blockonly_micro16_g32.log"),
        ("ep4_micro8_detailed", clean_m8, ROOT / "logs" / "trace_ep4_b16_micro8_g32_full.log"),
        ("ep4_micro16_detailed", clean_m16, ROOT / "logs" / "trace_ep4_b16_micro16_g32_full.log"),
    ):
        traced = log_time(trace_log)
        observer_rows.append(
            {
                "run": label,
                "clean_seconds": clean,
                "instrumented_seconds": traced,
                "observer_tax_percent": 100.0 * (traced / clean - 1.0),
                "eligible_for_e2e_claim": False,
            }
        )
    write_csv(ROOT / "OBSERVER_TAX.csv", observer_rows)

    bottleneck_rows = []
    for phase_name in ("all", "early", "middle", "late"):
        selected = block[block.phase == phase_name]
        for _, row in selected.iterrows():
            bottleneck_rows.append(
                {
                    "topology": "ep4",
                    "submitted_batch": 16,
                    "microbatch": 8,
                    "phase": phase_name,
                    "stage": row.stage,
                    "observer_heavy_critical_sum_ms": row.sum_critical_ms,
                    "share_percent": 100.0 * row.share_of_observed_block_sum,
                }
            )
    write_csv(ROOT / "BOTTLENECK_BREAKDOWN.csv", bottleneck_rows)

    phase_rows = []
    for phase_name in ("all", "early", "middle", "late"):
        p = phase[phase.phase == phase_name].iloc[0]
        phase_rows.append(
            {
                "phase": phase_name,
                "live_to_physical_ratio": denoise[phase_name]["live_to_physical_ratio_mean"],
                "accepted_per_forward": denoise[phase_name]["accepted_per_forward_mean"],
                "remote_fraction": p.remote_fraction_mean,
                "active_experts": p.active_experts_mean,
                "tiny_expert_fraction_le4": p.tiny_expert_fraction_le4_mean,
                "rank_load_cv": p.rank_load_cv_mean,
                "rank_fanout": p.rank_fanout_mean_mean,
            }
        )
    write_csv(ROOT / "DENOISING_PHASE.csv", phase_rows)

    topology_rows = []
    for topology, hbm_mib in (("tp4", 58367), ("ep4", 58227)):
        q = quality[(quality.task == "gsm8k") & (quality.topology == topology)]
        topology_rows.append(
            {
                "run_id": "gsm8k32_gen128_median",
                "topology": topology,
                "dense_tp": 4,
                "moe_tp": 4 if topology == "tp4" else 1,
                "ep": 1 if topology == "tp4" else 4,
                "dp": 1,
                "backend": "TP collectives" if topology == "tp4" else "DeepEP normal + BF16 Triton",
                "batch": 4,
                "quality_valid": True,
                "hbm_max_gib": hbm_mib / 1024.0,
                "latency_s": float(q.wall_seconds.median()),
                "throughput_req_s": 32.0 / float(q.wall_seconds.median()),
                "tokens_s": float(q.tokens_per_second.median()),
                "nfe": int(q.nfe.median()),
                "notes": "bounded GSM8K score identical 5/32",
            }
        )
    write_csv(ROOT / "TOPOLOGY_RESULTS.csv", topology_rows)

    summary = {
        "block_mlp_share": mlp_share,
        "block_attention_share": attention_share,
        "inner_moe_stage_share": inner_share,
        "weighted_perfect_expert_balance_fraction": balance_fraction,
        "lag1_branch_overlap": stable_branch,
        "lag1_exact_topk_set": exact_route,
        "deepep_payload_sensitive_fraction_at_m256": payload_sensitive_fraction,
        "live_removed_fraction": live_removed,
        "freshlane_expert_only_e2e_upper_pct": freshlane_expert_only,
        "freshlane_full_routed_e2e_upper_pct": freshlane_full_routed,
        "trivial_microbatch_wall_gain_pct": trivial_batch_gain,
        "strongest_novel_credible_oracle_pct": load_balance_e2e,
        "final_label": "CHARACTERIZATION-ONLY",
    }
    (ROOT / "ANALYSIS_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Figure 1: HBM.
    plt.figure(figsize=(5.8, 3.8))
    plt.bar(["TP4", "EP4"], [58367 / 1024, 58227 / 1024], color=["#4c78a8", "#f58518"])
    plt.axhline(80, color="black", ls="--", lw=1, label="H100 capacity")
    plt.ylabel("Peak observed HBM / rank (GiB)")
    plt.ylim(0, 85)
    plt.legend()
    save("01_hbm_by_topology.png")

    # Figures 2/3: quality-valid longer run topology medians.
    qmed = quality[quality.task == "gsm8k"].groupby("topology").median(numeric_only=True)
    plt.figure(figsize=(5.8, 3.8))
    plt.bar(["TP4", "EP4"], [qmed.loc["tp4", "wall_seconds"], qmed.loc["ep4", "wall_seconds"]])
    plt.ylabel("GSM8K-32 batch completion (s), median")
    save("02_latency_by_topology.png")
    plt.figure(figsize=(5.8, 3.8))
    plt.bar(["TP4", "EP4"], [qmed.loc["tp4", "tokens_per_second"], qmed.loc["ep4", "tokens_per_second"]])
    plt.ylabel("Generated tokens / s, median")
    save("03_throughput_by_topology.png")

    # Figures 4/5: submitted batch clean medians with all restarts.
    ag = paired[paired.record_type == "aggregate"]
    for metric, ylabel, filename in (
        ("wall_seconds", "32-request batch completion (s)", "04_latency_vs_batch.png"),
        ("generated_tokens_per_second", "Generated tokens / s", "05_throughput_vs_batch.png"),
    ):
        plt.figure(figsize=(6.4, 4.0))
        for topology in ("tp4", "ep4"):
            d = ag[ag.topology == topology].sort_values("batch")
            plt.plot(d.batch, d[metric], marker="o", label=topology.upper())
        plt.xscale("log", base=2)
        plt.xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
        plt.xlabel("Submitted request batch (fixed mini_batch_size=4)")
        plt.ylabel(ylabel)
        plt.grid(alpha=.25)
        plt.legend()
        save(filename)

    # Figure 6: representative stage share (not a clean E2E decomposition).
    stage_names = ["prepare_attn", "attention", "prepare_mlp", "mlp", "postprocess"]
    shares = [100.0 * float(block_all.loc[s, "share_of_observed_block_sum"]) for s in stage_names]
    plt.figure(figsize=(7.0, 3.8))
    plt.bar(stage_names, shares)
    plt.ylabel("Observer-heavy block critical sum (%)")
    plt.xticks(rotation=20, ha="right")
    save("06_stage_share_representative.png")

    # Figure 7: detailed EP stage medians for two physical-M regimes.
    stages = ["router", "dispatch", "expert", "combine", "shared", "gather"]
    x = np.arange(len(stages))
    plt.figure(figsize=(7.2, 4.0))
    for idx, (label, data) in enumerate((("micro8 / M≈256", m8_stage), ("micro16 / variable M", m16_stage))):
        d = data[data.phase == "all"].set_index("stage")
        plt.bar(x + (idx - .5) * .36, [d.loc[s, "median_critical_ms"] for s in stages], width=.36, label=label)
    plt.xticks(x, stages, rotation=20)
    plt.ylabel("Critical-rank CUDA time (ms), median")
    plt.legend()
    save("07_dispatch_expert_combine_by_work_shape.png")

    # Figures 8/9: expert group geometry and rank imbalance.
    plt.figure(figsize=(6.2, 3.8))
    plt.bar(["M≈256", "variable M\n(up to 512)"], [100*m8_logical.tiny_expert_fraction_le4.mean(), 100*m16_logical.tiny_expert_fraction_le4.mean()])
    plt.ylabel("Active experts with ≤4 rows (%)")
    save("08_expert_gemm_shape_distribution.png")
    plt.figure(figsize=(6.2, 3.8))
    vals = [m8_logical.rank_load_cv, m16_logical.rank_load_cv]
    plt.boxplot(vals, tick_labels=["M≈256", "variable M\n(up to 512)"], showfliers=False)
    plt.ylabel("Rank-load coefficient of variation")
    save("09_rank_imbalance_by_work_shape.png")

    # Figure 10: denoising live ratio.
    pnames = ["early", "middle", "late"]
    plt.figure(figsize=(6.2, 3.8))
    plt.bar(pnames, [100*denoise[p]["live_to_physical_ratio_mean"] for p in pnames])
    plt.ylabel("Decision-live / physically executed rows (%)")
    save("10_denoising_phase_workload.png")

    # Figure 11: temporal route and rank persistence.
    t = temporal[(temporal.phase == "all") & (temporal.layer_group == "all")].sort_values("lag")
    plt.figure(figsize=(6.5, 4.0))
    plt.plot(t.lag, 100*t.topk_expert_overlap_fraction, marker="o", label="top-k expert overlap")
    plt.plot(t.lag, 100*t.destination_overlap_fraction, marker="o", label="destination-rank overlap")
    plt.plot(t.lag, 100*t.rank_load_cosine, marker="o", label="rank-load cosine")
    plt.xticks([1, 2, 4, 8])
    plt.xlabel("Denoising lag")
    plt.ylabel("Similarity (%)")
    plt.ylim(35, 101)
    plt.legend()
    save("11_temporal_route_persistence.png")

    # Figure 12: candidate oracle comparison, clearly separating collisions.
    cdf = pd.DataFrame(candidate_rows)
    eligible = cdf[~cdf.candidate.str.startswith("B_")]
    plt.figure(figsize=(8.2, 4.8))
    y = np.arange(len(eligible))
    plt.barh(y + .18, eligible.perfect_e2e_pct.astype(float), height=.36, label="perfect")
    plt.barh(y - .18, eligible.feasible_e2e_pct.astype(float), height=.36, label="feasible/credible")
    plt.yticks(y, [x.split("_")[0] for x in eligible.candidate])
    plt.xlabel("Projected request E2E upper bound (%)")
    plt.legend()
    save("12_candidate_oracle_comparison.png")

    # Figure 13: identical bounded score, variable latency.
    plt.figure(figsize=(6.0, 4.0))
    for topology, marker in (("tp4", "o"), ("ep4", "s")):
        d = quality[(quality.task == "gsm8k") & (quality.topology == topology)]
        plt.scatter(d.wall_seconds, 100*d.accuracy, marker=marker, s=55, label=topology.upper())
    plt.xlabel("GSM8K-32 batch completion (s)")
    plt.ylabel("Bounded accuracy (%)")
    plt.legend()
    save("13_quality_latency_pareto.png")

    # Figure 14: faster-communication durability for the only comm-derived live oracle.
    factors = np.array([1.0, .75, .5, .25])
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(factors, replica_feasible_upper*factors, marker="o", label="block replica upper bound")
    plt.plot(factors, np.full_like(factors, load_balance_e2e), marker="o", label="perfect rank balance")
    plt.gca().invert_xaxis()
    plt.xlabel("Communication cost multiplier (lower is faster)")
    plt.ylabel("Projected request E2E upper bound (%)")
    plt.legend()
    save("14_faster_backend_sensitivity.png")

    # Figure 15: strongest actual tuning effect, explicitly labeled non-research.
    plt.figure(figsize=(7.0, 4.0))
    for topology in ("tp4", "ep4"):
        d = micro[micro.topology == topology].sort_values("model_forward_microbatch")
        plt.plot(d.model_forward_microbatch, d.wall_seconds_median, marker="o", label=topology.upper())
    plt.xscale("log", base=2)
    plt.xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
    plt.xlabel("Existing dInfer model-forward mini_batch_size")
    plt.ylabel("32-request batch completion (s), median")
    plt.legend()
    plt.grid(alpha=.25)
    save("15_best_static_microbatch_vs_default.png")


if __name__ == "__main__":
    main()
