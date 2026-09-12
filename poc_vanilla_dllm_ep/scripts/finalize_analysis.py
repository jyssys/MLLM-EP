#!/usr/bin/env python3
"""Create final tables and the pre-registered diagnostic figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=160)
    plt.close()


def analysis_csv(name: str) -> Path:
    plain = RESULTS / "analysis" / name
    return plain if plain.exists() else plain.with_suffix(plain.suffix + ".gz")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((RESULTS / "stage0_clean/summary.json").read_text())
    quality = json.loads((RESULTS / "stage0_clean/quality_summary.json").read_text())
    timing = json.loads((RESULTS / "stage1_timing/analysis/temporal_summary.json").read_text())
    temporal = json.loads((RESULTS / "analysis/temporal_summary.json").read_text())
    drafts = json.loads((RESULTS / "oracles/rank_local_draft_summary.json").read_text())
    candidates = json.loads((RESULTS / "oracles/candidate_oracles.json").read_text())
    elastic = json.loads((RESULTS / "oracles/elastic_ep_microbench_summary.json").read_text())

    eps = [1, 2, 4]
    walls = [stage0[str(ep)]["request_wall_median_s"] for ep in eps]
    plt.bar([str(ep) for ep in eps], walls)
    plt.ylabel("8-request clean wall (s)")
    plt.xlabel("EP degree")
    save("01_vanilla_latency_vs_ep.png")

    timing_rows = {int(row["ep"]): row for row in timing["stage"]}
    stage_names = ["router", "prepare", "dispatch", "expert", "combine"]
    bottom = [0.0] * 3
    for stage in stage_names:
        values = [100 * timing_rows[ep][f"critical_{stage}_request_share"] for ep in eps]
        plt.bar([str(ep) for ep in eps], values, bottom=bottom, label=stage)
        bottom = [a + b for a, b in zip(bottom, values)]
    plt.ylabel("critical-rank sub-stage share (%)\n(non-additive maxima)")
    plt.xlabel("EP degree")
    plt.legend(fontsize=7)
    save("02_vanilla_stage_breakdown.png")

    remote_gb = [stage0[str(ep)]["remote_total_bytes_median"] / 1e9 for ep in eps]
    plt.bar([str(ep) for ep in eps], remote_gb)
    plt.ylabel("Remote hidden traffic / 8 requests (GB)")
    plt.xlabel("EP degree")
    save("03_remote_traffic_vs_ep.png")

    temporal_df = pd.read_csv(analysis_csv("temporal_rows.csv"))
    ep4 = temporal_df[temporal_df.ep == 4]
    fanout = ep4.groupby("iteration").fanout.mean()
    plt.plot(fanout.index, fanout.values, marker=".")
    plt.ylim(0, 4.2)
    plt.xlabel("Denoising iteration")
    plt.ylabel("Mean destination-rank fanout")
    save("04_fanout_vs_iteration.png")

    lags = [1, 2, 4, 8]
    ep4_lag = pd.read_csv(analysis_csv("route_lag_metrics.csv"))
    ep4_lag = ep4_lag[ep4_lag.ep == 4]
    for metric, label in (("stable_branch_fraction", "stable branch"),
                          ("route_set_equal_fraction", "top-k set equal"),
                          ("top8_hot_expert_jaccard", "hot-expert Jaccard")):
        plt.plot(lags, [ep4_lag[ep4_lag.lag == lag][metric].mean() for lag in lags],
                 marker="o", label=label)
    plt.xlabel("Iteration lag")
    plt.ylabel("Fraction")
    plt.legend()
    save("05_temporal_route_similarity.png")

    h = temporal["hidden_and_output_by_lag"]
    for metric, label in (("hidden_cosine_mean", "hidden"),
                          ("branch_output_cosine_mean", "stable branch output"),
                          ("combined_output_cosine_mean", "combined MoE")):
        plt.plot(lags, [h[str(lag)][metric] for lag in lags], marker="o", label=label)
    plt.xlabel("Iteration lag")
    plt.ylabel("Cosine similarity")
    plt.legend()
    save("06_hidden_expert_similarity.png")

    local_k = [row["local_draft_k"] for row in drafts]
    plt.plot(local_k, [100 * row["proposal_union_hit_fraction"] for row in drafts], marker="o")
    plt.xlabel("Rank-local top-k")
    plt.ylabel("Four-view exact-token union hit (%)")
    save("07_rank_local_proposal_hit.png")

    coverage = [100 * row["all_rank_agreement_coverage"] for row in drafts]
    precision = [100 * row["all_rank_agreement_precision"] for row in drafts]
    plt.plot(coverage, precision, marker="o")
    for k, x, y in zip(local_k, coverage, precision):
        plt.annotate(f"k={k}", (x, y))
    plt.xlabel("4/4 consensus coverage (%)")
    plt.ylabel("4/4 consensus precision (%)")
    save("08_consensus_precision_coverage.png")

    best_draft = next(row for row in drafts if row["local_draft_k"] == 4)
    reductions = [100 * best_draft["perfect_future_multistep_oracles"][str(k)]
                  ["perfect_model_forward_reduction_fraction"] for k in lags]
    plt.bar([str(k) for k in lags], reductions)
    plt.xlabel("Local steps K per global verifier")
    plt.ylabel("Optimistic global-forward reduction (%)")
    save("09_global_ep_invocation_oracle.png")

    replica = candidates["candidate_c_block_hot_replica"]
    budgets = [1, 2, 4, 8, 16, 32, 64, 128]
    plt.plot([replica[str(b)]["peak_hbm_budget_bytes"] / 2**20 for b in budgets],
             [100 * replica[str(b)]["optimistic_net_request_gain_fraction"] for b in budgets],
             marker="o")
    plt.xlabel("Replica HBM budget (MiB)")
    plt.ylabel("Optimistic net E2E gain (%)")
    save("10_hot_replica_budget.png")

    ms = sorted(int(m) for m in elastic["by_m"])
    for ep in eps:
        plt.plot(ms, [elastic["by_m"][str(m)]["latency_ms"][str(ep)] for m in ms],
                 marker=".", label=f"EP{ep}")
    plt.xscale("log", base=2)
    plt.xlabel("MoE input rows M")
    plt.ylabel("Layer MoE p50 (ms)")
    plt.legend()
    save("11_best_ep_degree_vs_shape.png")

    candidate_rows = [
        ("RLR-GV", 0.0, 0.0),
        ("K-step", 0.000052, 0.0),
        ("Replica", 0.153, 0.153),
        ("Elastic", 0.0, 0.0),
        ("Scope", 0.0, 0.0),
        ("Delta", 3.113, 0.0),
        ("Overlap", 6.225, 0.0),
        ("Near-tie", 0.59, 0.0),
        ("Layout", 0.10, 0.0),
    ]
    with (RESULTS / "candidate_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("candidate", "optimistic_or_impossible_cap_pct", "feasible_pct"))
        writer.writerows(candidate_rows)
    plt.bar([x[0] for x in candidate_rows], [x[1] for x in candidate_rows], label="cap")
    plt.bar([x[0] for x in candidate_rows], [x[2] for x in candidate_rows], label="feasible")
    plt.axhline(5, color="red", linestyle="--", label="implementation gate")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Request E2E (%)")
    plt.legend()
    save("12_candidate_oracle_comparison.png")

    plt.scatter([100 * quality[str(ep)]["overall_accuracy_median"] for ep in eps], walls)
    for ep, x, y in zip(eps, [100 * quality[str(ep)]["overall_accuracy_median"] for ep in eps], walls):
        plt.annotate(f"EP{ep}", (x, y))
    plt.xlabel("Bounded median quality (%)")
    plt.ylabel("8-request clean wall (s)")
    save("13_quality_latency_pareto.png")

    factors = [1.0, 0.75, 0.5, 0.25]
    r4 = replica["4"]
    denominator = candidates["request_wall_ms_denominator"]
    penalty = r4["proportional_critical_expert_penalty_ms"] + r4["copy_cost_ms"]
    replica_sensitivity = [100 * max(r4["perfect_future_saved_comm_ms"] * f - penalty, 0) /
                           denominator for f in factors]
    overlap_cap = 100 * candidates["candidate_h_overlap"][
        "free_all_dispatch_combine_request_cap_fraction"]
    delta_cap = 100 * candidates["candidate_f_temporal_delta"][
        "fp8_delta_absolute_request_cap_fraction"]
    rows = []
    for index, factor in enumerate(factors):
        rows.append((factor, replica_sensitivity[index], overlap_cap * factor, delta_cap * factor))
    with (RESULTS / "faster_runtime_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("communication_cost_factor", "replica_net_pct", "free_overlap_cap_pct", "fp8_delta_cap_pct"))
        writer.writerows(rows)
    plt.plot(factors, replica_sensitivity, marker="o", label="exact replica net")
    plt.plot(factors, [overlap_cap * f for f in factors], marker="o", label="free-comm cap")
    plt.plot(factors, [delta_cap * f for f in factors], marker="o", label="FP8-delta cap")
    plt.xlabel("Communication/runtime cost multiplier")
    plt.ylabel("Request E2E oracle/cap (%)")
    plt.legend()
    save("14_faster_runtime_sensitivity.png")

    plt.axis("off")
    plt.text(0.5, 0.60, "TEAM adaptation gate closed", ha="center", fontsize=16)
    plt.text(0.5, 0.45, "No independent vanilla candidate reached 5% E2E", ha="center")
    save("15_team_adaptation_gate.png")


if __name__ == "__main__":
    main()
