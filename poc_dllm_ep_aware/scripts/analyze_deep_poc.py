#!/usr/bin/env python3
"""Build the auditable analysis tables and figures for the deep dLLM EP PoC.

The script deliberately keeps three evidence classes separate:

* clean request timing;
* observer-heavy structural traces;
* analytical upper bounds.

REFLEX and DES rows are marked as port-invalid for quality whenever the
paper-faithful policy is run outside the unavailable official execution
contract.  Their traces are useful for mapping an action to physical EP work,
but never become method speedup evidence.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "poc_dllm_ep_aware"
RUN = PROJECT / "results/deep_20260912_030000"
ANALYSIS = RUN / "analysis"
PLOTS = RUN / "plots"
PRIOR = ROOT / "poc_team_positive_ep2/results/team_positive_20260911_211456/analysis"


def load_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def reduction(new: float, old: float) -> float:
    return 100.0 * (1.0 - new / old)


def median_timing(path: Path) -> tuple[float, list[float], list[int]]:
    rows = load_json(path)
    values = [float(row["elapsed_ms"]) for row in rows]
    nfes = [int(row["nfe"]) for row in rows]
    return float(np.median(values)), values, nfes


def aggregate_trace(path: Path, method: str, ep: int) -> dict:
    rows = read_jsonl(path)
    total = sum(row["assignments"] for row in rows)
    remote = sum(row["remote_assignments"] for row in rows)
    calls = len(rows)
    output = {
        "method": method,
        "ep": ep,
        "calls": calls,
        "physical_rows": sum(row["physical_rows"] for row in rows),
        "assignments": total,
        "avg_k": sum(row["assignments"] for row in rows)
        / max(1, sum(row["physical_rows"] for row in rows)),
        "mean_unique_experts": float(np.mean([row["unique_experts"] for row in rows])),
        "mean_coreset_size": float(np.nanmean([
            np.nan if row["coreset_size"] is None else row["coreset_size"] for row in rows
        ])) if any(row["coreset_size"] is not None for row in rows) else np.nan,
        "remote_assignments": remote,
        "remote_fraction": remote / max(1, total),
        "remote_dispatch_bytes": sum(row["remote_dispatch_bytes"] for row in rows),
        "mean_fanout": float(np.mean([row["destination_rank_fanout"] for row in rows])),
        "mean_remote_fanout": float(np.mean([row["remote_fanout"] for row in rows])),
        "mean_critical_rank": float(np.mean([row["critical_rank_assignments"] for row in rows])),
    }
    for stage in ("router", "prepare", "dispatch", "expert", "combine", "moe"):
        output[f"{stage}_sum_ms"] = sum(row[f"{stage}_ms"] for row in rows)
        output[f"{stage}_median_ms"] = float(np.median([row[f"{stage}_ms"] for row in rows]))
    return output


def team_ep4() -> tuple[pd.DataFrame, dict]:
    team_root = RUN / "team"
    clean = []
    for restart in (1, 2, 4):
        b = load_json(team_root / f"ep4_clean2_r{restart}_baseline_rank0.json")
        t = load_json(team_root / f"ep4_clean2_r{restart}_team_rank0.json")
        clean.append({
            "restart": restart,
            "baseline_s": float(b["request_wall_s"]),
            "team_s": float(t["request_wall_s"]),
            "speedup_x": float(b["request_wall_s"] / t["request_wall_s"]),
        })
    clean_df = pd.DataFrame(clean)
    traces = {}
    for mode in ("baseline", "team"):
        data = load_json(team_root / f"ep4_trace_{mode}_rank0.json")
        rows = data["stage_rows"]
        traces[mode] = {
            "wall_ms": data["request_wall_s"] * 1000,
            "calls": len(rows),
            "physical_rows": sum(r["physical_rows"] for r in rows),
            "computed_rows": sum(r["computed_rows"] for r in rows),
            "assignments": sum(r["assignments"] for r in rows),
            "remote_assignments": sum(r["remote_assignments_from_source"] for r in rows),
            "remote_bytes": sum(r["remote_dispatch_bytes_hidden"] for r in rows),
            "fanout": float(np.mean([r["destination_rank_fanout"] for r in rows])),
            "remote_fanout": float(np.mean([r["remote_destination_rank_fanout"] for r in rows])),
            "mean_active_local_experts": float(np.mean([r["active_local_experts"] for r in rows])),
            **{f"{s}_sum_ms": sum(r[f"{s}_ms"] for r in rows)
               for s in ("router", "prepare", "dispatch", "expert", "combine", "moe")},
            **{f"{s}_median_ms": float(np.median([r[f"{s}_ms"] for r in rows]))
               for s in ("router", "prepare", "dispatch", "expert", "combine", "moe")},
        }
    return clean_df, traces


def score_gsm8k(outputs: Path) -> tuple[int, int]:
    """Conservative score using the final explicit numeric span.

    This is enough for the bounded four-item port-fidelity check; the report
    explicitly does not present it as a full benchmark reproduction.
    """
    import re
    correct = 0
    rows = load_json(outputs)
    for row in rows:
        text = row["generated"].replace(",", "")
        nums = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)
        prediction = nums[-1] if nums else None
        reference = str(row["reference"]).replace(",", "")
        if prediction is not None and float(prediction) == float(reference):
            correct += 1
    return correct, len(rows)


def fit_cost_models(surface: pd.DataFrame) -> pd.DataFrame:
    # Hold out entire route-shape configurations, not repetitions of shapes
    # seen during calibration.
    test = (surface["config_id"] % 5) == 0
    train = ~test
    target = surface["total_ms"]

    definitions = {
        "C0_bytes_only": ["total_transport_bytes"],
        "C1_bytes_fanout": ["total_transport_bytes", "fanout"],
        "C2_dispatch_expert_combine": [
            "remote_dispatch_bytes", "combine_bytes", "fanout", "assignments",
            "critical_rank_assignments", "active_experts_per_active_rank",
            "fragmentation_proxy"],
        "C3_critical_rank": ["critical_rank_assignments", "fanout",
                             "active_experts_per_active_rank", "fragmentation_proxy"],
    }
    rows = []
    for name, features in definitions.items():
        model = LinearRegression().fit(surface.loc[train, features], target[train])
        prediction = np.maximum(0, model.predict(surface.loc[test, features]))
        truth = target[test].to_numpy()
        rows.append({
            "model": name,
            "test_rows": len(truth),
            "rmse_ms": math.sqrt(mean_squared_error(truth, prediction)),
            "mape_pct": float(np.mean(np.abs(prediction - truth) / np.maximum(truth, .05)) * 100),
            "r2": r2_score(truth, prediction),
        })

    return pd.DataFrame(rows)


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    prior_summary = load_json(PRIOR / "summary.json")
    ep4_clean, team_trace = team_ep4()
    ep4_clean.to_csv(ANALYSIS / "team_ep4_clean_restarts.csv", index=False)
    pd.DataFrame.from_dict(team_trace, orient="index").reset_index(names="mode").to_csv(
        ANALYSIS / "team_ep4_structural_trace.csv", index=False)

    llada_root = RUN / "llada/traces_v2"
    trace_summaries = []
    trace_frames = []
    for ep in (2, 4):
        for method in ("vanilla", "reflex", "des"):
            directory = llada_root / f"{method}_ep{ep}"
            trace = directory / "rank0_trace.jsonl"
            summary = aggregate_trace(trace, method, ep)
            timing_median, timings, nfes = median_timing(directory / "rank0_timing.json")
            summary.update(request_median_ms=timing_median, request_timings_ms=json.dumps(timings),
                           nfe=json.dumps(nfes), quality_valid=(method == "vanilla"),
                           evidence="valid_reference_control" if method == "vanilla"
                           else "structural_only_port_quality_failed")
            trace_summaries.append(summary)
            frame = pd.DataFrame(read_jsonl(trace))
            frame["method"] = method
            frame["ep"] = ep
            frame["fragmentation_proxy"] = frame["local_active_experts"] / np.maximum(
                1, frame["critical_rank_assignments"])
            trace_frames.append(frame)
    llada = pd.DataFrame(trace_summaries)
    llada.to_csv(ANALYSIS / "llada_structural_characterization.csv", index=False)
    all_traces = pd.concat(trace_frames, ignore_index=True)
    all_traces.to_csv(ANALYSIS / "llada_action_trace_rank0.csv", index=False)

    raw_surface = pd.read_csv(RUN / "microbench/ep4_cost_surface_raw.csv")
    group_keys = [
        "config_id", "assignments", "fanout", "remote_fraction_actual",
        "active_experts_per_active_rank", "critical_rank_assignments",
        "remote_assignments", "remote_dispatch_bytes", "combine_bytes",
    ]
    # Same-device event spans are joined by repetition by taking the slowest
    # participating rank, then medians are taken over 30 repetitions.
    per_rep = raw_surface.groupby(group_keys + ["repetition"], as_index=False).agg({
        "dispatch_ms": "max", "expert_ms": "max", "combine_ms": "max", "total_ms": "max"
    })
    surface = per_rep.groupby(group_keys, as_index=False).median(numeric_only=True)
    surface["total_transport_bytes"] = 2 * surface.remote_dispatch_bytes + surface.combine_bytes
    surface["fragmentation_proxy"] = (
        surface.active_experts_per_active_rank / np.maximum(1, surface.critical_rank_assignments)
    )
    surface.to_csv(ANALYSIS / "ep4_cost_surface_medians.csv", index=False)
    cost_models = fit_cost_models(surface)
    cost_models.to_csv(ANALYSIS / "cost_model_validation.csv", index=False)

    quality_rows = []
    quality_paths = {
        "vanilla": RUN / "llada/clean_v2/vanilla_single_r1/outputs.json",
        "reflex": RUN / "llada/clean_v2/reflex_single_r1/outputs.json",
        "des": RUN / "llada/clean_v4/des_single_r1/outputs.json",
    }
    timing_paths = {
        "vanilla": RUN / "llada/clean_v2/vanilla_single_r1/rank0_timing.json",
        "reflex": RUN / "llada/clean_v2/reflex_single_r1/rank0_timing.json",
        "des": RUN / "llada/clean_v4/des_single_r1/rank0_timing.json",
    }
    for method in quality_paths:
        correct, total = score_gsm8k(quality_paths[method])
        med, values, nfes = median_timing(timing_paths[method])
        quality_rows.append({"method": method, "correct": correct, "total": total,
                             "accuracy": correct / total, "median_latency_ms": med,
                             "median_nfe": float(np.median(nfes)),
                             "quality_valid": method == "vanilla"})
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(ANALYSIS / "llada_bounded_quality.csv", index=False)

    baseline4 = llada[(llada.method == "vanilla") & (llada.ep == 4)].iloc[0]
    reflex4 = llada[(llada.method == "reflex") & (llada.ep == 4)].iloc[0]
    des4 = llada[(llada.method == "des") & (llada.ep == 4)].iloc[0]
    team_optional_fraction = (
        team_trace["team"]["assignments"] - team_trace["team"]["calls"] * 32 * 8
    ) / team_trace["team"]["assignments"]
    # The validated fused EP2 residual gives the strongest clean all-EP-comm
    # share available.  Applying only the optional-row fraction is generous.
    team_all_comm_pct = float(pd.read_csv(PRIOR / "candidate_oracles.csv")
                              .query("candidate == 'all EP communication elimination'")
                              .iloc[0].perfect_oracle_e2e_pct)
    team_width_oracle = team_all_comm_pct * team_optional_fraction
    # Compare one physical token row, not whole trajectories: REFLEX/DES ports
    # changed NFE because their quality contract failed, so trajectory totals
    # are not a valid action-level comparison.
    reflex_pair_reduction = 1 - reflex4.avg_k / baseline4.avg_k
    llada_moe_share = baseline4.moe_sum_ms / baseline4.request_median_ms * 100
    reflex_oracle = llada_moe_share * reflex_pair_reduction
    comm_share = (baseline4.dispatch_sum_ms + baseline4.combine_sum_ms) / baseline4.request_median_ms * 100
    des_topology_oracle = comm_share / 4.0
    previous_early = float(pd.read_csv(PRIOR / "candidate_oracles.csv")
                           .query("candidate == 'perfect early speculative-branch commitment'")
                           .iloc[0].perfect_oracle_e2e_pct)
    candidate_rows = [
        ("A_cost_aware_reranking", team_width_oracle,
         "TEAM optional-row communication upper bound; future utility unknown", "weak/kill"),
        ("B_EP_budgeted_work", reflex_oracle,
         "REFLEX pair-removal proportional MoE upper bound; quality-invalid port", "kill"),
        ("C_marginal_utility_per_EP_cost", max(team_width_oracle, reflex_oracle),
         "best measured action-family bound before utility loss", "kill"),
        ("D_topology_aware_action_selection", des_topology_oracle,
         "generous one-of-four-rank communication elimination; DES fanout stayed four", "kill"),
        ("E_TEAM_EP_aware_speculation_width", team_width_oracle,
         "assumes unchanged NFE while removing every extra speculative assignment", "kill"),
        ("F_joint_decode_EP_future_oracle", previous_early,
         "prior measured perfect early-commitment oracle; includes future knowledge", "weak"),
        ("perfect_all_EP_communication", max(team_all_comm_pct, comm_share),
         "physically impossible ceiling, not a candidate", "ceiling"),
    ]
    candidates = pd.DataFrame(candidate_rows, columns=[
        "candidate", "perfect_e2e_oracle_pct", "evidence_boundary", "gate"
    ])
    candidates.to_csv(ANALYSIS / "candidate_oracles.csv", index=False)

    sensitivity = pd.DataFrame([
        {"communication_cost_multiplier": factor,
         "candidate": "TEAM_EP_aware_width",
         "oracle_e2e_pct": team_width_oracle * factor}
        for factor in (1.0, .75, .5, .25)
    ])
    sensitivity.to_csv(ANALYSIS / "communication_sensitivity.csv", index=False)

    characterization = []
    # TEAM is the only method with a faithful positive-control chain.
    characterization.extend([
        {"method": "TEAM", "model": "SDAR-30B-A3B", "dataset": "bounded GSM8K/HumanEval",
         "setting": "single", "quality": "bounded-positive", "latency_ms": np.nan,
         "speedup_vs_vanilla": prior_summary["stage_a"]["median_speedup_x"],
         "algorithmic_proxy_delta_pct": -prior_summary["stage_a"]["trace_nfe_reduction_pct"],
         "expert_pairs_delta_pct": prior_summary["stage_a"]["trace_assignment_change_pct"],
         "remote_assignments_delta_pct": np.nan, "remote_bytes_delta_pct": np.nan,
         "fanout_delta": np.nan, "evidence": "official positive control"},
        {"method": "TEAM", "model": "SDAR-30B-A3B", "dataset": "GSM8K anchor",
         "setting": "EP2", "quality": "trajectory-preserved", "latency_ms": 3337.717,
         "speedup_vs_vanilla": prior_summary["stage_b"]["fused_speedup_x"],
         "algorithmic_proxy_delta_pct": -41.67, "expert_pairs_delta_pct": 31.5,
         "remote_assignments_delta_pct": 33.61, "remote_bytes_delta_pct": 33.61,
         "fanout_delta": 0.0, "evidence": "true EP2, fused local experts"},
        {"method": "TEAM", "model": "SDAR-30B-A3B", "dataset": "GSM8K anchor",
         "setting": "EP4", "quality": "prefix-parity-only", "latency_ms": float(ep4_clean.team_s.median() * 1000),
         "speedup_vs_vanilla": float(np.median(ep4_clean.speedup_x)),
         "algorithmic_proxy_delta_pct": reduction(team_trace["team"]["calls"], team_trace["baseline"]["calls"]) * -1,
         "expert_pairs_delta_pct": pct(team_trace["team"]["assignments"], team_trace["baseline"]["assignments"]),
         "remote_assignments_delta_pct": pct(team_trace["team"]["remote_assignments"], team_trace["baseline"]["remote_assignments"]),
         "remote_bytes_delta_pct": pct(team_trace["team"]["remote_bytes"], team_trace["baseline"]["remote_bytes"]),
         "fanout_delta": team_trace["team"]["fanout"] - team_trace["baseline"]["fanout"],
         "evidence": "two clean restarts; third NCCL hang excluded"},
    ])
    for method in ("reflex", "des"):
        q = quality[quality.method == method].iloc[0]
        v = quality[quality.method == "vanilla"].iloc[0]
        characterization.append({
            "method": method.upper(), "model": "LLaDA-MoE-7B-A1B", "dataset": "GSM8K-4",
            "setting": "single", "quality": f"port-invalid {int(q.correct)}/{int(q.total)} vs {int(v.correct)}/{int(v.total)}",
            "latency_ms": q.median_latency_ms, "speedup_vs_vanilla": v.median_latency_ms / q.median_latency_ms,
            "algorithmic_proxy_delta_pct": np.nan, "expert_pairs_delta_pct": np.nan,
            "remote_assignments_delta_pct": np.nan, "remote_bytes_delta_pct": np.nan,
            "fanout_delta": np.nan, "evidence": "paper-faithful semantic port; official runtime unavailable"})
        for ep in (2, 4):
            row = llada[(llada.method == method) & (llada.ep == ep)].iloc[0]
            base = llada[(llada.method == "vanilla") & (llada.ep == ep)].iloc[0]
            proxy = reduction(row.avg_k, base.avg_k) if method == "reflex" else reduction(38, 64)
            characterization.append({
                "method": method.upper(), "model": "LLaDA-MoE-7B-A1B", "dataset": "GSM8K-1",
                "setting": f"EP{ep}", "quality": "invalid—structural trace only",
                "latency_ms": row.request_median_ms, "speedup_vs_vanilla": base.request_median_ms / row.request_median_ms,
                "algorithmic_proxy_delta_pct": -proxy,
                "expert_pairs_delta_pct": pct(row.avg_k, base.avg_k),
                "remote_assignments_delta_pct": pct(
                    row.remote_assignments / row.physical_rows,
                    base.remote_assignments / base.physical_rows),
                "remote_bytes_delta_pct": pct(
                    row.remote_dispatch_bytes / row.physical_rows,
                    base.remote_dispatch_bytes / base.physical_rows),
                "fanout_delta": row.mean_fanout - base.mean_fanout,
                "evidence": "reference assignment A2A; excluded from speedup claims"})
    char = pd.DataFrame(characterization)
    char.to_csv(ANALYSIS / "cross_method_characterization.csv", index=False)

    # 1 TEAM speedup vs EP degree.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 4], [prior_summary["stage_a"]["median_speedup_x"],
                        prior_summary["stage_b"]["fused_speedup_x"],
                        np.median(ep4_clean.speedup_x)], marker="o")
    ax.axhline(1, color="black", lw=.8)
    ax.set(xlabel="EP degree", ylabel="TEAM speedup vs SDAR", title="TEAM scaling")
    save_plot(PLOTS / "01_team_speedup_vs_ep.png")

    # 2/3 structural port speedups, conspicuously marked invalid.
    for idx, method in enumerate(("REFLEX", "DES"), start=2):
        subset = char[char.method == method]
        xs = [1, 2, 4]
        ys = [float(subset[subset.setting == ("single" if ep == 1 else f"EP{ep}")].speedup_vs_vanilla.iloc[0])
              for ep in xs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="x", ls="--", color="tab:gray")
        ax.text(.5, .88, "PORT QUALITY INVALID—NOT SPEEDUP EVIDENCE", transform=ax.transAxes,
                ha="center", color="crimson", weight="bold")
        ax.axhline(1, color="black", lw=.8)
        ax.set(xlabel="EP degree", ylabel="apparent speedup", title=f"{method} structural port")
        save_plot(PLOTS / f"0{idx}_{method.lower()}_speedup_vs_ep_invalid.png")

    # 4 bounded quality.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(quality.method.str.upper(), quality.accuracy * 100,
           color=["tab:blue", "tab:red", "tab:red"])
    ax.set(ylabel="GSM8K-4 exact answer (%)", title="Bounded port-fidelity gate")
    save_plot(PLOTS / "04_accuracy_delta.png")

    # 5-7 physical EP4 deltas.
    metric_specs = [
        ("remote_assignments", "05_remote_assignments.png"),
        ("remote_dispatch_bytes", "06_remote_bytes.png"),
        ("mean_fanout", "07_rank_fanout.png"),
    ]
    ep4_struct = llada[llada.ep == 4].copy()
    for metric, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(ep4_struct.method.str.upper(), ep4_struct[metric])
        ax.set(title=f"LLaDA EP4 structural trace: {metric}", ylabel=metric)
        ax.text(.5, .93, "REFLEX/DES quality-invalid", transform=ax.transAxes,
                ha="center", color="crimson")
        save_plot(PLOTS / filename)

    # 8 algorithmic proxy versus physical outcome.
    plot_rows = char[(char.setting == "EP4")].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in plot_rows.iterrows():
        ax.scatter(-row.algorithmic_proxy_delta_pct, row.remote_bytes_delta_pct, s=70)
        ax.annotate(row.method, (-row.algorithmic_proxy_delta_pct, row.remote_bytes_delta_pct))
    ax.axhline(0, color="black", lw=.8); ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="algorithmic proxy reduction (%)", ylabel="remote bytes change (%)",
           title="Proxy improvement does not imply EP improvement")
    save_plot(PLOTS / "08_proxy_vs_physical_EP.png")

    # 9 EP4 stage composition.
    stages = ["router", "prepare", "dispatch", "expert", "combine"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(ep4_struct))
    for stage in stages:
        values = ep4_struct[f"{stage}_sum_ms"].to_numpy()
        ax.bar(ep4_struct.method.str.upper(), values, bottom=bottom, label=stage)
        bottom += values
    ax.legend(ncol=3, fontsize=8)
    ax.set(ylabel="trace sum (ms)", title="LLaDA EP4 structural stage totals")
    save_plot(PLOTS / "09_ep4_stage_breakdown.png")

    # 10 action utility versus physical cost.
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in plot_rows.iterrows():
        ax.scatter(-row.algorithmic_proxy_delta_pct, row.remote_assignments_delta_pct, s=70)
        ax.annotate(row.method, (-row.algorithmic_proxy_delta_pct, row.remote_assignments_delta_pct))
    ax.set(xlabel="native proxy reduction (%)", ylabel="remote assignment change (%)",
           title="Action proxy vs EP action cost")
    save_plot(PLOTS / "10_action_utility_vs_ep_cost.png")

    # 11 predictor accuracy.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cost_models.model, cost_models.rmse_ms)
    ax.tick_params(axis="x", rotation=25)
    ax.set(ylabel="held-out RMSE (ms/call)", title="Interpretable EP cost models")
    save_plot(PLOTS / "11_cost_predictor_accuracy.png")

    # 12 candidate oracles.
    cand_plot = candidates[candidates.candidate != "perfect_all_EP_communication"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(cand_plot.candidate, cand_plot.perfect_e2e_oracle_pct)
    ax.axvline(5, color="orange", ls="--"); ax.axvline(12, color="red", ls="--")
    ax.set(xlabel="perfect request E2E oracle (%)", title="Candidate competition")
    save_plot(PLOTS / "12_candidate_oracles.png")

    # 13 bounded quality-latency Pareto.
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in quality.iterrows():
        ax.scatter(row.median_latency_ms, row.accuracy * 100, s=70,
                   color="tab:blue" if row.method == "vanilla" else "tab:red")
        ax.annotate(row.method.upper(), (row.median_latency_ms, row.accuracy * 100))
    ax.set(xlabel="single-GPU median latency (ms)", ylabel="GSM8K-4 accuracy (%)",
           title="No quality-safe REFLEX/DES Pareto point in this port")
    save_plot(PLOTS / "13_quality_latency_pareto.png")

    # 14 communication sensitivity.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sensitivity.communication_cost_multiplier, sensitivity.oracle_e2e_pct, marker="o")
    ax.invert_xaxis()
    ax.set(xlabel="communication cost multiplier", ylabel="TEAM-width oracle (%)",
           title="Candidate durability under faster communication")
    save_plot(PLOTS / "14_communication_sensitivity.png")

    # 15 explicitly show why no live candidate was built.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["best perfect oracle", "live prototype"], [previous_early, 0],
           color=["tab:orange", "tab:gray"])
    ax.axhline(8, color="red", ls="--", label="prototype gate")
    ax.text(1, .4, "not implemented\n(oracle below gate)", ha="center")
    ax.legend()
    ax.set(ylabel="request E2E improvement (%)", title="Original vs EP-aware policy")
    save_plot(PLOTS / "15_original_vs_ep_aware.png")

    summary = {
        "final_label": "CHARACTERIZATION-ONLY",
        "team_single_median_speedup_x": prior_summary["stage_a"]["median_speedup_x"],
        "team_ep2_fused_speedup_x": prior_summary["stage_b"]["fused_speedup_x"],
        "team_ep4_clean_restart_speedups_x": ep4_clean.speedup_x.tolist(),
        "team_ep4_median_speedup_x": float(np.median(ep4_clean.speedup_x)),
        "team_ep4_assignment_change_pct": pct(team_trace["team"]["assignments"], team_trace["baseline"]["assignments"]),
        "team_ep4_remote_assignment_change_pct": pct(team_trace["team"]["remote_assignments"], team_trace["baseline"]["remote_assignments"]),
        "team_ep4_remote_bytes_change_pct": pct(team_trace["team"]["remote_bytes"], team_trace["baseline"]["remote_bytes"]),
        "team_ep4_fanout_baseline": team_trace["baseline"]["fanout"],
        "team_ep4_fanout_team": team_trace["team"]["fanout"],
        "reflex_ep4_avgk_reduction_pct": 100 * reflex_pair_reduction,
        "reflex_ep4_remote_assignment_reduction_pct": reduction(
            reflex4.remote_assignments / reflex4.physical_rows,
            baseline4.remote_assignments / baseline4.physical_rows),
        "reflex_ep4_fanout": reflex4.mean_fanout,
        "des_active_coreset_size": 38,
        "des_ep4_fanout": des4.mean_fanout,
        "best_candidate": "F_joint_decode_EP_future_oracle",
        "best_candidate_perfect_e2e_oracle_pct": previous_early,
        "prototype_implemented": False,
        "prototype_reason": "best perfect oracle below 8% promotion gate",
        "evidence_warnings": [
            "TEAM EP4 uses exact NCCL reference A2A with fused local experts, not DeepEP production dispatch",
            "TEAM EP4 has three successful clean restarts; a fourth attempt hung in NCCL and was excluded",
            "REFLEX/DES official code was unavailable; minimal semantic ports failed bounded quality and are structural-only",
            "LLaDA EP2/EP4 trajectories are not numerically invariant, so cross-degree latency is not a scaling claim",
        ],
    }
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
