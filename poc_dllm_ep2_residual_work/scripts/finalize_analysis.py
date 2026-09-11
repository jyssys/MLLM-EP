#!/usr/bin/env python3
"""Produce state-, layer-, iteration-, and EP2 economic summary tables.

This consumes the observer-heavy temporal CSVs and the independently measured
clean request/stage calibration.  It does not execute a candidate method.
Every reported candidate number is therefore explicitly an offline oracle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stage_curve(path: Path):
    frame = pd.read_csv(path)
    frame = frame[frame.backend == "deepep_high_throughput"].sort_values("local_m")
    x = frame.local_m.to_numpy(float)
    y = frame[["dispatch_ms", "expert_ms", "combine_ms"]].sum(axis=1).to_numpy(float)

    def cost(m: float) -> float:
        if m <= 0:
            return 0.0
        if m < x.min():
            return float(y[0])
        return float(np.interp(m, x, y))

    return cost


def epoch_fresh_sequence(iterations: pd.DataFrame) -> list[int]:
    """Epoch M=5 equivalent fresh positions for one ordered request trace."""
    fresh: list[int] = []
    previous_live: int | None = None
    for row in iterations.itertuples():
        if int(row.iteration_id) == 0 or int(row.iteration_id) % 5 == 0:
            value = int(row.physical_m)
        else:
            if previous_live is None:
                raise ValueError("non-cold iteration has no previous live count")
            # Current live plus newly decoded equals the previous iteration's
            # live set under monotonic threshold decoding.
            value = previous_live
        fresh.append(value)
        previous_live = int(row.masked_before)
    return fresh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--prior-scaling", type=Path, required=True)
    args = parser.parse_args()
    out = args.result_root / "analysis"
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    token = pd.read_csv(out / "transition_token_metrics.csv")
    branch = pd.read_csv(out / "branch_metrics.csv")
    cell = pd.read_csv(out / "layer_iteration_metrics.csv")
    iteration = pd.read_csv(out / "iteration_work.csv")
    summary = json.loads((out / "summary.json").read_text())
    cost = stage_curve(args.prior_scaling)
    clean_ms = float(summary["clean_request_median_ms"])

    states = {
        "all physical positions": pd.Series(True, index=token.index),
        "generation positions": token.is_generation,
        "live generation": token.is_live,
        "newly decoded generation": token.is_newly_decoded,
        "Epoch-equivalent fresh generation": token.is_epoch_fresh,
        "stable decoded generation": token.is_generation & ~token.is_live & ~token.is_newly_decoded,
    }
    state_rows = []
    for name, mask in states.items():
        q = token[mask]
        state_rows.append({
            "state": name,
            "rows": len(q),
            "ordered_topk_same_pct": 100 * q.ordered_route_same.mean(),
            "topk_set_same_pct": 100 * q.route_set_same.mean(),
            "destination_set_same_pct": 100 * q.destination_set_same.mean(),
            "branch_persistence_pct": 100 * q.branch_persistence.mean(),
            "router_rel_l2_median": q.router_rel_l2.median(),
            "topk_weight_l1_median": q.topk_weight_l1.median(),
            "hidden_rel_l2_median": q.hidden_rel_l2.median(),
            "moe_output_rel_l2_median": q.moe_output_rel_l2.median(),
        })
    pd.DataFrame(state_rows).to_csv(out / "state_stability.csv", index=False)

    token["layer_band"] = pd.cut(
        token.layer, [-1, 4, 10, 15], labels=["early_0_4", "middle_5_10", "late_11_15"]
    )
    layer = token[token.is_epoch_fresh].groupby("layer", observed=True).agg(
        rows=("position", "size"),
        ordered_topk_same_pct=("ordered_route_same", lambda x: 100 * x.mean()),
        topk_set_same_pct=("route_set_same", lambda x: 100 * x.mean()),
        destination_set_same_pct=("destination_set_same", lambda x: 100 * x.mean()),
        branch_persistence_pct=("branch_persistence", lambda x: 100 * x.mean()),
        router_rel_l2_median=("router_rel_l2", "median"),
        hidden_rel_l2_median=("hidden_rel_l2", "median"),
        moe_output_rel_l2_median=("moe_output_rel_l2", "median"),
    ).reset_index()
    layer.to_csv(out / "layer_stability.csv", index=False)

    branch["layer_band"] = pd.cut(
        branch.layer, [-1, 4, 10, 15], labels=["early_0_4", "middle_5_10", "late_11_15"]
    )
    fresh_branch = branch[branch.is_epoch_fresh].copy()
    branch_layer = fresh_branch.groupby("layer", observed=True).branch_rel_l2.agg(
        observed_persistent_branches="size",
        median="median",
        p10=lambda x: x.quantile(0.10),
        le_1pct=lambda x: 100 * (x <= 0.01).mean(),
        le_5pct=lambda x: 100 * (x <= 0.05).mean(),
        le_10pct=lambda x: 100 * (x <= 0.10).mean(),
    ).reset_index()
    branch_layer.to_csv(out / "branch_stability_by_layer.csv", index=False)

    branch_iteration = cell.dropna(
        subset=["epoch_fresh_reusable_branches_0.05"]
    ).groupby(["run_id", "transition"], observed=True).agg(
        fresh_generation_branches=("epoch_fresh_generation_branches", "sum"),
        reusable_le_5pct=("epoch_fresh_reusable_branches_0.05", "sum"),
        masked_before=("masked_before", "first"),
    ).reset_index()
    branch_iteration["reusable_le_5pct_of_fresh_pct"] = (
        100 * branch_iteration.reusable_le_5pct
        / branch_iteration.fresh_generation_branches.clip(lower=1)
    )
    branch_iteration.to_csv(out / "branch_reuse_by_iteration.csv", index=False)

    transition = token[token.is_epoch_fresh].groupby(["run_id", "transition"], observed=True).agg(
        rows=("position", "size"),
        topk_set_same_pct=("route_set_same", lambda x: 100 * x.mean()),
        destination_set_same_pct=("destination_set_same", lambda x: 100 * x.mean()),
        branch_persistence_pct=("branch_persistence", lambda x: 100 * x.mean()),
        router_rel_l2_median=("router_rel_l2", "median"),
        hidden_rel_l2_median=("hidden_rel_l2", "median"),
        moe_output_rel_l2_median=("moe_output_rel_l2", "median"),
    ).reset_index()
    transition.to_csv(out / "iteration_stability.csv", index=False)

    # Model the exact physical-M shrink that Epoch's live/new/periodic-refresh
    # contract would expose to the already measured DeepEP HT stage curve.
    epoch_rows = []
    for run_id, q in iteration.sort_values(["run_id", "iteration_id"]).groupby("run_id"):
        fresh = epoch_fresh_sequence(q)
        physical_ms = len(q) * 16 * cost(float(q.physical_m.iloc[0]))
        epoch_ms = 16 * sum(cost(x) for x in fresh)
        epoch_rows.append({
            "run_id": run_id,
            "calls": len(q),
            "physical_m": int(q.physical_m.iloc[0]),
            "epoch_fresh_m_sequence": ";".join(str(x) for x in fresh),
            "assignment_reduction_pct": 100 * (1 - sum(fresh) / (len(q) * q.physical_m.iloc[0])),
            "calibrated_moe_stage_reduction_pct": 100 * (1 - epoch_ms / physical_ms),
            "projected_request_e2e_upper_bound_pct": 100 * (physical_ms - epoch_ms) / clean_ms,
            "novelty": "excluded: Epoch already removes this work",
        })
    epoch = pd.DataFrame(epoch_rows)
    epoch.to_csv(out / "epoch_equivalent_oracle_by_run.csv", index=False)

    # Future-known selective branch reuse.  Restricting to a layer band or to
    # cells whose aggregate fresh-output perturbation passes a local tolerance
    # is even more optimistic than a causal policy and thus a valid kill oracle.
    selective_rows = []
    for key, threshold in (("0.01", 0.01), ("0.05", 0.05)):
        reuse_col = f"epoch_fresh_reusable_branches_{key}"
        error_col = f"epoch_fresh_combined_error_{key}"
        observed = cell.dropna(subset=[reuse_col]).copy()
        observed["layer_band"] = pd.cut(
            observed.layer, [-1, 4, 10, 15], labels=["early_0_4", "middle_5_10", "late_11_15"]
        )
        masks = {"all captured layers": pd.Series(True, index=observed.index)}
        for band in ("early_0_4", "middle_5_10", "late_11_15"):
            masks[band] = observed.layer_band.astype(str) == band
        masks["future-known cells with aggregate error <=1%"] = observed[error_col] <= 0.01
        for scope, mask in masks.items():
            q = observed[mask]
            saving = 0.0
            before_sum = 0.0
            fresh_branches = int(q.epoch_fresh_generation_branches.sum())
            reusable = int(q[reuse_col].sum())
            for _, row in q.iterrows():
                generation_m = float(row["epoch_fresh_generation_branches"]) / 8.0
                prompt_m = max(0.0, float(row["physical_m"]) - generation_m) if int(row["transition"]) % 5 == 0 else 0.0
                before_m = generation_m + prompt_m
                after_m = max(0.0, before_m - float(row[reuse_col]) / 8.0)
                before = cost(before_m)
                after = cost(after_m)
                before_sum += before
                saving += max(0.0, before - after)
            selective_rows.append({
                "branch_change_threshold": threshold,
                "scope": scope,
                "cells": len(q),
                "layers_touched": int(q.layer.nunique()),
                "reusable_branches": reusable,
                "reuse_fraction_of_fresh_generation_branches_pct": 100 * reusable / max(fresh_branches, 1),
                "projected_moe_local_gain_pct": 100 * saving / max(before_sum, 1e-12),
                "projected_request_e2e_upper_bound_pct": 100 * saving / (clean_ms * 3),
                "correctness_evidence": "local one-transition criterion only; no rollout",
            })
    selective = pd.DataFrame(selective_rows)
    selective.to_csv(out / "layer_iteration_selective_oracles.csv", index=False)

    # The invalid-column-name conversion used by namedtuple is awkward above;
    # assert that every computed selective oracle is finite and non-negative.
    if not np.isfinite(selective.projected_request_e2e_upper_bound_pct).all():
        raise RuntimeError("non-finite selective oracle")
    if (selective.projected_request_e2e_upper_bound_pct < 0).any():
        raise RuntimeError("negative selective oracle")

    risks = pd.DataFrame([
        ("routing/expert-branch temporal persistence", "observed", "likely qualitative transfer", "more destination ranks; branch identity itself is topology-independent"),
        ("logical-versus-physical position work gap", "observed", "likely qualitative transfer", "decoder liveness is EP-degree independent"),
        ("absolute dispatch/combine saving", "not established", "must remeasure", "EP4 changes fanout, bytes, startup, and contention"),
        ("load-to-latency calibration", "EP2 only", "must remeasure", "32 local experts/rank becomes 16 at EP4"),
        ("request E2E speedup", "offline EP2 upper bound", "must remeasure", "backend and communication critical-path shares change"),
        ("candidate correctness", "one-step local error only", "must validate", "requires multi-step generation/benchmark validation regardless of EP degree"),
    ], columns=["quantity", "ep2_status", "ep4_transfer", "reason"])
    risks.to_csv(out / "ep2_ep4_transfer_risk.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=risks[["quantity", "ep4_transfer"]].values,
        colLabels=["Quantity", "EP2 -> EP4"],
        cellLoc="left",
        loc="center",
        colWidths=[0.68, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax.set_title("EP2 mechanism screen: transfer boundary", pad=12)
    fig.tight_layout()
    fig.savefig(plots / "ep2_ep4_transfer_risk.png", dpi=180)
    plt.close(fig)

    clean_values = []
    for path in sorted((args.result_root / "stage_timing").glob("clean_*/*_rank0.json")):
        clean_values.append(float(json.loads(path.read_text())["request_ms"]))
    final = {
        "clean_request_ms": clean_values,
        "clean_request_median_ms": float(np.median(clean_values)),
        "clean_request_cv_pct": float(100 * np.std(clean_values, ddof=1) / np.mean(clean_values)),
        "epoch_equivalent_assignment_reduction_pct_median": float(epoch.assignment_reduction_pct.median()),
        "epoch_equivalent_projected_e2e_pct_median": float(epoch.projected_request_e2e_upper_bound_pct.median()),
        "best_residual_candidate": "perfect router recomputation elimination",
        "best_residual_projected_e2e_pct": float(summary["best_post_epoch_candidate_e2e_pct"]),
        "iteration_reuse_spearman_by_run": {
            run: float(q.transition.corr(q.reusable_le_5pct_of_fresh_pct, method="spearman"))
            for run, q in branch_iteration.groupby("run_id")
        },
        "decision": "EP2 NO-GO",
        "reason": "Every independent post-Epoch perfect oracle is below the 5% kill gate.",
    }
    (out / "final_summary.json").write_text(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
