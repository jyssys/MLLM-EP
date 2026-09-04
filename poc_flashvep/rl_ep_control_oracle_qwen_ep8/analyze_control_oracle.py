#!/usr/bin/env python3
"""Offline decomposition for the Qwen3-30B-A3B EP8 control PoC.

This script deliberately keeps measured and counterfactual quantities separate.
KEEP and TEMP are paired real vLLM runs.  PERSIST is an EPLB plan plus a real
single-expert weight-broadcast measurement; it is *not* called an end-to-end
PERSIST action because placement/routing was not installed in vLLM.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_schedule(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    return pd.DataFrame(rows)[["wave", "rep", "condition", "target_tokens", "domains"]]


def load_drivers(root: Path, action: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("driver.dp_rank*.json")):
        rank = int(re.search(r"rank(\d+)", path.name).group(1))
        value = json.loads(path.read_text())
        for r in value.get("records", []):
            if r.get("measured"):
                rows.append({
                    "action": action,
                    "wave": int(r["wave"]),
                    "rep": int(r.get("rep", int(r["wave"]) // 6)),
                    "condition": r.get("condition"),
                    "dp_rank": rank,
                    "wall_ms": float(r.get("wall_ms", np.nan)),
                    "output_tokens": json.dumps(r.get("output_tokens")),
                })
    return pd.DataFrame(rows)


def add_stage(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    df = df.copy()
    df["moe_stage_sum_ms" + suffix] = (
        df["dispatch_max_ms" + suffix] + df["expert_max_ms" + suffix] + df["combine_max_ms" + suffix]
    )
    return df


def load_action_raw(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "action_raw").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                x["source"] = p.name
                rows.append(x)
    return pd.DataFrame(rows)


def load_migration(path: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((path / "migration_raw").glob("migration_rank*.json")):
        x = json.loads(p.read_text())
        x["path"] = str(p)
        rows.append(x)
    return pd.DataFrame(rows)


def eplb_summary(plan_path: Path) -> pd.DataFrame:
    x = json.loads(plan_path.read_text())
    rows = []
    for name, p in x["plans"].items():
        rows.append({
            "action": name,
            "replicas": p["num_replicas"],
            "physical_experts_per_gpu": p["physical_experts_per_gpu"],
            "baseline_rank_ratio": p["median_baseline_rank_ratio"],
            "predicted_rank_ratio": p["median_predicted_rank_ratio"],
            "predicted_count_proxy_reduction": p["median_ratio_reduction"],
        })
    return pd.DataFrame(rows)


def plot_results(pair: pd.DataFrame, drivers: pd.DataFrame, action_raw: pd.DataFrame, plan: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if not pair.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        vals = pair[["moe_stage_sum_keep", "moe_stage_sum_temp"]].to_numpy()
        ax.boxplot(vals, tick_labels=["KEEP", "TEMP measured"])
        ax.set_ylabel("max-rank dispatch+expert+combine stage (ms)")
        ax.set_title("Paired measured routed-MoE cost")
        fig.tight_layout(); fig.savefig(out / "keep_vs_temp_stage.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(pair["moe_stage_sum_reduction"].dropna() * 100, bins=24, color="#6b8e9e")
        ax.axvline(0, color="k", lw=.8); ax.set_xlabel("TEMP reduction vs KEEP (%)")
        ax.set_ylabel("invocation/layer pairs"); fig.tight_layout(); fig.savefig(out / "temp_reduction_distribution.png", dpi=160); plt.close(fig)
    if not drivers.empty:
        wall = drivers.groupby(["action", "condition"], as_index=False).wall_ms.median()
        fig, ax = plt.subplots(figsize=(10, 4)); wall.pivot(index="condition", columns="action", values="wall_ms").plot.bar(ax=ax)
        ax.set_ylabel("driver wall median (ms)"); fig.tight_layout(); fig.savefig(out / "driver_wall_by_condition.png", dpi=160); plt.close(fig)
    if not plan.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(plan)); width=.35
        ax.bar(x-width/2, plan.baseline_rank_ratio, width, label="linear baseline")
        ax.bar(x+width/2, plan.predicted_rank_ratio, width, label="EPLB predicted")
        ax.set_xticks(x, plan.action); ax.set_ylabel("median rank max/mean")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "eplb_rank_load_proxy.png", dpi=160); plt.close(fig)
    if not action_raw.empty:
        ar = action_raw.copy()
        ar["invalid_fraction"] = ar.invalid_slots / ar.assignments_before.clip(lower=1)
        fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(ar.invalid_fraction, bins=24, color="#c77966")
        ax.set_xlabel("TEMP invalid/dropped route-slot fraction"); ax.set_ylabel("router calls")
        fig.tight_layout(); fig.savefig(out / "temp_route_cost.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=Path, required=True)
    ap.add_argument("--temp", type=Path, required=True)
    ap.add_argument("--migration", type=Path, required=True)
    ap.add_argument("--prior", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)

    schedule = load_schedule(args.keep / "schedule.json")
    keep = pd.read_csv(args.keep / "invocation_metrics.csv")
    temp = pd.read_csv(args.temp / "invocation_metrics.csv")
    measured_cols = ["expert_max_ms", "expert_mean_ms", "dispatch_max_ms", "dispatch_mean_ms",
                     "combine_max_ms", "combine_mean_ms", "critical_path_max_ms", "critical_path_mean_ms",
                     "rank_ratio", "rank_cv"]
    keep = keep.rename(columns={c: c + "_keep" for c in measured_cols})
    temp = temp.rename(columns={c: c + "_temp" for c in measured_cols})
    keep["moe_stage_sum_keep"] = keep.dispatch_max_ms_keep + keep.expert_max_ms_keep + keep.combine_max_ms_keep
    temp["moe_stage_sum_temp"] = temp.dispatch_max_ms_temp + temp.expert_max_ms_temp + temp.combine_max_ms_temp
    join_cols = ["wave", "layer"]
    pair = keep.merge(temp, on=join_cols, suffixes=("", "_duplicate"), validate="one_to_one")
    # KEEP already carries the condition; schedule contributes the explicit
    # repetition and target token metadata without duplicating that column.
    pair = pair.merge(schedule[["wave", "rep", "target_tokens"]], on="wave", how="left")
    for metric in ["moe_stage_sum", "expert_max_ms", "dispatch_max_ms", "combine_max_ms", "critical_path_max_ms", "rank_ratio"]:
        pair[metric + "_reduction"] = 1 - pair[metric + "_temp"] / pair[metric + "_keep"].clip(lower=1e-9)
    pair.to_csv(out / "action_pair_metrics.csv", index=False)

    drivers = pd.concat([load_drivers(args.keep, "KEEP"), load_drivers(args.temp, "TEMP_BALANCE")], ignore_index=True)
    drivers.to_csv(out / "driver_latency.csv", index=False)
    driver_wide = drivers.pivot_table(index=["wave", "rep", "condition"], columns="action", values="wall_ms", aggfunc="median").reset_index()
    if {"KEEP", "TEMP_BALANCE"}.issubset(driver_wide.columns):
        driver_wide["temp_wall_reduction"] = 1 - driver_wide["TEMP_BALANCE"] / driver_wide["KEEP"]
    driver_wide.to_csv(out / "driver_latency_paired.csv", index=False)

    action_raw = load_action_raw(args.temp)
    if not action_raw.empty:
        action_raw["invalid_fraction"] = action_raw.invalid_slots / action_raw.assignments_before.clip(lower=1)
        action_raw.to_csv(out / "temp_action_raw_summary.csv", index=False)

    migration = load_migration(args.migration)
    migration.to_csv(out / "migration_timings.csv", index=False)
    plan = eplb_summary(args.plan)
    plan.to_csv(out / "eplb_plan_summary.csv", index=False)
    import shutil
    if args.plan.resolve() != (out / "placement_plan.json").resolve():
        shutil.copy2(args.plan, out / "placement_plan.json")

    # Aggregate logical expert loads and an EPLB-style count diagnostic from
    # the prior exact route histogram.  The prior run has repeated deterministic
    # waves; that limitation is retained rather than interpreted as general
    # future predictability.
    prior_temporal = pd.read_csv(args.prior / "temporal_metrics.csv")
    prior_temporal.to_csv(out / "hotspot_persistence.csv", index=False)

    def med(s): return float(np.nanmedian(s)) if len(s) else float("nan")

    # Action oracle: only A0/A1 are observed.  A2 receives a conservative cost
    # that includes the measured one-expert migration broadcast, so it cannot
    # win merely because the count-only plan predicts perfect packing.
    p = pair[["wave", "rep", "condition", "layer", "moe_stage_sum_keep", "moe_stage_sum_temp", "expert_max_ms_keep", "expert_max_ms_temp"]].copy()
    p["A0_KEEP_cost_ms"] = p.moe_stage_sum_keep
    p["A1_TEMP_cost_ms"] = p.moe_stage_sum_temp
    # EPLB rank-load proxy is layer-global; use small-plan reduction to adjust
    # only expert stage, and charge one measured broadcast once per condition.
    small_red = float(plan.loc[plan.action == "EPLB_SMALL", "predicted_count_proxy_reduction"].iloc[0]) if not plan.empty else 0.0
    mig_ms = float(migration.broadcast_two_tensors_ms.max()) if not migration.empty else float("nan")
    p["A2_PERSIST_predicted_cost_ms"] = p.moe_stage_sum_keep - p.expert_max_ms_keep * small_red + (mig_ms / 4.0 if np.isfinite(mig_ms) else 1e6)
    # A TEMP run that drops route slots is not a valid quality-preserving
    # action.  Keep its measured timing in the table, but make the safe action
    # oracle reject it rather than treating dropped work as a speedup.
    invalid_median = med(action_raw.invalid_fraction) if not action_raw.empty else 0.0
    raw_costs = p[["A0_KEEP_cost_ms", "A1_TEMP_cost_ms", "A2_PERSIST_predicted_cost_ms"]]
    raw_best = raw_costs.idxmin(axis=1).str.replace("_cost_ms", "", regex=False).map({"A0_KEEP": "KEEP", "A1_TEMP": "TEMP_BALANCE", "A2_PERSIST_predicted": "PERSIST_BALANCE"})
    p["A1_TEMP_safe_cost_ms"] = np.where(invalid_median <= 1e-12, p["A1_TEMP_cost_ms"], np.inf)
    costs = p[["A0_KEEP_cost_ms", "A1_TEMP_safe_cost_ms", "A2_PERSIST_predicted_cost_ms"]]
    p["best_action"] = costs.idxmin(axis=1).str.replace("_safe_cost_ms", "", regex=False).str.replace("_cost_ms", "", regex=False)
    p["best_action"] = p.best_action.map({"A0_KEEP": "KEEP", "A1_TEMP": "TEMP_BALANCE", "A2_PERSIST_predicted": "PERSIST_BALANCE"})
    p.to_csv(out / "action_oracle.csv", index=False)
    regime_map = {
        "balanced_2k": "balanced",
        "long_balanced": "persistent_domain_proxy",
        "long_math": "persistent_domain_proxy",
        "vision_proxy_long": "persistent_domain_proxy",
        "hetero_512_1k_2k_4k": "transient_mix_proxy",
        "short_mixed": "weak_or_transient_proxy",
    }
    episode = p.groupby("condition", as_index=False).agg(
        n_rows=("layer", "size"),
        keep_cost_ms=("A0_KEEP_cost_ms", "median"),
        temp_raw_cost_ms=("A1_TEMP_cost_ms", "median"),
        persist_pred_cost_ms=("A2_PERSIST_predicted_cost_ms", "median"),
        safe_best_action=("best_action", lambda x: x.mode().iloc[0]),
    )
    episode["regime"] = episode.condition.map(regime_map).fillna("unclassified")
    episode["evidence"] = "repeated deterministic condition proxy; no domain-switch trace"
    episode.to_csv(out / "dynamic_episode_action_summary.csv", index=False)

    summary = {
        "model": "Qwen3-30B-A3B",
        "topology": "TP2/DP4/EP8/PP1",
        "physical_gpus": list(range(8)),
        "experts": 128, "top_k": 8, "experts_per_gpu": 16,
        "stage0_from_previous_validated_capture": {
            "expert_ratio_median": 1.2866829393886752,
            "expert_ratio_p90": 1.4577590417126927,
            "expert_ratio_max": 1.8779217512690818,
        },
        "keep": {"n_pairs": int(len(keep)), "expert_ratio_median": float(keep.expert_max_ms_keep.div(keep.expert_mean_ms_keep.clip(lower=1e-9)).median()), "moe_stage_median_ms": med(keep.moe_stage_sum_keep)},
        "temp_measured": {
            "n_pairs": int(len(temp)),
            "expert_ratio_median": float(temp.expert_max_ms_temp.div(temp.expert_mean_ms_temp.clip(lower=1e-9)).median()),
            "moe_stage_median_ms": med(temp.moe_stage_sum_temp),
            "paired_moe_stage_reduction_median": med(pair.moe_stage_sum_reduction),
            "paired_expert_reduction_median": med(pair.expert_max_ms_reduction),
            "paired_dispatch_reduction_median": med(pair.dispatch_max_ms_reduction),
            "paired_combine_reduction_median": med(pair.combine_max_ms_reduction),
            "rank_ratio_reduction_median": med(pair.rank_ratio_reduction),
            "changed_fraction_median": med(action_raw.changed_fraction) if not action_raw.empty else None,
            "invalid_fraction_median": med(action_raw.invalid_fraction) if not action_raw.empty else None,
        },
        "driver_wall": {
            "keep_median_ms": med(drivers.loc[drivers.action == "KEEP", "wall_ms"]),
            "temp_median_ms": med(drivers.loc[drivers.action == "TEMP_BALANCE", "wall_ms"]),
            "temp_wall_reduction_median": med(driver_wide.temp_wall_reduction) if "temp_wall_reduction" in driver_wide else None,
        },
        "migration": {
            "ranks": int(len(migration)),
            "bytes_per_expert_broadcast": int(migration.bytes.max()) if not migration.empty else None,
            "broadcast_ms_median": med(migration.broadcast_two_tensors_ms),
            "broadcast_ms_max": float(migration.broadcast_two_tensors_ms.max()) if not migration.empty else None,
        },
        "eplb": plan.to_dict(orient="records"),
        "oracle": {
            "action_distribution": p.best_action.value_counts(normalize=True).to_dict(),
            "raw_timing_best_action_distribution": raw_best.value_counts(normalize=True).to_dict(),
            "temp_rejected_for_invalid_routes": bool(invalid_median > 1e-12),
            "n_rows": int(len(p)),
            "dynamic_gain_vs_keep": 0.0 if len(p) else None,
            "status": "conservative_counterfactual; PERSIST not installed in vLLM",
        },
        "persistence": {
            "adjacent_hot_expert_recurrence_median": float(prior_temporal.hot_expert_recurrence.median()) if not prior_temporal.empty else None,
            "n_wave_views": int(prior_temporal.n_waves.median()) if not prior_temporal.empty else None,
            "interpretation": "deterministic repeated schedule only; no domain-switch episode evidence",
        },
        "rl_policy": "NOT_RUN: action gate not passed",
        "gate": "NO_GO_FOR_RL_HEADROOM",
    }
    (out / "gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(pair, drivers, action_raw, plan, out / "figures")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
