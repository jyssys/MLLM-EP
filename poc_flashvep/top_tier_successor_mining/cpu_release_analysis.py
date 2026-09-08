"""Consolidate already measured evidence after user-requested GPU release.

No torch/CUDA import, no new model execution. Missing E2E evidence remains null.
Incomplete threshold search is never labelled a completed official optimum.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--task", type=Path, required=True)
    args = ap.parse_args()
    root = args.results
    out = root / "analysis/cpu_release_checkpoint"
    out.mkdir(parents=True, exist_ok=True)
    quality = []
    screens = {
        "sere_gqa64": ("SERE", "EXPLORATORY_FP32_GRAM_B1"),
        "sere_chart64": ("SERE", "EXPLORATORY_FP32_GRAM_B1"),
        "sere_chart128_phase": ("SERE", "EXPLORATORY_FP32_GRAM_PHASE_CONTROL_B1"),
        "sere_chart128_b16_partial": ("SERE", "EXPLORATORY_FP32_GRAM_B16"),
        "modes_chart128_pilot": ("MoDES", "PILOT_CALIBRATION_NOT_OFFICIAL_FULL_FRONTIER"),
    }
    for name, (method, status) in screens.items():
        for row in json.loads((root / "analysis" / name / "summary.json").read_text()):
            quality.append({"method": method, "screen": name, "policy": row["policy"],
                "dataset": row["dataset"], "requests": row["n"],
                "accuracy_percent": row["accuracy_percent"],
                "paired_quality_delta_pp": row["paired_delta_pp"],
                "quality_ci_low_pp": row["paired_delta_95ci_pp"][0],
                "quality_ci_high_pp": row["paired_delta_95ci_pp"][1],
                "request_e2e_ms": None, "quality_matched_e2e_gain_percent": None,
                "efficiency_status": "NOT_MEASURED_CLEANLY", "evidence_status": status})
    pd.DataFrame(quality).to_csv(args.task / "QUALITY_EFFICIENCY_PARETO.csv", index=False)

    folder = root / "quality/modes_frontier1024_grid100"
    raw = [json.loads(s) for s in (folder / "evaluations.jsonl").read_text().splitlines()]
    df = pd.DataFrame(raw)
    unique = df.drop_duplicates(["tau_text", "tau_vision"], keep="last")
    assert not unique[["kl", "skip_fraction"]].isna().any().any()
    partial = {"raw_records": len(df), "unique_evaluated_points": len(unique),
               "completed_search": (folder / "completed.json").exists(),
               "finished_targets": json.loads((folder / "frontier.json").read_text()),
               "best_observed_partial_candidates": []}
    for target in (.7, .85):
        feasible = unique[unique.skip_fraction >= target]
        best = feasible.loc[feasible.kl.idxmin()]
        partial["best_observed_partial_candidates"].append({"target_skip": target,
            "tau_text": best.tau_text, "tau_vision": best.tau_vision,
            "actual_skip": best.skip_fraction, "kl": best.kl,
            "scope": "BEST_RECORDED_POINT_ONLY_NOT_UNFINISHED_SEARCH_OPTIMUM"})
    (out / "modes_partial_frontier.json").write_text(json.dumps(partial, indent=2))
    unique.to_csv(out / "modes_measured_points.csv", index=False)

    plans = pd.read_csv(root / "analysis/libra_supplement_full96/prediction_plan_results.csv")
    keys = ["requests", "layer", "tokens_per_source"]
    assert not plans.duplicated(keys + ["modality"]).any()
    paired = plans[plans.modality == "vision"].merge(plans[plans.modality == "text"],
        on=keys, suffixes=("_vision", "_text"), validate="one_to_one")
    for column in ("topk_recall", "local_fraction_delta", "pred_local_fraction",
                   "pred_rank_max_mean", "prefetch_cpu_ms", "rebalance_cpu_ms_max"):
        paired[column + "_vision_minus_text"] = paired[column + "_vision"] - paired[column + "_text"]
    paired.to_csv(out / "libra_matched_modality_plans.csv", index=False)
    stats = {"matched_conditions": len(paired), "distinct_request_quartets": paired.requests.nunique(),
        "scope": "ACTUAL_CPU_PLANNER_ON_FRESH_ROUTES_NOT_GPU_LATENCY_ORACLE",
        "modality_route_masks": "Same request quartet, target layer and source token budget; no fake padding",
        "matched_median_vision_minus_text": {c: float(paired[c].median()) for c in paired.columns
                                             if c.endswith("_vision_minus_text")},
        "planner_oracle_local_fraction_gain": {}}
    for modality, group in plans.groupby("modality"):
        stats["planner_oracle_local_fraction_gain"][modality] = {
            "median_pp": 100*float(group.local_fraction_delta.median()),
            "p90_pp": 100*float(group.local_fraction_delta.quantile(.9)),
            "max_pp": 100*float(group.local_fraction_delta.max()),
            "fraction_ge_5pp": float((group.local_fraction_delta >= .05).mean()),
            "fraction_ge_10pp": float((group.local_fraction_delta >= .1).mean())}
    (out / "libra_matched_summary.json").write_text(json.dumps(stats, indent=2))

    native = root / "online/libra_native_real4layer_sanity_v2"
    rows = [json.loads(s) for path in native.glob("rank*.jsonl") for s in path.read_text().splitlines()]
    measured = [r for r in rows if r["repeat"] >= 0 and r["policy"] == "libra"]
    sanity = {"observations": len(measured),
        "greedy_first_equal": sum(r["greedy_first_equal"] for r in measured),
        "min_cosine": min(r["logit_cosine"] for r in measured),
        "max_absolute_logit_error": max(r["logit_max_abs"] for r in measured),
        "max_relative_logit_l2": max(r["relative_logit_l2"] for r in measured),
        "scope": "REDUCED_4_LAYER_REAL_WEIGHTS_FUNCTIONAL_ONLY_CONCURRENT_CALIBRATION",
        "performance_claim_allowed": False, "full_mllm_claim_allowed": False}
    (out / "libra_native_functional_summary.json").write_text(json.dumps(sanity, indent=2))
    print(json.dumps({"quality_rows": len(quality), "modes": partial,
                      "libra_planner": stats, "libra_native": sanity}, indent=2))


if __name__ == "__main__":
    main()
