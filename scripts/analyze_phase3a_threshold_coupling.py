#!/usr/bin/env python3
"""Summarize the three-point Phase-3A threshold/EP coupling experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from virtual_ep.schema import TraceBundle


THRESHOLDS = ((0.95, "threshold_095"), (0.80, "threshold_080"), (0.60, "threshold_060"))


def _prediction(path: Path) -> dict:
    totals = {
        "dispatch_ms": 0.0,
        "expert_ms": 0.0,
        "combine_ms": 0.0,
        "stage_ms": 0.0,
        "remote_bytes": 0,
    }
    max_mean = []
    cv = []
    fanout = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            totals["dispatch_ms"] += float(row["dispatch_ms"])
            totals["expert_ms"] += float(row["critical_expert_ms"])
            totals["combine_ms"] += float(row["combine_ms"])
            totals["stage_ms"] += float(row["moe_stage_ms"])
            totals["remote_bytes"] += int(row["remote_dispatch_bytes"])
            max_mean.append(float(row["max_mean_load"]))
            cv.append(float(row["rank_load_cv"]))
            fanout.append(float(row["mean_token_fanout"]))
    totals.update(
        mean_max_mean=float(np.mean(max_mean)),
        mean_cv=float(np.mean(cv)),
        mean_fanout=float(np.mean(fanout)),
        invocations=len(max_mean),
    )
    return totals


def _actual_ep2(path: Path) -> dict:
    document = json.loads(path.read_text())
    cases = document["cases"]
    result = {
        "dispatch_ms": sum(case["EP_dispatch"]["median_ms"] for case in cases),
        "expert_ms": sum(case["routed_expert_compute"]["median_ms"] for case in cases),
        "combine_ms": sum(case["EP_combine"]["median_ms"] for case in cases),
        "stage_ms": sum(case["EP_stage_excluding_bridge"]["median_ms"] for case in cases),
        "invocations": len(cases),
        "measurement": "all-invocation true-EP2 route replay",
    }
    return result


def _trajectory(trace: TraceBundle) -> dict:
    iterations = trace.iterations
    request_ids = np.unique(iterations["request_id"])
    request_nfe = {
        int(request): int(np.count_nonzero(iterations["request_id"] == request))
        for request in request_ids
    }
    request_blocks = {
        int(request): int(
            len(np.unique(iterations["block_id"][iterations["request_id"] == request]))
        )
        for request in request_ids
    }
    return {
        "requests": len(request_ids),
        "total_nfe": int(len(iterations["request_id"])),
        "mean_nfe": float(np.mean(list(request_nfe.values()))),
        "median_nfe": float(np.median(list(request_nfe.values()))),
        "total_blocks": int(sum(request_blocks.values())),
        "mean_blocks": float(np.mean(list(request_blocks.values()))),
        "total_refinement_iterations": int(len(iterations["request_id"])),
        "masked_before_sum": int(iterations["masked_before"].sum()),
        "accepted_sum": int(iterations["accepted"].sum()),
        "masked_after_sum": int(iterations["masked_after"].sum()),
        "per_request_nfe": request_nfe,
    }


def _percent_reduction(value: float, baseline: float) -> float:
    return (baseline - value) / baseline * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for threshold, dirname in THRESHOLDS:
        root = args.run_dir / dirname
        generations = [json.loads(line) for line in (root / "generations.jsonl").read_text().splitlines()]
        if [row["sample_id"] for row in generations] != list(range(32)):
            raise RuntimeError(f"{dirname} is not the fixed IDs 0..31 cohort")
        if any(float(row["threshold"]) != threshold for row in generations):
            raise RuntimeError(f"{dirname} mixed threshold provenance")
        trace = TraceBundle.load(root / "trace.npz")
        if float(trace.metadata["threshold"]) != threshold:
            raise RuntimeError(f"{dirname} trace threshold mismatch")
        rows.append(
            {
                "threshold": threshold,
                "accuracy": float(np.mean([row["correct"] for row in generations])),
                "correct": int(sum(row["correct"] for row in generations)),
                "generation_tokens": int(sum(row["generation_tokens"] for row in generations)),
                "remaining_masks": int(sum(row["remaining_mask_tokens"] for row in generations)),
                "trajectory": _trajectory(trace),
                "actual_ep2": _actual_ep2(root / "true_ep2_replay" / "aggregate.json"),
                "predicted_ep2": _prediction(root / "ep2_prediction" / "invocations.csv"),
                "simulated_ep4": _prediction(root / "ep4_prediction" / "invocations.csv"),
            }
        )

    baseline = rows[0]
    for row in rows:
        row["change_vs_095_percent"] = {
            "nfe_reduction": _percent_reduction(
                row["trajectory"]["total_nfe"], baseline["trajectory"]["total_nfe"]
            ),
            "actual_ep2_stage_reduction": _percent_reduction(
                row["actual_ep2"]["stage_ms"], baseline["actual_ep2"]["stage_ms"]
            ),
            "predicted_ep2_stage_reduction": _percent_reduction(
                row["predicted_ep2"]["stage_ms"], baseline["predicted_ep2"]["stage_ms"]
            ),
            "simulated_ep4_stage_reduction": _percent_reduction(
                row["simulated_ep4"]["stage_ms"], baseline["simulated_ep4"]["stage_ms"]
            ),
            "remote_bytes_reduction_ep2": _percent_reduction(
                row["predicted_ep2"]["remote_bytes"], baseline["predicted_ep2"]["remote_bytes"]
            ),
            "remote_bytes_reduction_ep4": _percent_reduction(
                row["simulated_ep4"]["remote_bytes"], baseline["simulated_ep4"]["remote_bytes"]
            ),
        }

    actual_reductions = np.asarray(
        [row["change_vs_095_percent"]["actual_ep2_stage_reduction"] for row in rows[1:]]
    )
    nfe_reductions = np.asarray(
        [row["change_vs_095_percent"]["nfe_reduction"] for row in rows[1:]]
    )
    aware_reductions = np.asarray(
        [row["change_vs_095_percent"]["predicted_ep2_stage_reduction"] for row in rows[1:]]
    )
    comparison = {
        "nfe_only_mae_reduction_points": float(np.mean(np.abs(nfe_reductions - actual_reductions))),
        "ep_aware_mae_reduction_points": float(np.mean(np.abs(aware_reductions - actual_reductions))),
        "winner": (
            "EP-aware"
            if np.mean(np.abs(aware_reductions - actual_reductions))
            < np.mean(np.abs(nfe_reductions - actual_reductions))
            else "NFE-only"
        ),
    }
    ep2_order = [row["threshold"] for row in sorted(rows, key=lambda row: row["actual_ep2"]["stage_ms"])]
    ep4_order = [row["threshold"] for row in sorted(rows, key=lambda row: row["simulated_ep4"]["stage_ms"])]
    result = {
        "cohort": "GSM8K fixed IDs 0..31",
        "independent_rollout_per_threshold": True,
        "rows": rows,
        "nfe_only_vs_ep_aware": comparison,
        "threshold_ranking": {
            "actual_ep2_fastest_to_slowest": ep2_order,
            "simulated_ep4_fastest_to_slowest": ep4_order,
            "consistent": ep2_order == ep4_order,
        },
        "cost_accounting": {
            "included": ["EP dispatch", "routed expert compute", "EP combine"],
            "replicated_state_bridge_allgather": "excluded",
            "actual_ep2_scope": "true-EP2 route replay, not production serving latency",
            "ep4_scope": "SIMULATED-EP4-EP2-CALIBRATED routed-MoE stage",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_name("threshold_summary.csv").open("w", newline="") as stream:
        fieldnames = [
            "threshold",
            "correct",
            "accuracy",
            "total_nfe",
            "mean_nfe",
            "total_blocks",
            "actual_ep2_dispatch_ms",
            "actual_ep2_expert_ms",
            "actual_ep2_combine_ms",
            "actual_ep2_stage_ms",
            "simulated_ep4_dispatch_ms",
            "simulated_ep4_expert_ms",
            "simulated_ep4_combine_ms",
            "simulated_ep4_stage_ms",
            "ep2_mean_max_mean_load",
            "ep2_mean_cv",
            "ep2_remote_bytes",
            "ep2_mean_fanout",
            "ep4_mean_max_mean_load",
            "ep4_mean_cv",
            "ep4_remote_bytes",
            "ep4_mean_fanout",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "threshold": row["threshold"],
                    "correct": row["correct"],
                    "accuracy": row["accuracy"],
                    "total_nfe": row["trajectory"]["total_nfe"],
                    "mean_nfe": row["trajectory"]["mean_nfe"],
                    "total_blocks": row["trajectory"]["total_blocks"],
                    "actual_ep2_dispatch_ms": row["actual_ep2"]["dispatch_ms"],
                    "actual_ep2_expert_ms": row["actual_ep2"]["expert_ms"],
                    "actual_ep2_combine_ms": row["actual_ep2"]["combine_ms"],
                    "actual_ep2_stage_ms": row["actual_ep2"]["stage_ms"],
                    "simulated_ep4_dispatch_ms": row["simulated_ep4"]["dispatch_ms"],
                    "simulated_ep4_expert_ms": row["simulated_ep4"]["expert_ms"],
                    "simulated_ep4_combine_ms": row["simulated_ep4"]["combine_ms"],
                    "simulated_ep4_stage_ms": row["simulated_ep4"]["stage_ms"],
                    "ep2_mean_max_mean_load": row["predicted_ep2"]["mean_max_mean"],
                    "ep2_mean_cv": row["predicted_ep2"]["mean_cv"],
                    "ep2_remote_bytes": row["predicted_ep2"]["remote_bytes"],
                    "ep2_mean_fanout": row["predicted_ep2"]["mean_fanout"],
                    "ep4_mean_max_mean_load": row["simulated_ep4"]["mean_max_mean"],
                    "ep4_mean_cv": row["simulated_ep4"]["mean_cv"],
                    "ep4_remote_bytes": row["simulated_ep4"]["remote_bytes"],
                    "ep4_mean_fanout": row["simulated_ep4"]["mean_fanout"],
                }
            )

    thresholds = [row["threshold"] for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, [row["accuracy"] * 100 for row in rows], marker="o", label="Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("GSM8K accuracy (%)")
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output.with_name("quality_vs_threshold.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, [row["change_vs_095_percent"]["nfe_reduction"] for row in rows], marker="o", label="NFE")
    plt.plot(thresholds, [row["change_vs_095_percent"]["actual_ep2_stage_reduction"] for row in rows], marker="o", label="Actual EP2 replay")
    plt.plot(thresholds, [row["change_vs_095_percent"]["simulated_ep4_stage_reduction"] for row in rows], marker="o", label="Simulated EP4")
    plt.xlabel("Threshold")
    plt.ylabel("Reduction vs threshold 0.95 (%)")
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output.with_name("threshold_ep_coupling.png"), dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
