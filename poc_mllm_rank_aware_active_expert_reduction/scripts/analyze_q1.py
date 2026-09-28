#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.contract import atomic_json


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks without a SciPy runtime dependency."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) < 2:
        return float("nan")
    ranked_a, ranked_b = _average_ranks(values_a), _average_ranks(values_b)
    if np.std(ranked_a) == 0 or np.std(ranked_b) == 0:
        return float("nan")
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def regression(y: np.ndarray, columns: list[np.ndarray], names: list[str]) -> dict[str, Any]:
    x = np.column_stack([np.ones(len(y)), *columns])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    prediction = x @ coefficients
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {"coefficients": {"intercept": float(coefficients[0]), **{
        name: float(value) for name, value in zip(names, coefficients[1:])}},
        "r2": 1.0 - residual / total if total else 0.0,
        "mae_ms": float(np.mean(np.abs(y - prediction)))}


def analyze_microbench(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    observations = payload["observations"]
    rows = []
    for condition in ("hot_repeated", "cold_l2_flushed"):
        for local_experts in (16, 32, 64):
            selected = [row for row in observations
                        if row["cache_condition"] == condition
                        and row["local_experts"] == local_experts]
            paths = np.asarray([row["token_expert_paths"] for row in selected], dtype=float)
            active = np.asarray([row["active_experts"] for row in selected], dtype=float)
            latency = np.asarray([row["median_ms"] for row in selected], dtype=float)
            sweep_a = [row for row in selected if row["sweep"] == "A_fixed_32_paths"]
            means = {value: float(np.mean([row["median_ms"] for row in sweep_a
                                          if row["active_experts"] == value]))
                     for value in (2, 4, 8, 16)}
            rows.append({
                "cache_condition": condition, "local_experts": local_experts,
                "fixed_32_path_mean_ms": means,
                "fixed_token_2_to_16_active_effect_pct": 100.0 * (means[16] / means[2] - 1.0),
                "token_spearman": spearman(paths, latency),
                "active_expert_spearman": spearman(active, latency),
                "token_only_model": regression(latency, [paths], ["token_paths"]),
                "active_only_model": regression(latency, [active], ["active_experts"]),
                "joint_model": regression(latency, [paths, active], ["token_paths", "active_experts"]),
            })
    cold = [row for row in rows if row["cache_condition"] == "cold_l2_flushed"]
    effect = float(np.mean([row["fixed_token_2_to_16_active_effect_pct"] for row in cold]))
    active_r2_increment = float(np.mean([
        row["joint_model"]["r2"] - row["token_only_model"]["r2"] for row in cold]))
    return {"source": str(path.resolve()), "rows": rows,
            "cold_fixed_token_mean_active_effect_pct": effect,
            "cold_joint_r2_increment_over_token_only": active_r2_increment,
            "microbench_gate_pass": effect >= 10.0 and active_r2_increment >= 0.05}


def read_stage(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "raw_stage").glob("worker_pid*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def analyze_stage(root: Path) -> dict[str, Any]:
    rows = read_stage(root)
    if not rows:
        return {"source": str(root.resolve()), "status": "UNAVAILABLE_NO_RAW_STAGE", "rows": 0}
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["decode_step"], row["layer"])].append(row)
    critical = []
    for key, values in grouped.items():
        chosen = max(values, key=lambda row: row["moe_ms"])
        critical.append(chosen)
    result = []
    for condition in sorted({row["condition"] for row in critical}):
        selected = [row for row in critical if row["condition"] == condition]
        dispatch = np.asarray([row["dispatch_ms"] for row in selected])
        expert = np.asarray([row["expert_execution_ms"] for row in selected])
        combine = np.asarray([row["combine_ms"] for row in selected])
        moe = np.asarray([row["moe_ms"] for row in selected])
        result.append({
            "condition": condition, "critical_invocations": len(selected),
            "dispatch_mean_ms": float(dispatch.mean()),
            "expert_execution_mean_ms": float(expert.mean()),
            "combine_mean_ms": float(combine.mean()), "moe_mean_ms": float(moe.mean()),
            "dispatch_share_pct": float(100 * dispatch.sum() / moe.sum()),
            "expert_execution_share_pct": float(100 * expert.sum() / moe.sum()),
            "combine_share_pct": float(100 * combine.sum() / moe.sum()),
        })
    pairs = np.asarray([row["token_expert_pairs"] for row in rows], dtype=float)
    active = np.asarray([row["distinct_active_experts"] for row in rows], dtype=float)
    latency = np.asarray([row["expert_execution_ms"] for row in rows], dtype=float)
    return {
        "source": str(root.resolve()), "status": "MEASURED_TRUE_DEEPEP",
        "raw_rank_invocations": len(rows), "critical_invocations": len(critical),
        "topologies": sorted({json.dumps(row["topology"], sort_keys=True) for row in rows}),
        "prepare_finalize_backends": sorted({row["prepare_finalize_backend"] for row in rows}),
        "expert_backends": sorted({row["expert_backend"] for row in rows}),
        "stage_breakdown": result,
        "rank_local_correlations": {
            "token_pairs_vs_expert_ms_spearman": spearman(pairs, latency),
            "active_experts_vs_expert_ms_spearman": spearman(active, latency),
            "active_weight_bytes_vs_expert_ms_spearman": spearman(
                active * 9437184, latency),
        },
        "rank_local_models": {
            "token_only": regression(latency, [pairs], ["token_pairs"]),
            "active_only": regression(latency, [active], ["active_experts"]),
            "joint": regression(latency, [pairs, active], ["token_pairs", "active_experts"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbench", type=Path, required=True)
    parser.add_argument("--ep2", type=Path)
    parser.add_argument("--ep4", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    micro = analyze_microbench(args.microbench)
    ep2 = analyze_stage(args.ep2) if args.ep2 else {"status": "NOT_RUN"}
    ep4 = analyze_stage(args.ep4) if args.ep4 else {"status": "NOT_RUN"}
    measured = [row for row in (ep2, ep4) if row.get("status") == "MEASURED_TRUE_DEEPEP"]
    expert_shares = [condition["expert_execution_share_pct"] for row in measured
                     for condition in row["stage_breakdown"]]
    stage_pass = bool(expert_shares) and float(np.mean(expert_shares)) >= 10.0
    payload = {
        "schema_version": "rank_aware_q1_analysis.v1", "microbench": micro,
        "ep2": ep2, "ep4": ep4,
        "gate": {"microbench_pass": micro["microbench_gate_pass"],
                 "stage_expert_share_pass": stage_pass,
                 "q1_pass": micro["microbench_gate_pass"] and stage_pass and len(measured) == 2,
                 "frozen_thresholds": {"fixed_token_effect_pct": 10.0,
                                       "joint_r2_increment": 0.05,
                                       "expert_stage_share_pct": 10.0}},
    }
    atomic_json(args.output, payload)
    print(json.dumps({"gate": payload["gate"],
                      "microbench_summary": {k: micro[k] for k in (
                          "cold_fixed_token_mean_active_effect_pct",
                          "cold_joint_r2_increment_over_token_only")},
                      "ep2": ep2.get("status"), "ep4": ep4.get("status")}, indent=2))


if __name__ == "__main__": main()
