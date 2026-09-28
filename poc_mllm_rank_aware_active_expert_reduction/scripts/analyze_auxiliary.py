#!/usr/bin/env python3
"""Bounded natural, discovery-only stress, and phase-mixed structural lanes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.contract import atomic_json, read_jsonl_gz
from poc_mllm_rank_aware_active_expert_reduction.scripts.analyze_q3_structural import (
    metrics, paths_for_wave, removal_native_tail, removal_rank_aware, reduction,
)


def available(row: dict[str, Any], step: int) -> bool:
    return any(int(item["step"]) == step for item in row["sampled_routes"])


def skew_score(row: dict[str, Any], ep: int = 4) -> float:
    scores = []
    for route in row["sampled_routes"]:
        counts = [0] * ep
        for expert in route["route_ids"]:
            counts[int(expert) // (128 // ep)] += 1
        scores.append(max(counts) / max(float(np.mean(counts)), 1e-12))
    return float(np.mean(scores))


def evaluate(paths: list[dict[str, Any]], ep: int, budget_fraction: float) -> dict[str, Any]:
    budget = max(1, int(round(len(paths) * budget_fraction)))
    base = metrics(paths, set(), ep)
    output = {"baseline": base, "path_reduction_pct": 100.0 * budget / len(paths)}
    for name, removed in (
        ("native_tail", removal_native_tail(paths, budget)),
        ("rank_aware_upper", removal_rank_aware(paths, budget, ep)),
    ):
        changed = metrics(paths, removed, ep)
        output[name] = {
            **changed,
            "max_active_expert_reduction_pct": reduction(base, changed, "max_active_experts"),
            "total_active_expert_reduction_pct": reduction(base, changed, "total_active_experts"),
            "max_active_weight_reduction_pct": reduction(base, changed, "max_active_weight_bytes"),
        }
    return output


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["lane"], row["ep"])].append(row["result"])
    rows = []
    for (lane, ep), values in sorted(grouped.items()):
        rows.append({
            "lane": lane, "ep": ep, "observations": len(values),
            "path_reduction_pct": float(np.mean([r["path_reduction_pct"] for r in values])),
            "native_tail_max_active_reduction_pct": float(np.mean([
                r["native_tail"]["max_active_expert_reduction_pct"] for r in values])),
            "rank_aware_max_active_reduction_pct": float(np.mean([
                r["rank_aware_upper"]["max_active_expert_reduction_pct"] for r in values])),
            "rank_aware_total_active_reduction_pct": float(np.mean([
                r["rank_aware_upper"]["total_active_expert_reduction_pct"] for r in values])),
            "baseline_rank_max_mean": float(np.mean([
                r["baseline"]["rank_max_mean"] for r in values])),
        })
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.20)
    args = parser.parse_args()
    all_rows = read_jsonl_gz(args.trace)
    discovery = [row for row in all_rows if row["analysis_split"] == "discovery"]
    heldout = [row for row in all_rows if row["analysis_split"] == "heldout"]

    # Natural lane: source-order held-out B8. Selection uses no held-out result.
    natural = heldout[:8]
    # Stress lane is discovery-only and frozen from a discovery routing-skew score.
    ranked = sorted(discovery, key=lambda row: (-skew_score(row), row["request_id"]))
    stress = ranked[:8]
    phase_steps = [16, 64, 96, 128, 16, 64, 96, 128]
    phase_rows = []
    for step in phase_steps:
        candidate = next(row for row in heldout
                         if row not in [item[0] for item in phase_rows] and available(row, step))
        phase_rows.append((candidate, step))

    records = []
    for lane, wave in (("natural_heldout_b8", natural),
                       ("stress_discovery_only_b8", stress)):
        common = sorted(set.intersection(*[
            {int(item["step"]) for item in row["sampled_routes"]} for row in wave]))
        for step in [value for value in common if value in (16, 64, 96, 128)]:
            for layer in range(48):
                paths = paths_for_wave(wave, step, layer)
                for ep in (2, 4, 8):
                    records.append({"lane": lane, "ep": ep,
                                    "result": evaluate(paths, ep, args.budget)})

    # Phase-mixed requests use different frozen decode positions in the same wave.
    for layer in range(48):
        paths = []
        for request_index, (row, step) in enumerate(phase_rows):
            route = next(item for item in row["sampled_routes"]
                         if int(item["step"]) == step and int(item["layer"]) == layer)
            for slot, (expert, weight) in enumerate(zip(route["route_ids"], route["route_weights"])):
                paths.append({"request_index": request_index, "request_id": row["request_id"],
                              "expert": int(expert), "weight": float(weight), "slot": slot,
                              "source_rank": int(route["source_ep_rank"])})
        for ep in (2, 4, 8):
            records.append({"lane": "phase_mixed_16_64_96_128_b8", "ep": ep,
                            "result": evaluate(paths, ep, args.budget)})

    payload = {
        "schema_version": "rank_aware_auxiliary_structural.v1",
        "evidence": "STRUCTURAL_ROUTE_REPLAY_ONLY",
        "source_trace": str(args.trace.resolve()),
        "budget": args.budget,
        "selection": {
            "natural_request_ids": [row["request_id"] for row in natural],
            "stress_request_ids": [row["request_id"] for row in stress],
            "stress_selection": "top discovery-only mean EP4 rank-max/mean; no held-out timing",
            "phase_mixed": [{"request_id": row["request_id"], "step": step}
                            for row, step in phase_rows],
        },
        "aggregate": aggregate(records),
        "ep8_label": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "latency_claim": "NONE",
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
