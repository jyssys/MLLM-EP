"""Build an explicitly unvalidated EP8 latency sensitivity projection.

Rank-local expert time comes from real 16-local-expert H100 measurements.
Eight-rank communication is not measured: its scenario spread is transferred
from the blind EP2->EP4 residual distribution and is therefore a sensitivity,
never a confidence interval or physical EP8 validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_ep_foundation.scripts.predict_replay_v4 import _features, _predict


def _kernel_vector(histogram: np.ndarray) -> np.ndarray:
    return np.asarray([
        math.log1p(float(histogram.sum())),
        float(np.count_nonzero(histogram)),
        math.log1p(float(histogram.max(initial=0))),
        math.log1p(float(sum(math.ceil(int(value) / 64) for value in histogram if value))),
    ])


def _kernel_model(calibration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if calibration.get("status") != "LOCAL_KERNEL_CALIBRATED":
        raise ValueError("EP8 local-kernel calibration is not complete")
    observations = calibration.get("observations", [])
    if not observations:
        raise ValueError("EP8 local-kernel calibration contains no observations")
    features = np.stack([
        _kernel_vector(np.asarray(row["local_expert_histogram"], dtype=np.int64))
        for row in observations
    ])
    values = np.asarray([row["median_ms"] for row in observations], dtype=float)
    scale = np.maximum(np.ptp(features, axis=0), 1.0)
    return features, values, scale


def _kernel_predict(histogram: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]
                    ) -> tuple[float, float]:
    features, values, scale = model
    distance = np.sqrt(np.square((features - _kernel_vector(histogram)) / scale).sum(1))
    selected = np.argsort(distance)[: min(4, len(distance))]
    exact = selected[distance[selected] < 1e-10]
    if len(exact):
        return float(values[exact].mean()), 0.0
    weights = 1.0 / np.maximum(distance[selected], 1e-8)
    return float(np.average(values[selected], weights=weights)), float(distance[selected[0]])


def _transfer_factors(transfer: dict[str, Any]) -> dict[str, float]:
    values = transfer.get("request_values", {}).get("untouched_test", [])
    ratios = np.asarray([
        float(row["actual_ep4_ms"]) / max(float(row["blind_ep2_model_prediction_ms"]), 1e-8)
        for row in values
    ], dtype=float)
    if len(ratios) < 4:
        raise ValueError("at least four untouched EP2->EP4 transfer ratios are required")
    return {
        "endpoint_optimistic": float(np.quantile(ratios, 0.10)),
        "calibrated_base": float(np.median(ratios)),
        "concurrency_stressed": float(np.quantile(ratios, 0.90)),
    }


def project(replay: dict[str, Any], cost_model: dict[str, Any],
            local_kernel: dict[str, Any], transfer: dict[str, Any]) -> dict[str, Any]:
    if int(replay.get("target_ep", -1)) != 8:
        raise ValueError("projection requires an EP8 structural replay")
    state = cost_model["models"]["ep4"]
    kernel = _kernel_model(local_kernel)
    factors = _transfer_factors(transfer)
    scenario_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in factors}
    for row in replay["rows"]:
        generic = _predict(state, _features(row, 8))
        histogram = np.asarray(row["expert_histogram"], dtype=np.int64)
        rank_compute = []
        rank_distance = []
        for rank in range(8):
            value, distance = _kernel_predict(histogram[rank * 16:(rank + 1) * 16], kernel)
            rank_compute.append(value)
            rank_distance.append(distance)
        critical_compute = max(rank_compute, default=0.0)
        for scenario, factor in factors.items():
            dispatch = float(generic[0]) * factor
            combine = float(generic[2]) * factor
            scenario_rows[scenario].append({
                "wave": int(row["wave"]), "layer": int(row["layer"]),
                "request_ids": list(row["request_ids"]),
                "dispatch_ms": dispatch,
                "expert_ms": critical_compute,
                "combine_ms": combine,
                "routed_ms": dispatch + critical_compute + combine,
                "critical_rank_local_kernel_distance": max(rank_distance, default=0.0),
                "communication_transfer_factor": factor,
            })
    scenarios = {}
    for name, rows in scenario_rows.items():
        totals: dict[str, float] = {}
        for row in rows:
            key = "+".join(row["request_ids"])
            totals[key] = totals.get(key, 0.0) + float(row["routed_ms"])
        scenarios[name] = {
            "communication_transfer_factor": factors[name],
            "rows": rows,
            "request_routed_ms": totals,
            "request_median_ms": float(np.median(list(totals.values()))),
        }
    return {
        "schema_version": "mllm_ep8_sensitivity.v1",
        "target_ep": 8,
        "target_topology": replay["target_topology"],
        "EP8_WORKLOAD_STATUS": "STRUCTURAL_VERIFIED",
        "EP8_COMPUTE_STATUS": "LOCAL_KERNEL_CALIBRATED",
        "EP8_COMM_STATUS": "EXTRAPOLATED",
        "EP8_FULL_LATENCY_STATUS": "UNVALIDATED_EXTRAPOLATION",
        "scenario_range_is_confidence_interval": False,
        "communication_sensitivity_source": (
            "untouched blind EP2-to-EP4 actual/predicted routed-stage ratio p10/median/p90"
        ),
        "expert_time_source": "actual single-H100 BF16 16-local-expert grouped-kernel replay",
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--local-kernel", type=Path, required=True)
    parser.add_argument("--ep2-to-ep4-transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = project(
        json.loads(args.replay.read_text()),
        json.loads(args.cost_model.read_text()),
        json.loads(args.local_kernel.read_text()),
        json.loads(args.ep2_to_ep4_transfer.read_text()),
    )
    atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "statuses": {key: value for key, value in result.items() if key.startswith("EP8_")},
        "scenario_request_medians_ms": {
            key: value["request_median_ms"] for key, value in result["scenarios"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
