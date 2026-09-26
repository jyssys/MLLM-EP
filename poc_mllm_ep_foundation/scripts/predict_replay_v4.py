"""Apply the frozen v4 cost model to a target-EP structural replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from poc_mllm_ep_foundation.contract import parse_physical_gpus
from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_ep_foundation.scripts.analyze_physical_anchors_v4 import FEATURE_NAMES


def _cv(values: np.ndarray) -> float:
    mean = float(values.mean()) if len(values) else 0.0
    return float(values.std() / mean) if mean else 0.0


def _features(row: dict, ep: int) -> np.ndarray:
    source = np.asarray(row["source_logical_rows"], dtype=float)
    assignment = np.asarray(row["assignment_matrix"], dtype=float)
    unique = np.asarray(row["unique_activation_matrix"], dtype=float)
    # Physical calibration records ``total_assignments`` on the destination
    # expert rank after dispatch.  Keep the virtual feature contract identical:
    # rows are sources, columns are destinations, so rank load is the column
    # sum.  The previous row sum described source fan-out instead and silently
    # changed the meaning of assignments_max/CV at prediction time.
    destination_assignments = assignment.sum(0)
    dispatched = unique.sum(0)
    histogram = np.asarray(row["expert_histogram"], dtype=float)
    local = 128 // ep
    tiles = np.asarray([
        sum(math.ceil(value / 64) for value in histogram[r * local:(r + 1) * local] if value)
        for r in range(ep)
    ], dtype=float)
    remote = unique.copy()
    np.fill_diagonal(remote, 0)
    summary = np.asarray([
        math.log1p(source.sum()), math.log1p(source.max(initial=0)), _cv(source),
        math.log1p(destination_assignments.sum()),
        math.log1p(destination_assignments.max(initial=0)),
        _cv(destination_assignments), math.log1p(dispatched.sum()),
        math.log1p(dispatched.max(initial=0)), _cv(dispatched),
        float(np.count_nonzero(histogram)), math.log1p(histogram.max(initial=0)),
        math.log1p(tiles.sum()), math.log1p(tiles.max(initial=0)),
        math.log1p(remote.sum()), math.log1p(remote.sum(1).max(initial=0)),
        math.log1p(remote.sum(0).max(initial=0)),
        float((unique > 0).sum(1).max(initial=0)), int(row["layer"]) / 47.0,
    ], dtype=float)
    layer_onehot = np.zeros(48, dtype=float)
    layer_onehot[int(row["layer"])] = 1.0
    padded_traffic = np.zeros((8, 8), dtype=float)
    padded_traffic[:ep, :ep] = unique
    return np.concatenate([
        summary, np.log1p(histogram), layer_onehot,
        np.log1p(padded_traffic.reshape(-1)),
    ])


def _predict(state: dict, features: np.ndarray) -> np.ndarray:
    if state["type"] == "componentwise":
        return np.asarray([
            _predict(component_state, features)[component]
            for component, component_state in enumerate(state["component_models"])
        ], dtype=float)
    indexes = state.get("feature_indices")
    if indexes is not None:
        features = features[np.asarray(indexes, dtype=int)]
    mean = np.asarray(state["mean"], dtype=float)
    scale = np.asarray(state["scale"], dtype=float)
    query = (features - mean) / scale
    if state["type"] == "standardized_ridge":
        design = np.r_[1.0, query]
        return np.maximum(0, design @ np.asarray(state["coef"], dtype=float))
    matrix = np.asarray(state["normalized_x"], dtype=float)
    targets = np.asarray(state["targets_ms"], dtype=float)
    distance = np.sqrt(np.square(matrix - query).sum(1))
    exact = np.flatnonzero(distance < 1e-10)
    if len(exact):
        return targets[exact].mean(0)
    selected = np.argsort(distance)[: min(int(state["neighbors"]), len(distance))]
    weights = 1.0 / np.maximum(distance[selected], 1e-8)
    return (targets[selected] * weights[:, None]).sum(0) / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpus", default="4")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host = parse_physical_gpus(args.physical_gpus)
    replay = json.loads(args.replay.read_text())
    model = json.loads(args.cost_model.read_text())
    ep = int(replay["target_ep"])
    model_key = f"ep{ep}" if ep in (2, 4) else "ep4"
    state = model["models"][model_key]
    minimum = np.asarray(state["feature_min"], dtype=float)
    maximum = np.asarray(state["feature_max"], dtype=float)
    rows = []
    for row in replay["rows"]:
        features = _features(row, ep)
        indexes = state.get("feature_indices")
        compared = features if indexes is None else features[np.asarray(indexes, dtype=int)]
        outside = np.flatnonzero((compared < minimum) | (compared > maximum)).astype(int).tolist()
        prediction = _predict(state, features)
        rows.append({
            "wave": row["wave"], "layer": row["layer"],
            "request_ids": row["request_ids"],
            "dispatch_ms": float(prediction[0]), "expert_ms": float(prediction[1]),
            "combine_ms": float(prediction[2]), "routed_ms": float(prediction.sum()),
            "in_domain": not outside, "ood_feature_indexes": outside,
        })
    request_totals = {}
    for row in rows:
        key = "+".join(row["request_ids"])
        request_totals.setdefault(key, 0.0)
        request_totals[key] += row["routed_ms"]
    evidence = (f"ACTUAL_EP{ep}_CALIBRATED" if ep in (2, 4)
                else "UNVALIDATED_EP8_EXTRAPOLATION_USING_EP4_MODEL")
    payload = {
        "schema_version": "mllm_ep_prediction.v1", "host_physical_gpus": host,
        "target_ep": ep, "target_topology": replay["target_topology"],
        "host_invariance_contract": "host GPUs are not model features",
        "evidence": evidence, "in_domain_rows": sum(row["in_domain"] for row in rows),
        "total_rows": len(rows), "rows": rows, "request_routed_ms": request_totals,
        "ep8_warning": (None if ep != 8 else
            "16-local-expert compute and 8-rank collective/E2E require separate evidence; point values are sensitivity only"),
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("rows", "request_routed_ms")}, indent=2))


if __name__ == "__main__":
    main()
