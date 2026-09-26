"""Blindly transfer the frozen EP2 predictor to actual EP4 observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_ep_foundation.scripts.analyze_physical_anchors_v4 import (
    _load, _ranking, _request_arrays, _summary,
)
from poc_mllm_ep_foundation.scripts.predict_replay_v4 import _predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ep4-root", type=Path, required=True)
    parser.add_argument("--cost-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coherent-request-target", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    model = json.loads(args.cost_model.read_text())
    state = model["models"]["ep2"]
    samples = _load(args.ep4_root, manifest, 4)
    features = np.stack([row["features"] for row in samples])
    predicted = np.stack([_predict(state, row) for row in features])
    metrics = {}
    values = {}
    for split in ("calibration", "development", "untouched_test"):
        actual, estimate, names = _request_arrays(
            samples, predicted, split,
            coherent_repeat_target=args.coherent_request_target,
        )
        if not len(actual):
            metrics[split] = {
                "status": "INSUFFICIENT_DATA",
                "reason": f"manifest contains no {split} requests",
                "request_routed_stage": None,
                "request_ranking_and_gain": None,
            }
            values[split] = []
            continue
        metrics[split] = {
            "status": "MEASURED",
            "request_routed_stage": _summary(actual, estimate),
            "request_ranking_and_gain": _ranking(actual, estimate),
        }
        values[split] = [
            {"request_group": name, "actual_ep4_ms": float(left),
             "blind_ep2_model_prediction_ms": float(right)}
            for name, left, right in zip(names, actual, estimate)
        ]
    payload = {
        "schema_version": "mllm_ep2_to_ep4_blind_transfer.v1",
        "fit_evidence": "EP2_ONLY; no EP4 target used for predictor fitting",
        "evaluation_evidence": "ACTUAL_EP4_REPEAT_MEDIAN",
        "request_target_policy": (
            "median_of_complete_repeat_totals"
            if args.coherent_request_target else
            "sum_of_per_layer_component_medians"
        ),
        "interpretation": (
            "A transfer diagnostic for EP8 uncertainty design; success would not validate EP8, "
            "and failure does not invalidate the separately EP4-calibrated model."
        ),
        "metrics": metrics, "request_values": values,
    }
    atomic_json(args.output, payload)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
