from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..io import atomic_json

FEATURES = ("assignments", "max_rank_assignments", "remote_unique_rows", "active_experts")
STAGES = ("dispatch_ms", "expert_ms", "combine_ms", "routed_ms")


@dataclass
class Prediction:
    values: dict[str, float | None]
    evidence: str
    ood_distance: float
    neighbors: list[str]


class NearestShapeCostModel:
    """Conservative exact-signature nearest-shape model with explicit OOD.

    It intentionally emits no milliseconds when calibration is missing or the
    query is outside the recorded shape envelope.  EP wall is predicted from
    measured parallel critical-path targets, never by summing rank times.
    """

    def __init__(self, calibration: dict[str, Any], max_distance: float = 0.75):
        self.calibration = calibration
        self.max_distance = max_distance
        signature = calibration.get("signature", {})
        if signature.get("hidden_size") != 2048 or signature.get("intermediate_size") != 768:
            raise ValueError("stale calibration shape; expected Qwen3-VL d=2048,m=768")
        if signature.get("num_experts") != 128 or signature.get("top_k") != 8:
            raise ValueError("stale calibration router contract")
        self.rows = calibration.get("observations", [])

    @staticmethod
    def _vector(row: dict[str, Any]) -> np.ndarray:
        return np.array([math.log1p(float(row[name])) for name in FEATURES], dtype=float)

    def predict(self, query: dict[str, Any], k: int = 5) -> Prediction:
        if not self.rows:
            return Prediction({name: None for name in STAGES}, "SIM_UNVALIDATED_NO_CALIBRATION", math.inf, [])
        matrix = np.stack([self._vector(row) for row in self.rows])
        scale = np.maximum(np.ptp(matrix, axis=0), 1.0)
        distances = np.linalg.norm((matrix - self._vector(query)) / scale, axis=1)
        order = np.argsort(distances)[: min(k, len(self.rows))]
        distance = float(distances[order[0]])
        if distance > self.max_distance:
            return Prediction({name: None for name in STAGES}, "OUT_OF_DOMAIN_NO_POINT_ESTIMATE", distance,
                              [str(self.rows[i].get("observation_id")) for i in order])
        weights = 1.0 / np.maximum(distances[order], 1e-6)
        values = {
            stage: float(np.average([float(self.rows[i][stage]) for i in order], weights=weights))
            for stage in STAGES
        }
        return Prediction(values, "CALIBRATED_NEAREST_SHAPE", distance,
                          [str(self.rows[i].get("observation_id")) for i in order])


def validate(calibration_path: Path, replay_path: Path, output: Path) -> dict[str, Any]:
    calibration = json.loads(calibration_path.read_text())
    replay = json.loads(replay_path.read_text())
    model = NearestShapeCostModel(calibration)
    truth = calibration.get("heldout_truth", {})
    predictions = []
    errors: list[float] = []
    signed: list[float] = []
    for row in replay["rows"]:
        key = f"{row['request_id']}:{row['layer']}"
        if key not in truth:
            continue
        pred = model.predict(row)
        actual = float(truth[key]["routed_ms"])
        value = pred.values["routed_ms"]
        if value is None:
            predictions.append({"key": key, "actual_ms": actual, "predicted_ms": None,
                                "evidence": pred.evidence, "ood_distance": pred.ood_distance})
            continue
        error = abs(value - actual) / max(actual, 1e-9)
        errors.append(error)
        signed.append(value - actual)
        predictions.append({"key": key, "actual_ms": actual, "predicted_ms": value,
                            "ape": error, "signed_error_ms": value - actual,
                            "evidence": pred.evidence, "ood_distance": pred.ood_distance})
    status = "NOT_RUN"
    if errors:
        median = float(np.median(errors))
        p90 = float(np.percentile(errors, 90))
        status = "PASS" if median <= 0.10 and p90 <= 0.20 else "FAIL"
    else:
        median = p90 = None
    payload = {
        "status": status,
        "request_level_split": calibration.get("split"),
        "evaluated": len(errors),
        "median_ape": median,
        "p90_ape": p90,
        "signed_bias_ms": float(np.mean(signed)) if signed else None,
        "ranking_spearman": None,
        "material_gain_sign_accuracy": None,
        "predictions": predictions,
        "caveat": "ranking/sign require matched configuration pairs and are never inferred from one baseline point",
    }
    atomic_json(output, payload)
    return payload

