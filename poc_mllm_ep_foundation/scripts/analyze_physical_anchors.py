from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


FEATURE_NAMES = (
    "log_assignments", "log_dispatched_rows", "active_experts",
    "max_expert_rows", "mean_expert_rows", "cv_expert_rows",
    "tile_count", "layer_fraction",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _feature(row: dict[str, Any]) -> list[float]:
    hist = np.asarray(row["expert_histogram"], dtype=float)
    active = hist[hist > 0]
    block = max(1, int(row.get("runtime_config", {}).get("BLOCK_SIZE_M", 64)))
    return [
        math.log1p(float(row["total_assignments"])),
        math.log1p(float(row["dispatched_rows"])),
        float(len(active)),
        float(active.max(initial=0)),
        float(active.mean() if len(active) else 0),
        float(active.std() / active.mean() if len(active) and active.mean() else 0),
        float(np.ceil(active / block).sum()),
        float(row["layer"]) / 47.0,
    ]


def _load(root: Path, manifest: dict[str, Any], ep: int) -> list[dict[str, Any]]:
    by_id = {str(row["request_id"]): row for row in manifest["requests"]}
    result = {str(row["request_id"]): row for row in _jsonl(root / "results.jsonl")}
    rows = []
    for name in sorted(glob.glob(str(root / "profile/raw_live/rank[0-9]*.jsonl"))):
        for row in _jsonl(Path(name)):
            if not bool(row.get("measured", True)):
                continue
            item = by_id[str(row["request_id"])]
            wall = result[str(row["request_id"])]
            rows.append({
                **row,
                "ep": ep,
                "split": item["anchor_split"],
                "task": item["task"],
                "vision_token_count": int(item["vision_token_count"]),
                "prompt_token_count": int(item["prompt_token_count"]),
                "wall_ms": float(wall["wall_ms"]),
                "features": _feature(row),
                "dispatch_ms": float(row["dispatch"].get("critical_ms", row["dispatch"]["ms"])),
                "expert_ms": float(row["expert"]["ms"]),
                "combine_ms": float(row["combine"].get("critical_ms", row["combine"]["ms"])),
                "stage_ms": float(
                    row["dispatch"].get("critical_ms", row["dispatch"]["ms"])
                    + row["expert"]["ms"]
                    + row["combine"].get("critical_ms", row["combine"]["ms"])
                ),
            })
    return rows


class Ridge:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        self.mean = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-8] = 1
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        return np.maximum(0, np.column_stack([np.ones(len(z)), z]) @ self.coef)

    def state(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "coef": self.coef.tolist()}


def _ape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-4) * 100


def _metric(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    a, p = np.asarray(actual), np.asarray(predicted)
    errors = _ape(a, p)
    return {
        "n": len(a),
        "median_ape_percent": float(np.median(errors)),
        "p90_ape_percent": float(np.quantile(errors, 0.9)),
        "median_signed_bias_ms": float(np.median(p - a)),
        "median_absolute_error_ms": float(np.median(np.abs(p - a))),
    }


def _group(rows: list[dict[str, Any]], predictions: dict[tuple[int, int], float] | None = None):
    inv: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        inv[(int(row["wave"]), int(row["layer"]))].append(row)
    grouped = []
    for (wave, layer), values in inv.items():
        split = {row["split"] for row in values}
        if len(split) != 1:
            raise AssertionError((wave, split))
        actual = max(row["stage_ms"] for row in values)
        predicted = None
        if predictions is not None:
            predicted = max(predictions[(row["ep_rank"], id(row))] for row in values)
        grouped.append({"wave": wave, "layer": layer, "split": next(iter(split)),
                        "actual_ms": actual, "predicted_ms": predicted,
                        "request_ids": sorted({row["request_id"] for row in values})})
    return grouped


def _request_metric(grouped: list[dict[str, Any]]) -> dict[str, Any]:
    requests: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in grouped:
        requests[row["wave"]].append(row)
    actual, predicted = [], []
    for values in requests.values():
        actual.append(sum(row["actual_ms"] for row in values))
        predicted.append(sum(row["predicted_ms"] for row in values))
    return _metric(actual, predicted)


def _fit_one(rows: list[dict[str, Any]]) -> tuple[dict[str, Ridge], dict[str, Any]]:
    train = [row for row in rows if row["split"] == "calibration"]
    dev = [row for row in rows if row["split"] == "development"]
    components = ("dispatch_ms", "expert_ms", "combine_ms")
    best = None
    for alpha in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
        models = {}
        for component in components:
            models[component] = Ridge(alpha).fit(
                np.asarray([row["features"] for row in train]),
                np.asarray([row[component] for row in train]),
            )
        dev_pred = {}
        for row in dev:
            x = np.asarray([row["features"]])
            dev_pred[(row["ep_rank"], id(row))] = sum(
                float(models[c].predict(x)[0]) for c in components
            )
        dev_group = _group(dev, dev_pred)
        score = _request_metric(dev_group)["median_ape_percent"]
        if best is None or score < best[0]:
            best = (score, models, alpha)
    assert best is not None
    models = best[1]
    all_predictions = {}
    for row in rows:
        x = np.asarray([row["features"]])
        all_predictions[(row["ep_rank"], id(row))] = sum(
            float(models[c].predict(x)[0]) for c in components
        )
    grouped = _group(rows, all_predictions)
    metrics = {}
    for split in ("calibration", "development", "untouched_test"):
        subset = [row for row in grouped if row["split"] == split]
        metrics[split] = {
            "invocation": _metric([row["actual_ms"] for row in subset],
                                  [row["predicted_ms"] for row in subset]),
            "request_routed_stage": _request_metric(subset),
        }
    return models, {"selected_alpha_on_dev": best[2], "metrics": metrics,
                    "observed_rows": len(rows), "observed_invocations": len(grouped)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ep2-root", type=Path, required=True)
    parser.add_argument("--ep4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = {"schema_version": "mllm_ep_cost_model.v2",
              "feature_names": FEATURE_NAMES, "models": {}, "validation": {}}
    for ep, root in ((2, args.ep2_root), (4, args.ep4_root)):
        rows = _load(root, manifest, ep)
        models, validation = _fit_one(rows)
        output["models"][f"ep{ep}"] = {key: value.state() for key, value in models.items()}
        output["validation"][f"ep{ep}"] = validation
    output["evidence"] = {
        "ep2": "ACTUAL_TP1_DP2_EP2_DEEPEP",
        "ep4": "ACTUAL_TP2_DP2_EP4_DEEPEP",
        "ep8": "UNVALIDATED_EXTRAPOLATION",
    }
    atomic_json(args.output, output)
    print(json.dumps(output["validation"], indent=2))


if __name__ == "__main__":
    main()
