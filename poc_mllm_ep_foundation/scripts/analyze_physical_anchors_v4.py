"""Fit a traffic-aware physical EP2/EP4 routed-stage cost model.

Unlike the v2 rank-local diagnostic, this model operates on the invocation
critical path.  It includes exact source/destination traffic features and
collapses repeated measurements of the same request-pair/layer to a median
before validation.  Model selection uses only the development split; the
untouched split is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.constants import NUM_EXPERTS
from poc_mllm_ep_foundation.io import atomic_json


BASE_FEATURE_NAMES = (
    "log_source_rows_sum", "log_source_rows_max", "source_rows_cv",
    "log_assignments_sum", "log_assignments_max", "assignments_cv",
    "log_dispatched_rows_sum", "log_dispatched_rows_max", "dispatched_rows_cv",
    "active_experts", "log_max_expert_rows", "log_tile_count_sum",
    "log_tile_count_max", "log_remote_unique_sum", "log_remote_out_max",
    "log_remote_in_max", "max_fanout", "layer_fraction",
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *(f"log_expert_rows_{expert}" for expert in range(NUM_EXPERTS)),
    *(f"layer_onehot_{layer}" for layer in range(48)),
    *(f"log_traffic_{source}_{destination}" for source in range(8)
      for destination in range(8)),
)
COMPONENTS = ("dispatch_ms", "expert_ms", "combine_ms")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _cv(values: np.ndarray) -> float:
    mean = float(values.mean()) if len(values) else 0.0
    return float(values.std() / mean) if mean else 0.0


def _one_invocation(values: list[dict[str, Any]], raw: Path, ep: int) -> dict[str, Any]:
    values = sorted(values, key=lambda row: int(row["ep_rank"]))
    if len(values) != ep:
        raise AssertionError((values[0]["wave"], values[0]["layer"], len(values), ep))
    source_rows = np.asarray([row["source_rows"] for row in values], dtype=float)
    assignments = np.asarray([row["total_assignments"] for row in values], dtype=float)
    dispatched = np.asarray([row["dispatched_rows"] for row in values], dtype=float)
    histograms = [np.asarray(row["expert_histogram"], dtype=float) for row in values]
    histogram = np.concatenate(histograms)
    tiles = np.asarray([
        sum(math.ceil(count / max(1, int(row.get("runtime_config", {}).get("BLOCK_SIZE_M", 64))))
            for count in row["expert_histogram"] if count)
        for row in values
    ], dtype=float)
    traffic = np.zeros((ep, ep), dtype=np.int64)
    local_experts = NUM_EXPERTS // ep
    for row in values:
        source = int(row["ep_rank"])
        with np.load(raw / row["route_file"]) as archive:
            ids = archive["topk_ids"].astype(np.int64)
        destinations = ids // local_experts
        for index in range(len(ids)):
            np.add.at(traffic[source], np.unique(destinations[index]), 1)
    remote = traffic.copy()
    np.fill_diagonal(remote, 0)
    request_ids = tuple(sorted({str(row["request_id"]) for row in values}))
    padded_traffic = np.zeros((8, 8), dtype=np.int64)
    padded_traffic[:ep, :ep] = traffic
    layer_onehot = np.zeros(48, dtype=float)
    layer_onehot[int(values[0]["layer"])] = 1.0
    summary_features = [
        math.log1p(source_rows.sum()), math.log1p(source_rows.max()), _cv(source_rows),
        math.log1p(assignments.sum()), math.log1p(assignments.max()), _cv(assignments),
        math.log1p(dispatched.sum()), math.log1p(dispatched.max()), _cv(dispatched),
        float(np.count_nonzero(histogram)), math.log1p(histogram.max(initial=0)),
        math.log1p(tiles.sum()), math.log1p(tiles.max(initial=0)),
        math.log1p(remote.sum()), math.log1p(remote.sum(1).max(initial=0)),
        math.log1p(remote.sum(0).max(initial=0)),
        float((traffic > 0).sum(1).max(initial=0)),
        int(values[0]["layer"]) / 47.0,
    ]
    return {
        "wave": int(values[0]["wave"]),
        "iteration": int(values[0]["iteration"]),
        "layer": int(values[0]["layer"]),
        "request_ids": request_ids,
        "features": np.concatenate([
            np.asarray(summary_features, dtype=float), np.log1p(histogram),
            layer_onehot, np.log1p(padded_traffic.reshape(-1)),
        ]),
        "targets": np.asarray([
            max(float(row["dispatch"].get("critical_ms", row["dispatch"]["ms"])) for row in values),
            max(float(row["expert"]["ms"]) for row in values),
            max(float(row["combine"].get("critical_ms", row["combine"]["ms"])) for row in values),
        ], dtype=float),
        "traffic_matrix": traffic,
    }


def _load(root: Path, manifest: dict[str, Any], ep: int) -> list[dict[str, Any]]:
    by_id = {str(row["request_id"]): row for row in manifest["requests"]}
    raw = root / "profile/raw_live"
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for name in sorted(glob.glob(str(raw / "rank[0-9]*.jsonl"))):
        for row in _jsonl(Path(name)):
            if bool(row.get("measured", True)):
                grouped[(int(row["wave"]), int(row["layer"]))].append(row)
    repeats = [_one_invocation(values, raw, ep) for values in grouped.values()]

    # Collapse repeated clean measurements before fitting or scoring.  This
    # makes the cost target the repeat-median target-hardware latency rather
    # than one externally contended sample.
    buckets: dict[tuple[tuple[str, ...], int], list[dict[str, Any]]] = defaultdict(list)
    for row in repeats:
        buckets[(row["request_ids"], row["layer"])].append(row)
    samples = []
    for (request_ids, layer), rows in sorted(buckets.items()):
        splits = {by_id[request_id]["anchor_split"] for request_id in request_ids}
        if len(splits) != 1:
            raise AssertionError((request_ids, splits))
        samples.append({
            "request_ids": request_ids,
            "request_group": "+".join(request_ids),
            "layer": layer,
            "split": next(iter(splits)),
            "features": np.median(np.stack([row["features"] for row in rows]), axis=0),
            "targets": np.median(np.stack([row["targets"] for row in rows]), axis=0),
            "repeat_targets": {
                int(row["iteration"]): row["targets"] for row in rows
            },
            "repeat_count": len(rows),
            "repeat_stage_cv": _cv(np.asarray([row["targets"].sum() for row in rows])),
        })
    return samples


class Ridge:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        self.feature_min = x.min(0)
        self.feature_max = x.max(0)
        self.mean = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-8] = 1
        design = np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])
        return np.maximum(0, design @ self.coef)

    def state(self) -> dict[str, Any]:
        return {"type": "standardized_ridge", "alpha": self.alpha,
                "feature_indices": getattr(self, "feature_indices", None),
                "feature_variant": getattr(self, "feature_variant", "all"),
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(),
                "feature_min": self.feature_min.tolist(),
                "feature_max": self.feature_max.tolist()}


class KNN:
    def __init__(self, neighbors: int):
        self.neighbors = neighbors

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNN":
        self.feature_min = x.min(0)
        self.feature_max = x.max(0)
        self.mean = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-8] = 1
        self.x = (x - self.mean) / self.scale
        self.y = y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        query = (x - self.mean) / self.scale
        output = []
        for row in query:
            distance = np.sqrt(np.square(self.x - row).sum(1))
            exact = np.flatnonzero(distance < 1e-10)
            if len(exact):
                output.append(self.y[exact].mean(0))
                continue
            chosen = np.argsort(distance)[: min(self.neighbors, len(distance))]
            weights = 1.0 / np.maximum(distance[chosen], 1e-8)
            output.append((self.y[chosen] * weights[:, None]).sum(0) / weights.sum())
        return np.maximum(0, np.asarray(output))

    def state(self) -> dict[str, Any]:
        return {"type": "standardized_inverse_distance_knn",
                "feature_indices": getattr(self, "feature_indices", None),
                "feature_variant": getattr(self, "feature_variant", "all"),
                "neighbors": self.neighbors, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "normalized_x": self.x.tolist(),
                "targets_ms": self.y.tolist(),
                "feature_min": self.feature_min.tolist(),
                "feature_max": self.feature_max.tolist()}


class Componentwise:
    """Frozen stage-specific models selected without looking at final test."""

    def __init__(self, models: list[Any], names: list[str],
                 feature_min: np.ndarray, feature_max: np.ndarray):
        self.models = models
        self.names = names
        self.feature_min = feature_min
        self.feature_max = feature_max

    def predict(self, x: np.ndarray) -> np.ndarray:
        columns = []
        for component, model in enumerate(self.models):
            indexes = np.asarray(model.feature_indices, dtype=int)
            columns.append(model.predict(x[:, indexes])[:, component])
        return np.column_stack(columns)

    def state(self) -> dict[str, Any]:
        return {
            "type": "componentwise",
            "selected_component_models": dict(zip(COMPONENTS, self.names)),
            "component_models": [model.state() for model in self.models],
            "feature_indices": None,
            "feature_variant": "componentwise",
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
        }


def _ape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-4) * 100


def _summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = _ape(actual, predicted)
    return {
        "n": len(actual),
        "median_ape_percent": float(np.median(error)),
        "p90_ape_percent": float(np.quantile(error, 0.9)),
        "median_signed_bias_ms": float(np.median(predicted - actual)),
        "median_absolute_error_ms": float(np.median(np.abs(predicted - actual))),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Small dependency-free average-rank implementation for Spearman."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2.0
        begin = end
    return ranks


def _ranking(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if len(actual) < 2:
        return {"scenario_pairs": 0, "material_pairs": 0,
                "sign_accuracy": None, "spearman": None,
                "gain_error_median_pp": None, "gain_error_p90_pp": None}
    actual_rank = _rankdata(actual)
    predicted_rank = _rankdata(predicted)
    spearman = float(np.corrcoef(actual_rank, predicted_rank)[0, 1])
    gain_error = []
    signs = []
    total = 0
    for left in range(len(actual)):
        for right in range(left + 1, len(actual)):
            total += 1
            # Symmetric materiality avoids choosing one workload as an
            # arbitrary denominator.  Only >=5% actual differences count.
            material = 200.0 * abs(actual[right] - actual[left]) / max(
                actual[right] + actual[left], 1e-8
            )
            if material < 5.0:
                continue
            actual_delta = 100.0 * (actual[right] - actual[left]) / max(actual[left], 1e-8)
            predicted_delta = 100.0 * (predicted[right] - predicted[left]) / max(predicted[left], 1e-8)
            signs.append(np.sign(actual_delta) == np.sign(predicted_delta))
            gain_error.append(abs(predicted_delta - actual_delta))
    return {
        "scenario_pairs": total,
        "material_pairs": len(gain_error),
        "materiality_rule": "actual symmetric difference >=5%",
        "sign_accuracy": float(np.mean(signs)) if signs else None,
        "spearman": spearman,
        "gain_error_median_pp": float(np.median(gain_error)) if gain_error else None,
        "gain_error_p90_pp": float(np.quantile(gain_error, 0.9)) if gain_error else None,
    }


def _bootstrap_error_ci(actual: np.ndarray, predicted: np.ndarray,
                        seed: int, repeats: int = 2000) -> dict[str, Any]:
    """Request-group bootstrap; layer invocations are never iid samples."""
    if not len(actual):
        return {"clusters": 0, "median_ape_95ci_percent": None,
                "p90_ape_95ci_percent": None}
    error = _ape(actual, predicted)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(error), size=(repeats, len(error)))
    draws = error[indexes]
    median = np.median(draws, axis=1)
    p90 = np.quantile(draws, 0.9, axis=1)
    return {
        "clusters": len(actual),
        "bootstrap_repeats": repeats,
        "unit": "request_group",
        "median_ape_95ci_percent": [float(np.quantile(median, 0.025)),
                                     float(np.quantile(median, 0.975))],
        "p90_ape_95ci_percent": [float(np.quantile(p90, 0.025)),
                                  float(np.quantile(p90, 0.975))],
    }


def _request_arrays(samples: list[dict[str, Any]], predicted: np.ndarray,
                    split: str, *, coherent_repeat_target: bool = False
                    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(samples):
        if row["split"] == split:
            groups[row["request_group"]].append(index)
    names = sorted(groups)
    actual_values = []
    for name in names:
        indexes = groups[name]
        if not coherent_repeat_target:
            actual_values.append(sum(samples[index]["targets"].sum() for index in indexes))
            continue
        totals: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for index in indexes:
            for iteration, target in samples[index]["repeat_targets"].items():
                totals[int(iteration)] += float(np.asarray(target).sum())
                counts[int(iteration)] += 1
        complete = [totals[iteration] for iteration in sorted(totals)
                    if counts[iteration] == len(indexes)]
        if not complete:
            raise ValueError(f"no complete request repeat for {name}")
        actual_values.append(float(np.median(complete)))
    actual = np.asarray(actual_values)
    estimate = np.asarray([sum(predicted[index].sum() for index in groups[name])
                           for name in names])
    return actual, estimate, names


def _request_repeat_variability(samples: list[dict[str, Any]], split: str
                                ) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        if row["split"] == split:
            groups[row["request_group"]].append(row)
    group_rows = []
    all_ape = []
    for name, rows in sorted(groups.items()):
        totals: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for row in rows:
            for iteration, target in row["repeat_targets"].items():
                totals[int(iteration)] += float(np.asarray(target).sum())
                counts[int(iteration)] += 1
        complete = np.asarray([
            totals[iteration] for iteration in sorted(totals)
            if counts[iteration] == len(rows)
        ], dtype=float)
        if not len(complete):
            continue
        median = float(np.median(complete))
        ape = np.abs(complete - median) / max(abs(median), 1e-4) * 100.0
        all_ape.extend(ape.tolist())
        group_rows.append({
            "request_group": name,
            "repeat_count": int(len(complete)),
            "median_ms": median,
            "cv": _cv(complete),
            "repeat_vs_median_ape_p90_percent": float(np.quantile(ape, 0.9)),
        })
    cvs = np.asarray([row["cv"] for row in group_rows], dtype=float)
    all_ape_array = np.asarray(all_ape, dtype=float)
    return {
        "request_groups": len(group_rows),
        "repeat_count_min": min((row["repeat_count"] for row in group_rows), default=0),
        "repeat_count_max": max((row["repeat_count"] for row in group_rows), default=0),
        "request_cv_median": float(np.median(cvs)) if len(cvs) else None,
        "request_cv_p90": float(np.quantile(cvs, 0.9)) if len(cvs) else None,
        "repeat_vs_group_median_ape_median_percent": (
            float(np.median(all_ape_array)) if len(all_ape_array) else None
        ),
        "repeat_vs_group_median_ape_p90_percent": (
            float(np.quantile(all_ape_array, 0.9)) if len(all_ape_array) else None
        ),
        "per_request_group": group_rows,
    }


def _request_component_arrays(samples: list[dict[str, Any]], predicted: np.ndarray,
                              split: str, component: int
                              ) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(samples):
        if row["split"] == split:
            groups[row["request_group"]].append(index)
    actual_values = []
    estimate_values = []
    for name in sorted(groups):
        indexes = groups[name]
        totals: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for index in indexes:
            for iteration, target in samples[index]["repeat_targets"].items():
                totals[int(iteration)] += float(np.asarray(target)[component])
                counts[int(iteration)] += 1
        complete = [totals[iteration] for iteration in sorted(totals)
                    if counts[iteration] == len(indexes)]
        if not complete:
            raise ValueError(f"no complete component repeat for {name}")
        actual_values.append(float(np.median(complete)))
        estimate_values.append(float(sum(predicted[index, component] for index in indexes)))
    return np.asarray(actual_values), np.asarray(estimate_values)


def _fit(samples: list[dict[str, Any]], *, coherent_repeat_target: bool = False
         ) -> tuple[Any, dict[str, Any]]:
    x = np.stack([row["features"] for row in samples])
    y = np.stack([row["targets"] for row in samples])
    train = np.asarray([row["split"] == "calibration" for row in samples])
    candidates: list[tuple[str, str, Any, np.ndarray]] = []
    layer_begin = len(BASE_FEATURE_NAMES) + NUM_EXPERTS
    traffic_begin = layer_begin + 48
    variants = {
        "aggregate": np.arange(len(BASE_FEATURE_NAMES)),
        "aggregate_layer": np.r_[np.arange(len(BASE_FEATURE_NAMES)),
                                  np.arange(layer_begin, traffic_begin)],
        "aggregate_traffic": np.r_[np.arange(len(BASE_FEATURE_NAMES)),
                                    np.arange(traffic_begin, len(FEATURE_NAMES))],
        "structural": np.arange(len(FEATURE_NAMES)),
    }
    for variant, indexes in variants.items():
        for alpha in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
            candidates.append((f"{variant}_ridge_alpha_{alpha:g}", variant,
                               Ridge(alpha), indexes))
        for neighbors in (4, 8, 16, 32):
            candidates.append((f"{variant}_knn_{neighbors}", variant,
                               KNN(neighbors), indexes))
    selection = []
    fitted = []
    candidate_records = []
    for name, variant, model, indexes in candidates:
        model.feature_indices = indexes.astype(int).tolist()
        model.feature_variant = variant
        model.fit(x[train][:, indexes], y[train])
        prediction = model.predict(x[:, indexes])
        actual, estimate, _ = _request_arrays(
            samples, prediction, "development",
            coherent_repeat_target=coherent_repeat_target,
        )
        metric = _summary(actual, estimate)
        selection_score = max(metric["median_ape_percent"] / 10.0,
                              metric["p90_ape_percent"] / 20.0)
        selection.append({"candidate": name, "selection_score": selection_score, **metric})
        fitted.append((selection_score, metric["median_ape_percent"], name, model, prediction))
        candidate_records.append((name, model, prediction))
    component_models = []
    component_names = []
    component_selection = {}
    for component, component_name in enumerate(COMPONENTS):
        choices = []
        for name, model, prediction in candidate_records:
            actual, estimate = _request_component_arrays(
                samples, prediction, "development", component
            )
            metric = _summary(actual, estimate)
            score = max(metric["median_ape_percent"] / 10.0,
                        metric["p90_ape_percent"] / 20.0)
            choices.append((score, metric["median_ape_percent"], name, model, metric))
        choices.sort(key=lambda item: (item[0], item[1], item[2]))
        score, _, name, model, metric = choices[0]
        component_models.append(model)
        component_names.append(name)
        component_selection[component_name] = {
            "candidate": name, "selection_score": score, **metric,
        }
    composite = Componentwise(component_models, component_names, x[train].min(0), x[train].max(0))
    composite_prediction = composite.predict(x)
    actual, estimate, _ = _request_arrays(
        samples, composite_prediction, "development",
        coherent_repeat_target=coherent_repeat_target,
    )
    composite_metric = _summary(actual, estimate)
    composite_score = max(composite_metric["median_ape_percent"] / 10.0,
                          composite_metric["p90_ape_percent"] / 20.0)
    selection.append({
        "candidate": "componentwise_stage_models",
        "selection_score": composite_score,
        "component_selection": component_selection,
        **composite_metric,
    })
    fitted.append((composite_score, composite_metric["median_ape_percent"],
                   "componentwise_stage_models", composite, composite_prediction))
    fitted.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, selected_name, model, prediction = fitted[0]
    metrics = {}
    request_values = {}
    for split_index, split in enumerate(("calibration", "development", "untouched_test")):
        mask = np.asarray([row["split"] == split for row in samples])
        actual_request, predicted_request, names = _request_arrays(
            samples, prediction, split,
            coherent_repeat_target=coherent_repeat_target,
        )
        metrics[split] = {
            "invocation": _summary(y[mask].sum(1), prediction[mask].sum(1)),
            "request_routed_stage": _summary(actual_request, predicted_request),
            "request_ranking_and_gain": _ranking(actual_request, predicted_request),
            "request_cluster_bootstrap": _bootstrap_error_ci(
                actual_request, predicted_request, 3407 + split_index
            ),
            "physical_repeat_variability": _request_repeat_variability(samples, split),
        }
        request_values[split] = [
            {"request_group": name, "actual_ms": float(actual), "predicted_ms": float(estimate)}
            for name, actual, estimate in zip(names, actual_request, predicted_request)
        ]
    dev_actual = np.asarray([row["actual_ms"] for row in request_values["development"]])
    dev_predicted = np.asarray([row["predicted_ms"] for row in request_values["development"]])
    radius_ms = float(np.quantile(np.abs(dev_predicted - dev_actual), 0.9))
    test = request_values["untouched_test"]
    coverage = (sum(abs(row["predicted_ms"] - row["actual_ms"]) <= radius_ms for row in test) / len(test)
                if test else None)
    normalized_width = (float(np.median([2 * radius_ms / row["actual_ms"] for row in test]))
                        if test else None)
    return model, {
        "selected_on_development": selected_name,
        "selection_objective": "minimize max(median_APE/10, P90_APE/20), then median APE",
        "candidate_selection": selection,
        "metrics": metrics,
        "request_values": request_values,
        "prediction_interval": {
            "method": "development request absolute-residual p90",
            "half_width_ms": radius_ms,
            "untouched_empirical_coverage": coverage,
            "untouched_median_normalized_full_width": normalized_width,
            "small_sample_warning": True,
        },
        "repeat_count_min": min(row["repeat_count"] for row in samples),
        "repeat_count_max": max(row["repeat_count"] for row in samples),
        "repeat_stage_cv_median": float(np.median([row["repeat_stage_cv"] for row in samples])),
        "observed_invocations_after_repeat_median": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ep2-root", type=Path, required=True)
    parser.add_argument("--ep4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-test-manifest", type=Path)
    parser.add_argument("--ep2-final-test-root", type=Path)
    parser.add_argument("--ep4-final-test-root", type=Path)
    parser.add_argument("--additional-development-manifest", type=Path)
    parser.add_argument("--ep2-additional-development-root", type=Path)
    parser.add_argument("--ep4-additional-development-root", type=Path)
    parser.add_argument("--supplemental-manifest", type=Path)
    parser.add_argument("--ep2-supplemental-root", type=Path)
    parser.add_argument("--ep4-supplemental-root", type=Path)
    parser.add_argument(
        "--coherent-request-target", action="store_true",
        help=("score a request against the median of complete repeat totals; "
              "the legacy target sums independent per-layer medians"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = {
        "schema_version": (
            "mllm_ep_cost_model.v6"
            if args.coherent_request_target or args.supplemental_manifest
            else "mllm_ep_cost_model.v4"
        ),
        "feature_names": FEATURE_NAMES,
        "components": COMPONENTS,
        "models": {}, "validation": {},
        "fit_policy": "calibration only; model selection on development; untouched test scored after freeze",
    }
    for ep, root in ((2, args.ep2_root), (4, args.ep4_root)):
        samples = _load(root, manifest, ep)
        development_root = (args.ep2_additional_development_root if ep == 2
                            else args.ep4_additional_development_root)
        development_args = (
            args.additional_development_manifest,
            args.ep2_additional_development_root,
            args.ep4_additional_development_root,
        )
        if any(value is not None for value in development_args):
            if any(value is None for value in development_args):
                raise ValueError(
                    "additional-development manifest and both topology roots are required together"
                )
            development_manifest = json.loads(
                args.additional_development_manifest.read_text()
            )
            development_samples = _load(
                development_root, development_manifest, ep
            )
            for sample in development_samples:
                sample["split"] = "development"
            samples.extend(development_samples)
        supplemental_root = (args.ep2_supplemental_root if ep == 2
                             else args.ep4_supplemental_root)
        supplemental_args = (
            args.supplemental_manifest,
            args.ep2_supplemental_root,
            args.ep4_supplemental_root,
        )
        if any(value is not None for value in supplemental_args):
            if any(value is None for value in supplemental_args):
                raise ValueError(
                    "supplemental manifest and both topology roots are required together"
                )
            # Once a repair adds a fresh final split, the former test has
            # already informed the repair and becomes development evidence.
            for sample in samples:
                if sample["split"] == "untouched_test":
                    sample["split"] = "development"
            supplemental_manifest = json.loads(
                args.supplemental_manifest.read_text()
            )
            supplemental_samples = _load(
                supplemental_root, supplemental_manifest, ep
            )
            if not any(row["split"] == "untouched_test"
                       for row in supplemental_samples):
                raise ValueError("supplemental root has no fresh untouched test")
            samples.extend(supplemental_samples)
        final_root = args.ep2_final_test_root if ep == 2 else args.ep4_final_test_root
        if args.final_test_manifest or final_root:
            if args.final_test_manifest is None or final_root is None:
                raise ValueError("final-test manifest and both topology roots are required together")
            # The original untouched split has already informed the feature
            # repair and is therefore development data now.  Only the newly
            # collected image-disjoint split remains untouched.
            for sample in samples:
                if sample["split"] == "untouched_test":
                    sample["split"] = "development"
            final_manifest = json.loads(args.final_test_manifest.read_text())
            final_samples = _load(final_root, final_manifest, ep)
            if any(row["split"] != "untouched_test" for row in final_samples):
                raise ValueError("new final-test root contains non-test samples")
            samples.extend(final_samples)
        model, validation = _fit(
            samples, coherent_repeat_target=args.coherent_request_target
        )
        output["models"][f"ep{ep}"] = model.state()
        output["validation"][f"ep{ep}"] = validation
    output["evidence"] = {
        "ep2": "ACTUAL_TP1_DP2_EP2_DEEPEP_REPEAT_MEDIAN",
        "ep4": "ACTUAL_TP2_DP2_EP4_DEEPEP_REPEAT_MEDIAN",
        "ep8": "UNVALIDATED_EXTRAPOLATION",
    }
    has_fresh_test = bool(args.final_test_manifest or args.supplemental_manifest)
    output["test_policy"] = (
        "original test promoted to development after feature repair; new image-disjoint final test scored once"
        if has_fresh_test else
        "single frozen split; do not use untouched result for further model changes"
    )
    output["request_target_policy"] = (
        "median_of_complete_repeat_totals"
        if args.coherent_request_target else
        "sum_of_per_layer_component_medians"
    )
    atomic_json(args.output, output)
    print(json.dumps(output["validation"], indent=2))


if __name__ == "__main__":
    main()
