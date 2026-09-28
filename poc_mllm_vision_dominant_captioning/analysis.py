from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .contract import DOMAINS, NUM_EXPERTS, NUM_LAYERS, WINDOWS
from .trace_data import dense

PREDICTORS = ("P_domain", "P_text", "P_full", "P_separated", "P_vision")


def _normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    total = vector.sum(axis=axis, keepdims=True)
    return np.divide(vector, total, out=np.zeros_like(vector, dtype=np.float64), where=total > 0)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.reshape(-1), right.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def jsd(left: np.ndarray, right: np.ndarray) -> float:
    p = _normalize(left.reshape(-1))
    q = _normalize(right.reshape(-1))
    middle = 0.5 * (p + q)
    valid_p, valid_q = p > 0, q > 0
    kl_p = float(np.sum(p[valid_p] * np.log(p[valid_p] / middle[valid_p])))
    kl_q = float(np.sum(q[valid_q] * np.log(q[valid_q] / middle[valid_q])))
    return (kl_p + kl_q) / (2.0 * math.log(2.0)) if middle.any() else 0.0


def _layers(rows: list[dict[str, float]]) -> np.ndarray:
    return np.stack([dense(row) for row in rows])


def materialize(row: dict[str, Any]) -> dict[str, Any]:
    vision_count = _layers([layer["vision_count"] for layer in row["prefill_layers"]])
    vision_mass = _layers([layer["vision_mass"] for layer in row["prefill_layers"]])
    text_count = _layers([layer["text_count"] for layer in row["prefill_layers"]])
    text_mass = _layers([layer["text_mass"] for layer in row["prefill_layers"]])
    decode = {}
    for window in WINDOWS:
        payload = row["decode_windows"][window]
        count = _layers([layer["count"] for layer in payload["layers"]])
        mass = _layers([layer["mass"] for layer in payload["layers"]])
        source = np.stack([
            np.stack([dense(item) for item in layer["source_expert"]])
            for layer in payload["layers"]
        ])
        decode[window] = {
            "count": count,
            "mass": mass,
            "source": source,
            "observed_steps": payload["observed_steps"],
        }
    features = {
        "P_text": _normalize(text_mass).reshape(-1),
        "P_full": _normalize(vision_mass + text_mass).reshape(-1),
        "P_separated": np.concatenate((_normalize(vision_mass).reshape(-1), _normalize(text_mass).reshape(-1))),
        "P_vision": _normalize(vision_mass).reshape(-1),
    }
    features = {key: value / max(float(np.linalg.norm(value)), 1e-12) for key, value in features.items()}
    return {
        "request_id": row["request_id"],
        "domain": row["domain"],
        "analysis_split": row["analysis_split"],
        "topology": row["topology"],
        "ep_size": row["ep_size"],
        "output_token_count": row["output_token_count"],
        "finish_reason": row["finish_reason"],
        "vision_token_count": row["vision_token_count"],
        "vision_count": vision_count,
        "vision_mass": vision_mass,
        "text_count": text_count,
        "text_mass": text_mass,
        "features": features,
        "decode": decode,
        "sampled_routes": row.get("sampled_routes", []),
    }


def rank_histogram(expert: np.ndarray, ep_size: int) -> np.ndarray:
    shape = expert.shape[:-1] + (ep_size, NUM_EXPERTS // ep_size)
    return expert.reshape(shape).sum(axis=-1)


def remote_fraction(source_expert: np.ndarray, ep_size: int) -> float:
    total = float(source_expert.sum())
    if not total:
        return 0.0
    source_count = source_expert.shape[-2]
    source_rank = np.arange(source_count, dtype=np.int64) % ep_size
    expert_rank = np.arange(NUM_EXPERTS, dtype=np.int64) // (NUM_EXPERTS // ep_size)
    remote_mask = source_rank[:, None] != expert_rank[None, :]
    flattened = source_expert.reshape((-1, source_count, NUM_EXPERTS))
    return float(flattened[:, remote_mask].sum()) / total


def _uniform_target(sample: dict[str, Any], window: str) -> dict[str, np.ndarray]:
    target = sample["decode"][window]
    source_shape = target["source"].shape
    count = np.ones((NUM_LAYERS, NUM_EXPERTS), dtype=np.float64)
    source = np.ones(source_shape, dtype=np.float64)
    return {"count": count, "mass": count.copy(), "source": source}


def predict(
    sample: dict[str, Any], memory: list[dict[str, Any]], predictor: str, window: str,
    k: int = 3,
) -> tuple[dict[str, np.ndarray], list[str]]:
    eligible = [item for item in memory if item["decode"][window]["count"].sum() > 0]
    if predictor == "P_domain":
        same = [item for item in eligible if item["domain"] == sample["domain"]]
        chosen = same or eligible
        chosen = chosen[:]
    else:
        feature = sample["features"][predictor]
        chosen = sorted(
            eligible,
            key=lambda item: float(np.dot(feature, item["features"][predictor])),
            reverse=True,
        )[:k]
    if predictor == "P_domain" and chosen:
        # A prototype is the mean of every causally available same-domain row.
        selected = chosen
    else:
        selected = chosen[:k]
    if not selected:
        return _uniform_target(sample, window), []
    return ({
        key: np.mean([item["decode"][window][key] for item in selected], axis=0)
        for key in ("count", "mass", "source")
    }, [item["request_id"] for item in selected])


def prediction_metrics(truth: dict[str, np.ndarray], prediction: dict[str, np.ndarray], ep_size: int) -> dict[str, float]:
    true_count, pred_count = truth["count"], prediction["count"]
    valid_layers = [layer for layer in range(NUM_LAYERS) if true_count[layer].sum() > 0]
    if not valid_layers:
        return {}
    hist_cos, hist_jsd, weight_cos, weight_jsd, topk = [], [], [], [], []
    for layer in valid_layers:
        hist_cos.append(cosine(true_count[layer], pred_count[layer]))
        hist_jsd.append(jsd(true_count[layer], pred_count[layer]))
        weight_cos.append(cosine(truth["mass"][layer], prediction["mass"][layer]))
        weight_jsd.append(jsd(truth["mass"][layer], prediction["mass"][layer]))
        true_top = set(np.argsort(true_count[layer])[-8:])
        pred_top = set(np.argsort(pred_count[layer])[-8:])
        topk.append(len(true_top & pred_top) / 8.0)
    true_active = set(np.flatnonzero(true_count.sum(axis=0)))
    pred_active = set(np.flatnonzero(pred_count.sum(axis=0)))
    active_recall = len(true_active & pred_active) / len(true_active) if true_active else 1.0
    active_precision = len(true_active & pred_active) / len(pred_active) if pred_active else 0.0

    true_rank = rank_histogram(true_count, ep_size)
    pred_rank = rank_histogram(pred_count, ep_size)
    rank_cos, rank_js, rank_mae, hot, maxmean = [], [], [], [], []
    for layer in valid_layers:
        a, b = true_rank[layer], pred_rank[layer]
        an, bn = _normalize(a), _normalize(b)
        rank_cos.append(cosine(a, b))
        rank_js.append(jsd(a, b))
        rank_mae.append(float(np.mean(np.abs(an - bn))))
        hot.append(float(int(np.argmax(a)) == int(np.argmax(b))))
        a_ratio = float(a.max() / max(a.mean(), 1e-12))
        b_ratio = float(b.max() / max(b.mean(), 1e-12))
        maxmean.append(abs(a_ratio - b_ratio))
    true_remote = remote_fraction(truth["source"], ep_size)
    pred_remote = remote_fraction(prediction["source"], ep_size)
    return {
        "expert_histogram_cosine": float(np.mean(hist_cos)),
        "expert_histogram_jsd": float(np.mean(hist_jsd)),
        "expert_weight_histogram_cosine": float(np.mean(weight_cos)),
        "expert_weight_histogram_jsd": float(np.mean(weight_jsd)),
        "decode_expert_top8_recall": float(np.mean(topk)),
        "active_expert_set_recall": active_recall,
        "active_expert_set_precision": active_precision,
        "active_expert_union_abs_error": abs(len(pred_active) - len(true_active)),
        "rank_distribution_cosine": float(np.mean(rank_cos)),
        "rank_distribution_jsd": float(np.mean(rank_js)),
        "per_rank_load_mae": float(np.mean(rank_mae)),
        "hottest_rank_accuracy": float(np.mean(hot)),
        "rank_max_mean_abs_error": float(np.mean(maxmean)),
        "remote_destination_abs_error": abs(pred_remote - true_remote),
    }


def evaluate_stream(
    samples: dict[str, dict[str, Any]], stream: list[str], initial_memory: list[str],
    predictor: str, window: str, ep_size: int, continual: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memory_ids = list(initial_memory)
    metrics, events = [], []
    for event_index, request_id in enumerate(stream):
        sample = samples[request_id]
        if request_id in memory_ids:
            raise AssertionError("current request leaked into calibration memory")
        memory = [samples[item] for item in memory_ids]
        prediction, neighbors = predict(sample, memory, predictor, window)
        truth = sample["decode"][window]
        measured = prediction_metrics(truth, prediction, ep_size)
        if measured:
            metrics.append({
                "request_id": request_id,
                "domain": sample["domain"],
                "predictor": predictor,
                "window": window,
                "continual": continual,
                **measured,
            })
        events.append({
            "event_index": event_index,
            "logical_prediction_timestamp": event_index * 2,
            "logical_decode_completion_timestamp": event_index * 2 + 1,
            "request_id": request_id,
            "domain": sample["domain"],
            "predictor": predictor,
            "window": window,
            "memory_size_before_prediction": len(memory_ids),
            "memory_last_completed_request": memory_ids[-1] if memory_ids else None,
            "neighbor_request_ids": neighbors,
            "current_truth_available_before_prediction": False,
            "prediction_recorded_before_decode_truth_update": True,
        })
        if continual:
            memory_ids.append(request_id)
            events[-1]["memory_size_after_completion"] = len(memory_ids)
    return metrics, events


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key not in {"continual"}]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def paired_bootstrap(
    candidate: list[dict[str, Any]], control: list[dict[str, Any]], metric: str,
    higher_is_better: bool, repeats: int = 2000, seed: int = 3407,
) -> dict[str, float | int]:
    left = {row["request_id"]: float(row[metric]) for row in candidate}
    right = {row["request_id"]: float(row[metric]) for row in control}
    ids = sorted(left.keys() & right.keys())
    sign = 1.0 if higher_is_better else -1.0
    values = np.asarray([sign * (left[item] - right[item]) for item in ids])
    if not len(values):
        return {"requests": 0, "beneficial_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = np.mean(values[rng.integers(0, len(values), size=(repeats, len(values)))], axis=1)
    return {
        "requests": len(values),
        "beneficial_delta": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def separation(samples: Iterable[dict[str, Any]], predictor: str) -> dict[str, Any]:
    rows = list(samples)
    vectors = np.stack([row["features"][predictor] for row in rows])
    domains = [row["domain"] for row in rows]
    similarity = vectors @ vectors.T
    distance = 1.0 - similarity
    within, between, silhouettes, correct = [], [], [], 0
    for index, domain in enumerate(domains):
        same = [j for j, other in enumerate(domains) if other == domain and j != index]
        other_by_domain = [[j for j, other in enumerate(domains) if other == target] for target in DOMAINS if target != domain]
        within.extend(distance[index, same].tolist())
        for group in other_by_domain:
            between.extend(distance[index, group].tolist())
        a = float(np.mean(distance[index, same])) if same else 0.0
        b = min(float(np.mean(distance[index, group])) for group in other_by_domain if group)
        silhouettes.append((b - a) / max(a, b, 1e-12))
        candidates = [j for j in range(len(rows)) if j != index]
        nearest = min(candidates, key=lambda j: distance[index, j])
        correct += int(domains[nearest] == domain)
    centroids = {}
    for domain in DOMAINS:
        centroid = vectors[[i for i, value in enumerate(domains) if value == domain]].mean(axis=0)
        centroids[domain] = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    matrix = {
        left: {right: float(1.0 - np.dot(centroids[left], centroids[right])) for right in DOMAINS}
        for left in DOMAINS
    }
    return {
        "within_domain_cosine_distance": float(np.mean(within)),
        "between_domain_cosine_distance": float(np.mean(between)),
        "between_minus_within": float(np.mean(between) - np.mean(within)),
        "nearest_neighbor_same_domain_rate": correct / len(rows),
        "silhouette": float(np.mean(silhouettes)),
        "centroid_distance_matrix": matrix,
    }


def layer_separation(samples: Iterable[dict[str, Any]], channel: str) -> list[dict[str, float]]:
    rows = list(samples)
    output = []
    for layer in range(NUM_LAYERS):
        vectors = []
        for row in rows:
            if channel == "P_vision":
                value = _normalize(row["vision_mass"][layer])
            elif channel == "P_text":
                value = _normalize(row["text_mass"][layer])
            elif channel == "P_full":
                value = _normalize(row["vision_mass"][layer] + row["text_mass"][layer])
            else:
                value = np.concatenate((_normalize(row["vision_mass"][layer]), _normalize(row["text_mass"][layer])))
            vectors.append(value / max(float(np.linalg.norm(value)), 1e-12))
        matrix = np.stack(vectors)
        similarity = matrix @ matrix.T
        within, between = [], []
        for i, row in enumerate(rows):
            for j in range(i + 1, len(rows)):
                target = within if row["domain"] == rows[j]["domain"] else between
                target.append(1.0 - float(similarity[i, j]))
        output.append({
            "layer": layer,
            "within": float(np.mean(within)),
            "between": float(np.mean(between)),
            "between_minus_within": float(np.mean(between) - np.mean(within)),
        })
    return output


def _greedy_groups(ids: list[str], vectors: dict[str, np.ndarray], size: int) -> list[list[str]]:
    remaining = list(ids)
    groups = []
    while remaining:
        group = [remaining.pop(0)]
        while remaining and len(group) < size:
            prototype = np.mean([vectors[item] for item in group], axis=0)
            prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
            choice = max(remaining, key=lambda item: float(np.dot(prototype, vectors[item])))
            remaining.remove(choice)
            group.append(choice)
        groups.append(group)
    return groups


def _oracle_groups(ids: list[str], samples: dict[str, dict[str, Any]], size: int) -> list[list[str]]:
    sets = {
        request: [set(np.flatnonzero(layer)) for layer in samples[request]["decode"]["1-256"]["count"]]
        for request in ids
    }
    remaining = list(ids)
    groups = []
    while remaining:
        group = [remaining.pop(0)]
        unions = [set(value) for value in sets[group[0]]]
        while remaining and len(group) < size:
            def added(candidate: str) -> int:
                return sum(len(unions[layer] | sets[candidate][layer]) - len(unions[layer]) for layer in range(NUM_LAYERS))
            choice = min(remaining, key=lambda candidate: (added(candidate), candidate))
            remaining.remove(choice)
            group.append(choice)
            for layer in range(NUM_LAYERS):
                unions[layer] |= sets[choice][layer]
        groups.append(group)
    def cost(group: list[str]) -> int:
        return sum(len(set().union(*(sets[item][layer] for item in group))) for layer in range(NUM_LAYERS))
    # Bounded deterministic pair-swap refinement of the future-truth greedy partition.
    for _ in range(5):
        improved = False
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                old = cost(groups[left]) + cost(groups[right])
                accepted = False
                for i in range(len(groups[left])):
                    for j in range(len(groups[right])):
                        a, b = groups[left][i], groups[right][j]
                        groups[left][i], groups[right][j] = b, a
                        if cost(groups[left]) + cost(groups[right]) < old:
                            improved = accepted = True
                            break
                        groups[left][i], groups[right][j] = a, b
                    if accepted: break
        if not improved: break
    return groups


def _group_metric(groups: list[list[str]], samples: dict[str, dict[str, Any]], ep_size: int) -> dict[str, float]:
    expert_unions, rank_unions, pairs, active, maxmeans, remote = [], [], [], [], [], []
    for group in groups:
        counts = [samples[item]["decode"]["1-256"]["count"] for item in group]
        sources = [samples[item]["decode"]["1-256"]["source"] for item in group]
        aggregate = np.sum(counts, axis=0)
        expert_unions.extend(np.count_nonzero(aggregate, axis=1).tolist())
        per_rank_active = np.stack([
            np.count_nonzero(aggregate[:, rank * (NUM_EXPERTS // ep_size):(rank + 1) * (NUM_EXPERTS // ep_size)], axis=1)
            for rank in range(ep_size)
        ], axis=1)
        rank_unions.extend(per_rank_active.mean(axis=1).tolist())
        pairs.append(float(aggregate.sum()))
        active.append(float(np.count_nonzero(aggregate)))
        ranks = rank_histogram(aggregate, ep_size)
        maxmeans.extend((ranks.max(axis=1) / np.maximum(ranks.mean(axis=1), 1e-12)).tolist())
        remote.append(remote_fraction(np.sum(sources, axis=0), ep_size))
    return {
        "mean_active_expert_union_per_layer": float(np.mean(expert_unions)),
        "mean_active_experts_per_rank": float(np.mean(rank_unions)),
        "tokens_per_active_expert_proxy": float(np.sum(pairs) / max(np.sum(active), 1.0)),
        "rank_max_mean": float(np.mean(maxmeans)),
        "remote_destination_fraction": float(np.mean(remote)),
    }


def locality(
    samples: dict[str, dict[str, Any]], ids: list[str], predictions: dict[str, dict[str, np.ndarray]],
    wave_sizes: tuple[int, ...] = (4, 8, 16),
) -> dict[str, Any]:
    result = {
        "_meta": {
            "oracle": "future-truth greedy marginal-union minimization plus bounded pair-swap refinement; non-deployable and not a globally optimal partition proof",
            "fair_wave_sizes_enforced": True,
        }
    }
    full_vectors = {request: predictions["P_full"][request]["count"].reshape(-1) for request in ids}
    sep_vectors = {request: predictions["P_separated"][request]["count"].reshape(-1) for request in ids}
    vision_vectors = {request: predictions["P_vision"][request]["count"].reshape(-1) for request in ids}
    for vectors in (full_vectors, sep_vectors, vision_vectors):
        for request, value in vectors.items():
            vectors[request] = value / max(float(np.linalg.norm(value)), 1e-12)
    for size in wave_sizes:
        policies = {
            "S0_FIFO": [ids[offset:offset + size] for offset in range(0, len(ids), size)],
            "S1_DOMAIN": [],
            "S2_P_full": _greedy_groups(ids, full_vectors, size),
            "S2b_P_vision": _greedy_groups(ids, vision_vectors, size),
            "S3_P_separated": _greedy_groups(ids, sep_vectors, size),
            "S4_ORACLE": _oracle_groups(ids, samples, size),
        }
        domain_tails: list[str] = []
        for domain in DOMAINS:
            subset = [item for item in ids if samples[item]["domain"] == domain]
            full = len(subset) // size
            policies["S1_DOMAIN"].extend(subset[offset * size:(offset + 1) * size] for offset in range(full))
            domain_tails.extend(subset[full * size:])
        policies["S1_DOMAIN"].extend(
            domain_tails[offset:offset + size] for offset in range(0, len(domain_tails), size)
        )
        expected_sizes = sorted(len(group) for group in policies["S0_FIFO"])
        for policy, groups in policies.items():
            if sorted(len(group) for group in groups) != expected_sizes:
                raise AssertionError(f"unfair wave sizes for {policy}: {[len(group) for group in groups]}")
        result[str(size)] = {"assignments": policies, "ep": {}}
        for ep_size in (2, 4, 8):
            result[str(size)]["ep"][str(ep_size)] = {}
            baseline = None
            for policy, groups in policies.items():
                metric = _group_metric(groups, samples, ep_size)
                if policy == "S0_FIFO":
                    baseline = metric["mean_active_expert_union_per_layer"]
                metric["union_reduction_vs_fifo"] = 0.0 if baseline is None else (
                    baseline - metric["mean_active_expert_union_per_layer"]
                ) / baseline
                result[str(size)]["ep"][str(ep_size)][policy] = metric
    return result


def overhead(samples: list[dict[str, Any]], repeats: int = 100) -> dict[str, float]:
    begin = time.perf_counter_ns()
    for _ in range(repeats):
        for sample in samples:
            _ = np.concatenate((_normalize(sample["vision_mass"]).reshape(-1), _normalize(sample["text_mass"]).reshape(-1)))
    signature_us = (time.perf_counter_ns() - begin) / 1e3 / (repeats * len(samples))
    memory = samples[: min(20, len(samples))]
    query = samples[-1]
    begin = time.perf_counter_ns()
    for _ in range(repeats * 10):
        predict(query, memory, "P_separated", "1-256")
    query_us = (time.perf_counter_ns() - begin) / 1e3 / (repeats * 10)
    database = []
    begin = time.perf_counter_ns()
    for _ in range(repeats * 100):
        database.append(query["request_id"])
        database.pop()
    update_us = (time.perf_counter_ns() - begin) / 1e3 / (repeats * 100)
    return {
        "signature_construction_cpu_us_per_request": signature_us,
        "knn_query_cpu_us": query_us,
        "database_update_cpu_us": update_us,
        "gpu_synchronization_in_predictor": False,
        "deployable_gpu_signature_export_overhead": "UNAVAILABLE_NOT_ZERO",
        "scope": "CPU operations on already-collected compact routing histograms",
    }
