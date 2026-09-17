#!/usr/bin/env python3
"""H1 persistent-straggler deep dive and H2 EP8 re-analysis.

The script intentionally keeps three evidence levels separate:

* structural expert/rank assignments from the 128-request aggregate trace;
* EP2-calibrated rank-expert and communication models;
* optimistic/estimated replication oracles.

It never treats the 277,134 layer/refinement invocations as independent
generations and never labels virtual EP4/EP8 results as measurements.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Iterable
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from virtual_ep.comm_model import CommunicationScenario


NUM_EXPERTS = 256
HIDDEN = 2048
TOP_K = 8
ROUTED_LAYERS = 19
EXPERT_WEIGHT_BYTES = (512 * 2048 + 512 * 2048 + 2048 * 512) * 2
PERCENTILES = (50, 75, 90, 95, 99)
REPLICA_BUDGETS = (1, 2, 4, 8)
POSITION_CLASSES = (
    "PROMPT_PREFIX", "PRIOR_GENERATED_BLOCKS", "CURRENT_BLOCK_MASKED",
    "CURRENT_BLOCK_NEWLY_ACCEPTED", "CURRENT_BLOCK_DECODED",
)


def load_discovery_trace(path: Path):
    """Load schema-v2 arrays without importing the GPU tracing module."""
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files if name != "metadata_json"}
        metadata = json.loads(str(source["metadata_json"]))
    if int(metadata.get("schema_version", -1)) != 2:
        raise ValueError("expected aggregate discovery trace schema v2")
    if arrays["expert_counts_by_class"].shape[1:] != (5, NUM_EXPERTS):
        raise ValueError("invalid expert_counts_by_class shape")
    return SimpleNamespace(arrays=arrays, metadata=metadata)


def distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {f"p{p}": float("nan") for p in PERCENTILES} | {"max": float("nan")}
    return {f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES} | {
        "max": float(np.max(array))
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def spearman_small(left: np.ndarray, right: np.ndarray) -> float:
    if np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


class BatchComputeModel:
    """Vectorized nearest-measured-shape model matching virtual_ep.ComputeModel."""

    def __init__(self, path: Path):
        with Path(path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.features = np.asarray([
            [int(row["active_experts"]), int(row["assignments"]),
             int(row["max_rows_per_expert"]), float(row["cv_rows_per_expert"])]
            for row in rows
        ], dtype=np.float64)
        self.latency = np.asarray([float(row["latency_ms"]) for row in rows])
        self.scale = np.maximum(np.ptp(self.features, axis=0), 1.0)
        self.tree = cKDTree(self.features / self.scale)
        self.path = str(path)

    @staticmethod
    def features_for(histograms: np.ndarray) -> np.ndarray:
        histograms = np.asarray(histograms, dtype=np.float64)
        if histograms.ndim == 1:
            histograms = histograms[None, :]
        active = np.count_nonzero(histograms, axis=1)
        assignments = histograms.sum(axis=1)
        maximum = histograms.max(axis=1, initial=0)
        mean = np.divide(assignments, active, out=np.zeros_like(assignments), where=active > 0)
        second = (histograms * histograms).sum(axis=1)
        variance = np.divide(second, active, out=np.zeros_like(second), where=active > 0) - mean * mean
        cv = np.divide(
            np.sqrt(np.maximum(variance, 0)), mean,
            out=np.zeros_like(mean), where=mean > 0,
        )
        return np.column_stack((active, assignments, maximum, cv))

    def predict_batch(self, histograms: np.ndarray) -> np.ndarray:
        query = self.features_for(histograms)
        _distance, indices = self.tree.query(query / self.scale)
        return self.latency[np.asarray(indices, dtype=np.int64)]

    def predict_one(self, histogram: np.ndarray) -> float:
        return float(self.predict_batch(np.asarray(histogram)[None, :])[0])


def rank_times(histograms: np.ndarray, ep: int, model: BatchComputeModel) -> np.ndarray:
    per_rank = NUM_EXPERTS // ep
    return np.column_stack([
        model.predict_batch(histograms[:, rank * per_rank:(rank + 1) * per_rank])
        for rank in range(ep)
    ])


def imbalance_metrics(vectors: np.ndarray) -> dict[str, dict[str, float]]:
    vectors = np.asarray(vectors, dtype=np.float64)
    ordered = np.sort(vectors, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2] if vectors.shape[1] > 1 else maximum
    mean = vectors.mean(axis=1)
    cv = np.divide(vectors.std(axis=1), mean, out=np.zeros_like(mean), where=mean > 0)
    max_mean = np.divide(maximum, mean, out=np.ones_like(mean), where=mean > 0)
    max_second = np.divide(maximum, second, out=np.ones_like(maximum), where=second > 0)
    wait = np.divide(
        vectors.shape[1] * maximum - vectors.sum(axis=1),
        vectors.shape[1] * maximum,
        out=np.zeros_like(maximum), where=maximum > 0,
    )
    return {
        "max_mean": distribution(max_mean),
        "max_second": distribution(max_second),
        "cv": distribution(cv),
        "synchronization_wait_fraction": distribution(wait),
        "critical_rank_distribution": np.bincount(
            np.argmax(vectors, axis=1), minlength=vectors.shape[1]
        ).astype(int).tolist(),
    }


def build_groups(arrays: dict[str, np.ndarray], request_set: set[int], layer_local: bool):
    groups = defaultdict(list)
    for index, request_value in enumerate(arrays["request_id"]):
        request = int(request_value)
        if request not in request_set:
            continue
        key = (request, int(arrays["block_id"][index]))
        if layer_local:
            key += (int(arrays["layer_id"][index]),)
        groups[key].append((int(arrays["iteration_id"][index]), index))
    for values in groups.values():
        values.sort()
    return groups


def persistence_analysis(
    arrays: dict[str, np.ndarray], times: np.ndarray, ep: int,
    train: set[int], heldout: set[int], seed: int = 20260917,
) -> dict:
    layer_groups = build_groups(arrays, heldout, layer_local=True)
    pools = defaultdict(list)
    for key, values in layer_groups.items():
        for _iteration, index in values:
            pools[(key[1], key[2])].append((key[0], times[index]))
    rng = np.random.default_rng(seed + ep)
    adjacent_match, adjacent_cos, adjacent_spear = [], [], []
    random_match, random_cos = [], []
    runs = []
    for key, values in layer_groups.items():
        sequence = [int(np.argmax(times[index])) for _, index in values]
        if sequence:
            run = 1
            for left, right in zip(sequence, sequence[1:]):
                if left == right:
                    run += 1
                else:
                    runs.append(run)
                    run = 1
            runs.append(run)
        for (_li, left), (_ri, right) in zip(values, values[1:]):
            left_vector, right_vector = times[left], times[right]
            adjacent_match.append(float(np.argmax(left_vector) == np.argmax(right_vector)))
            adjacent_cos.append(cosine(left_vector, right_vector))
            adjacent_spear.append(spearman_small(left_vector, right_vector))
            candidates = [value for request, value in pools[(key[1], key[2])] if request != key[0]]
            if not candidates:
                candidates = [value for _, value in pools[(key[1], key[2])]]
            candidate = candidates[int(rng.integers(len(candidates)))]
            random_match.append(float(np.argmax(left_vector) == np.argmax(candidate)))
            random_cos.append(cosine(left_vector, candidate))

    # Same-request cross-block boundary, layer-local.
    by_request_layer = defaultdict(dict)
    for (request, block, layer), values in layer_groups.items():
        by_request_layer[(request, layer)][block] = values
    cross_block = []
    for blocks in by_request_layer.values():
        ordered = sorted(blocks)
        for previous, current in zip(ordered, ordered[1:]):
            cross_block.append(float(
                np.argmax(times[blocks[previous][-1][1]]) ==
                np.argmax(times[blocks[current][0][1]])
            ))

    train_mask = np.isin(arrays["request_id"], list(train))
    train_critical = np.argmax(times[train_mask], axis=1)
    majority_rank = int(np.argmax(np.bincount(train_critical, minlength=ep)))
    held_mask = np.isin(arrays["request_id"], list(heldout))
    overall_majority_accuracy = float(np.mean(np.argmax(times[held_mask], axis=1) == majority_rank))
    # A stable expert-popularity bias is layer-specific.  This is the fair
    # layer-local global baseline; a single rank across all layers is retained
    # only as a weaker diagnostic.
    majority_by_layer = {}
    for layer in range(1, ROUTED_LAYERS + 1):
        mask = train_mask & (arrays["layer_id"] == layer)
        ranks = np.argmax(times[mask], axis=1)
        majority_by_layer[layer] = int(np.argmax(np.bincount(ranks, minlength=ep)))
    held_indices = np.flatnonzero(held_mask)
    per_layer_majority_accuracy = float(np.mean([
        int(np.argmax(times[index])) == majority_by_layer[int(arrays["layer_id"][index])]
        for index in held_indices
    ]))

    # Forward aggregate: sum 19 layer rank-times for each refinement forward.
    forward = defaultdict(lambda: np.zeros(ep, dtype=np.float64))
    for index in np.flatnonzero(held_mask):
        key = (int(arrays["request_id"][index]), int(arrays["block_id"][index]),
               int(arrays["iteration_id"][index]))
        forward[key] += times[index]
    forward_groups = defaultdict(list)
    for (request, block, iteration), vector in forward.items():
        forward_groups[(request, block)].append((iteration, vector))
    for values in forward_groups.values():
        values.sort()
    forward_match, forward_cos, forward_spear, forward_runs = [], [], [], []
    for values in forward_groups.values():
        sequence = [int(np.argmax(vector)) for _, vector in values]
        if sequence:
            run = 1
            for left, right in zip(sequence, sequence[1:]):
                if left == right:
                    run += 1
                else:
                    forward_runs.append(run); run = 1
            forward_runs.append(run)
        for (_, left), (_, right) in zip(values, values[1:]):
            forward_match.append(float(np.argmax(left) == np.argmax(right)))
            forward_cos.append(cosine(left, right))
            forward_spear.append(spearman_small(left, right))
    train_forward = defaultdict(lambda: np.zeros(ep, dtype=np.float64))
    for index in np.flatnonzero(train_mask):
        key = (int(arrays["request_id"][index]), int(arrays["block_id"][index]),
               int(arrays["iteration_id"][index]))
        train_forward[key] += times[index]
    train_forward_critical = [int(np.argmax(vector)) for vector in train_forward.values()]
    forward_majority_rank = int(np.argmax(np.bincount(train_forward_critical, minlength=ep)))
    forward_majority_accuracy = float(np.mean([
        int(np.argmax(vector)) == forward_majority_rank for vector in forward.values()
    ]))
    return {
        "layer_local": {
            "adjacent_critical_rank_match": float(np.mean(adjacent_match)),
            "adjacent_rank_time_cosine": float(np.mean(adjacent_cos)),
            "adjacent_rank_time_spearman": float(np.mean(adjacent_spear)),
            "run_length": distribution(runs),
            "same_request_cross_block_match": float(np.mean(cross_block)),
            "matched_random_critical_rank_match": float(np.mean(random_match)),
            "matched_random_rank_time_cosine": float(np.mean(random_cos)),
            "global_majority_rank_overall": majority_rank,
            "global_majority_accuracy_overall": overall_majority_accuracy,
            "global_majority_rank_by_layer": {str(k): v for k, v in majority_by_layer.items()},
            "global_majority_accuracy": per_layer_majority_accuracy,
            "within_block_increment_over_global_pp": 100 * (float(np.mean(adjacent_match)) - per_layer_majority_accuracy),
            "within_block_increment_over_random_pp": 100 * (float(np.mean(adjacent_match)) - float(np.mean(random_match))),
            "pairs": len(adjacent_match),
        },
        "forward_aggregate": {
            "adjacent_critical_rank_match": float(np.mean(forward_match)),
            "adjacent_rank_time_cosine": float(np.mean(forward_cos)),
            "adjacent_rank_time_spearman": float(np.mean(forward_spear)),
            "run_length": distribution(forward_runs),
            "global_majority_rank": forward_majority_rank,
            "global_majority_accuracy": forward_majority_accuracy,
            "within_block_increment_over_global_pp": 100 * (float(np.mean(forward_match)) - forward_majority_accuracy),
            "refinement_forwards": len(forward),
        },
    }


def excess_scores(histogram: np.ndarray, time_vector: np.ndarray, ep: int) -> np.ndarray:
    """Allocate modeled critical-rank excess by exact expert-row share.

    The measured compute model predicts only rank-level grouped-MLP time.  It
    does not expose per-expert kernel time.  We therefore allocate the excess
    (not the entire rank time) proportionally to useful rows on the critical
    rank.  This is explicit, conservative, and sums exactly to the modeled
    excess; it is not presented as a measured per-expert timestamp.
    """
    result = np.zeros(NUM_EXPERTS, dtype=np.float64)
    critical = int(np.argmax(time_vector))
    excess = float(time_vector[critical] - np.mean(time_vector))
    if excess <= 0:
        return result
    per_rank = NUM_EXPERTS // ep
    start, end = critical * per_rank, (critical + 1) * per_rank
    counts = np.asarray(histogram[start:end], dtype=np.float64)
    if counts.sum() > 0:
        result[start:end] = excess * counts / counts.sum()
    return result


def concentration_analysis(
    histograms: np.ndarray, times: np.ndarray, request_ids: np.ndarray,
    heldout: set[int], ep: int,
) -> dict:
    shares = {count: [] for count in (1, 2, 4, 8)}
    needed = {fraction: [] for fraction in (.25, .50, .75, .90)}
    high_shares = []
    for index in np.flatnonzero(np.isin(request_ids, list(heldout))):
        scores = np.sort(excess_scores(histograms[index], times[index], ep))[::-1]
        total = scores.sum()
        if total <= 0:
            continue
        cumulative = np.cumsum(scores)
        for count in shares:
            shares[count].append(float(cumulative[min(count, len(scores)) - 1] / total))
        for fraction in needed:
            needed[fraction].append(int(np.searchsorted(cumulative, fraction * total) + 1))
        time_vector = times[index]
        if time_vector.max() / time_vector.mean() >= 1.15:
            high_shares.append(float(cumulative[3] / total))
    return {
        "attribution": "modeled critical-rank excess allocated by exact expert-row share",
        "top_k_excess_mass_fraction": {f"top_{k}": distribution(v) for k, v in shares.items()},
        "minimum_experts_for_excess_fraction": {
            str(int(100 * fraction)): distribution(values) for fraction, values in needed.items()
        },
        "high_imbalance_top4_ge_50_fraction": float(np.mean(np.asarray(high_shares) >= .5)) if high_shares else 0.0,
        "high_imbalance_states": len(high_shares),
    }


def score_indices(
    indices: list[int], histograms: np.ndarray, times: np.ndarray, ep: int,
) -> np.ndarray:
    score = np.zeros(NUM_EXPERTS, dtype=np.float64)
    for index in indices:
        score += excess_scores(histograms[index], times[index], ep)
    return score


def hot_expert_predictability(
    arrays: dict[str, np.ndarray], histograms: np.ndarray, times: np.ndarray,
    ep: int, train: set[int], heldout: set[int],
) -> dict:
    train_global = defaultdict(lambda: np.zeros(NUM_EXPERTS, dtype=np.float64))
    for index in np.flatnonzero(np.isin(arrays["request_id"], list(train))):
        train_global[int(arrays["layer_id"][index])] += excess_scores(
            histograms[index], times[index], ep
        )
    groups = build_groups(arrays, heldout, layer_local=True)
    methods = {"early_1": defaultdict(list), "early_1_2": defaultdict(list),
               "global_hot": defaultdict(list)}
    critical = {"early_1": [], "early_1_2": [], "global_rank": []}
    global_rank = int(np.argmax(np.bincount(
        np.argmax(times[np.isin(arrays["request_id"], list(train))], axis=1), minlength=ep
    )))
    for key, values in groups.items():
        if len(values) < 2:
            continue
        indices = [index for _, index in values]
        future_one = indices[1:]
        future_two = indices[2:]
        for name, observed, future in (
            ("early_1", indices[:1], future_one),
            ("early_1_2", indices[:2], future_two),
        ):
            if not future:
                continue
            prediction = score_indices(observed, histograms, times, ep)
            target = score_indices(future, histograms, times, ep)
            target_total = target.sum()
            if target_total <= 0:
                continue
            for k in (1, 2, 4, 8):
                predicted = set(np.argsort(prediction)[-k:])
                actual = set(np.argsort(target)[-k:])
                overlap = len(predicted & actual)
                methods[name][k].append((overlap / k, overlap / k, target[list(predicted)].sum() / target_total))
            observed_rank = int(np.argmax(np.sum(times[observed], axis=0)))
            future_ranks = np.argmax(times[future], axis=1)
            critical[name].extend((future_ranks == observed_rank).astype(float).tolist())
        target = score_indices(future_one, histograms, times, ep)
        if target.sum() > 0:
            prediction = train_global[key[2]]
            for k in (1, 2, 4, 8):
                predicted = set(np.argsort(prediction)[-k:])
                actual = set(np.argsort(target)[-k:])
                overlap = len(predicted & actual)
                methods["global_hot"][k].append((overlap / k, overlap / k,
                                                  target[list(predicted)].sum() / target.sum()))
        critical["global_rank"].extend(
            (np.argmax(times[future_one], axis=1) == global_rank).astype(float).tolist()
        )
    output = {}
    for method, by_k in methods.items():
        output[method] = {}
        for k, values in by_k.items():
            matrix = np.asarray(values)
            output[method][f"k{k}"] = {
                "recall_mean": float(matrix[:, 0].mean()),
                "precision_mean": float(matrix[:, 1].mean()),
                "excess_mass_recall_mean": float(matrix[:, 2].mean()),
                "excess_mass_recall": distribution(matrix[:, 2]),
                "units": len(matrix),
            }
    output["critical_rank_accuracy"] = {
        name: float(np.mean(values)) for name, values in critical.items() if values
    }
    output["dllm_increment_early2_minus_global_excess_mass_pp"] = {
        f"k{k}": 100 * (
            output["early_1_2"][f"k{k}"]["excess_mass_recall_mean"] -
            output["global_hot"][f"k{k}"]["excess_mass_recall_mean"]
        ) for k in (1, 2, 4, 8)
    }
    return output


def communication_arrays(
    arrays: dict[str, np.ndarray], ep: int, communication: CommunicationScenario,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outgoing = arrays[f"outgoing_bytes_ep{ep}"][mask]
    incoming = arrays[f"incoming_bytes_ep{ep}"][mask]
    dispatch = np.asarray([communication.dispatch.latency(o, i) for o, i in zip(outgoing, incoming)])
    combine = np.asarray([communication.combine.latency(i, o) for o, i in zip(outgoing, incoming)])
    return dispatch, combine


def perfect_balance(
    arrays: dict[str, np.ndarray], times: np.ndarray, ep: int,
    heldout: set[int], communication: CommunicationScenario,
) -> dict:
    mask = np.isin(arrays["request_id"], list(heldout))
    selected = times[mask]
    maximum, mean = selected.max(axis=1), selected.mean(axis=1)
    dispatch, combine = communication_arrays(arrays, ep, communication, mask)
    baseline_stage = dispatch + maximum + combine
    ideal_stage = dispatch + mean + combine
    expert_reduction = np.divide(maximum - mean, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    stage_reduction = np.divide(
        baseline_stage - ideal_stage, baseline_stage,
        out=np.zeros_like(baseline_stage), where=baseline_stage > 0,
    )
    return {
        "expert_stage_reduction": distribution(expert_reduction),
        "routed_moe_stage_reduction": distribution(stage_reduction),
        "aggregate_expert_stage_reduction": float((maximum.sum() - mean.sum()) / maximum.sum()),
        "aggregate_routed_moe_stage_reduction": float(
            (baseline_stage.sum() - ideal_stage.sum()) / baseline_stage.sum()
        ),
        "baseline_component_ms": {
            "dispatch": float(dispatch.sum()), "expert": float(maximum.sum()),
            "combine": float(combine.sum()), "stage": float(baseline_stage.sum()),
        },
        "oracle_component_ms": {
            "dispatch": float(dispatch.sum()), "expert": float(mean.sum()),
            "combine": float(combine.sum()), "stage": float(ideal_stage.sum()),
        },
    }


def ordered_replicas(
    indices: list[int], histograms: np.ndarray, times: np.ndarray,
    ep: int, limit: int = 8,
) -> list[tuple[int, int]]:
    score = score_indices(indices, histograms, times, ep)
    loads = histograms[indices].sum(axis=0).reshape(ep, NUM_EXPERTS // ep).sum(axis=1).astype(float)
    result = []
    for expert in np.argsort(score)[::-1]:
        if score[expert] <= 0 or len(result) >= limit:
            break
        owner = int(expert) // (NUM_EXPERTS // ep)
        candidate = np.argmin(np.where(np.arange(ep) == owner, np.inf, loads))
        result.append((int(expert), int(candidate)))
        # Planning proxy: a replica can absorb at most half of the observed
        # original/destination gap while retaining the same expert semantics.
        count = float(histograms[indices, int(expert)].sum())
        moved = min(count, max(0.0, (loads[owner] - loads[candidate]) / 2))
        loads[owner] -= moved; loads[candidate] += moved
    return result


def build_replica_policies(
    arrays: dict[str, np.ndarray], histograms: np.ndarray,
    times: np.ndarray, ep: int, train: set[int], heldout: set[int],
) -> dict[str, dict]:
    train_by_layer = defaultdict(list)
    for index in np.flatnonzero(np.isin(arrays["request_id"], list(train))):
        train_by_layer[int(arrays["layer_id"][index])].append(index)
    global_map = {layer: ordered_replicas(indices, histograms, times, ep)
                  for layer, indices in train_by_layer.items()}
    held_groups = build_groups(arrays, heldout, layer_local=True)
    full, early1, early2, combined = {}, {}, {}, {}
    for key, values in held_groups.items():
        indices = [index for _, index in values]
        full[key] = ordered_replicas(indices, histograms, times, ep)
        early1[key] = ordered_replicas(indices[:1], histograms, times, ep)
        early2[key] = ordered_replicas(indices[:2], histograms, times, ep)
        merged = []
        for pair in zip(global_map.get(key[2], []), early2[key]):
            for item in pair:
                if item[0] not in {expert for expert, _ in merged}:
                    merged.append(item)
        for item in global_map.get(key[2], []) + early2[key]:
            if item[0] not in {expert for expert, _ in merged}:
                merged.append(item)
        combined[key] = merged[:8]
    return {
        "global_static": {"scope": "layer", "map": global_map, "observe": 0},
        "full_future_block": {"scope": "block", "map": full, "observe": 0},
        "early_1": {"scope": "block", "map": early1, "observe": 1},
        "early_1_2": {"scope": "block", "map": early2, "observe": 2},
        "global_plus_early_1_2": {"scope": "block", "map": combined, "observe": 2},
    }


def balance_histogram(
    histogram: np.ndarray, replicas: list[tuple[int, int]], ep: int,
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    per_rank = NUM_EXPERTS // ep
    local = [histogram[rank * per_rank:(rank + 1) * per_rank].astype(float).copy()
             for rank in range(ep)]
    loads = np.asarray([row.sum() for row in local], dtype=float)
    moved = []
    extra = [[] for _ in range(ep)]
    for expert, destination in replicas:
        owner = expert // per_rank
        count = int(round(local[owner][expert - owner * per_rank]))
        if count <= 0 or owner == destination:
            continue
        amount = int(np.clip(round((loads[owner] - loads[destination]) / 2), 0, count))
        if amount <= 0:
            continue
        local[owner][expert - owner * per_rank] -= amount
        extra[destination].append(float(amount))
        loads[owner] -= amount; loads[destination] += amount
        moved.append((expert, destination, amount))
    return [np.concatenate((local[rank], np.asarray(extra[rank], dtype=float)))
            for rank in range(ep)], moved


def estimate_replica_traffic(
    assignment: np.ndarray, unique: np.ndarray, physical_rows: int,
    moves: list[tuple[int, int, int]], ep: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recompute an aggregate A/U estimate after replica assignment moves.

    Aggregate-v2 has exact full A/U but not source×expert identity.  Source
    shares are therefore inferred from the original owner's A column.  U/A
    supplies the measured dedup rate, and existing destination presence
    discounts additions.  This is used only for a calibrated sensitivity; it
    is not called exact protocol traffic.
    """
    updated_a = np.asarray(assignment, dtype=float).copy()
    updated_u = np.asarray(unique, dtype=float).copy()
    source_rows = np.full(ep, physical_rows // ep, dtype=float)
    source_rows[:physical_rows % ep] += 1
    per_rank = NUM_EXPERTS // ep
    moved_total = 0.0
    for expert, destination, amount in moves:
        owner = expert // per_rank
        column = np.maximum(updated_a[:, owner], 0)
        if column.sum() <= 0:
            source_share = source_rows / source_rows.sum()
        else:
            source_share = column / column.sum()
        moved_by_source = amount * source_share
        old_unique_rate = np.divide(
            updated_u[:, owner], np.maximum(updated_a[:, owner], 1),
        )
        removal = np.minimum(updated_u[:, owner], moved_by_source * old_unique_rate)
        presence = np.divide(
            updated_u[:, destination], source_rows,
            out=np.zeros(ep), where=source_rows > 0,
        )
        addition = moved_by_source * old_unique_rate * np.clip(1 - presence, 0, 1)
        updated_a[:, owner] -= moved_by_source
        updated_a[:, destination] += moved_by_source
        updated_u[:, owner] -= removal
        updated_u[:, destination] = np.minimum(source_rows, updated_u[:, destination] + addition)
        moved_total += amount
    remote = updated_u.copy(); np.fill_diagonal(remote, 0)
    outgoing = remote.sum(axis=1) * HIDDEN * 2
    incoming = remote.sum(axis=0) * HIDDEN * 2
    return outgoing, incoming, moved_total


@njit(parallel=True, cache=True)
def batch_balance_and_traffic(
    histograms, assignments, uniques, physical_rows, candidate_experts,
    candidate_destinations, ep: int, budget: int,
):
    """Numba implementation of the exact-row split + aggregate A/U estimate."""
    count = histograms.shape[0]
    per_rank = NUM_EXPERTS // ep
    output_hist = np.zeros((count, ep, per_rank + budget), dtype=np.float32)
    outgoing = np.zeros((count, ep), dtype=np.float64)
    incoming = np.zeros((count, ep), dtype=np.float64)
    moved_total = np.zeros(count, dtype=np.float64)
    for i in prange(count):
        loads = np.zeros(ep, dtype=np.float64)
        extras = np.zeros(ep, dtype=np.int64)
        for rank in range(ep):
            for local in range(per_rank):
                value = histograms[i, rank * per_rank + local]
                output_hist[i, rank, local] = value
                loads[rank] += value
        updated_a = assignments[i].astype(np.float64)
        updated_u = uniques[i].astype(np.float64)
        row_counts = np.empty(ep, dtype=np.float64)
        base_rows = physical_rows[i] // ep
        remainder = physical_rows[i] % ep
        for rank in range(ep):
            row_counts[rank] = base_rows + (1 if rank < remainder else 0)
        for slot in range(budget):
            expert = int(candidate_experts[i, slot])
            destination = int(candidate_destinations[i, slot])
            if expert < 0 or destination < 0:
                continue
            owner = expert // per_rank
            local = expert - owner * per_rank
            expert_count = int(round(output_hist[i, owner, local]))
            amount = int(round((loads[owner] - loads[destination]) / 2.0))
            if amount < 0:
                amount = 0
            if amount > expert_count:
                amount = expert_count
            if amount <= 0:
                continue
            output_hist[i, owner, local] -= amount
            target = per_rank + extras[destination]
            output_hist[i, destination, target] = amount
            extras[destination] += 1
            loads[owner] -= amount
            loads[destination] += amount
            moved_total[i] += amount
            column_sum = 0.0
            for source in range(ep):
                if updated_a[source, owner] > 0:
                    column_sum += updated_a[source, owner]
            for source in range(ep):
                source_share = (updated_a[source, owner] / column_sum
                                if column_sum > 0 else row_counts[source] / physical_rows[i])
                moved_source = amount * source_share
                old_rate = updated_u[source, owner] / max(updated_a[source, owner], 1.0)
                removal = min(updated_u[source, owner], moved_source * old_rate)
                presence = updated_u[source, destination] / max(row_counts[source], 1.0)
                addition = moved_source * old_rate * max(0.0, 1.0 - presence)
                updated_a[source, owner] -= moved_source
                updated_a[source, destination] += moved_source
                updated_u[source, owner] -= removal
                updated_u[source, destination] = min(
                    row_counts[source], updated_u[source, destination] + addition
                )
        for source in range(ep):
            for destination in range(ep):
                if source != destination:
                    value = updated_u[source, destination] * HIDDEN * 2
                    outgoing[i, source] += value
                    incoming[i, destination] += value
    return output_hist, outgoing, incoming, moved_total


def policy_key(arrays: dict[str, np.ndarray], index: int, scope: str):
    if scope == "layer":
        return int(arrays["layer_id"][index])
    return (int(arrays["request_id"][index]), int(arrays["block_id"][index]),
            int(arrays["layer_id"][index]))


def evaluate_replica_policy(
    arrays: dict[str, np.ndarray], histograms: np.ndarray,
    base_times: np.ndarray, model: BatchComputeModel, communication: CommunicationScenario,
    ep: int, heldout: set[int], policy: dict,
) -> dict:
    selected_indices = np.flatnonzero(np.isin(arrays["request_id"], list(heldout)))
    base_vectors = base_times[selected_indices]
    base_ordered = np.sort(base_vectors, axis=1)
    base_max = base_ordered[:, -1]
    base_second = base_ordered[:, -2]
    base_mean = base_vectors.mean(axis=1)
    base_dispatch = np.asarray([
        communication.dispatch.latency(arrays[f"outgoing_bytes_ep{ep}"][index],
                                       arrays[f"incoming_bytes_ep{ep}"][index])
        for index in selected_indices
    ])
    base_combine = np.asarray([
        communication.combine.latency(arrays[f"incoming_bytes_ep{ep}"][index],
                                      arrays[f"outgoing_bytes_ep{ep}"][index])
        for index in selected_indices
    ])
    base_stage = base_dispatch + base_max + base_combine
    records = {}
    block_savings = {budget: defaultdict(float) for budget in REPLICA_BUDGETS}
    block_baseline = defaultdict(float)
    for ordinal, index in enumerate(selected_indices):
        block = (int(arrays["request_id"][index]), int(arrays["block_id"][index]))
        block_baseline[block] += base_stage[ordinal]
    block_instances = {budget: defaultdict(set) for budget in REPLICA_BUDGETS}
    per_rank = NUM_EXPERTS // ep
    candidate_experts = np.full((len(selected_indices), 8), -1, dtype=np.int16)
    candidate_destinations = np.full((len(selected_indices), 8), -1, dtype=np.int8)
    for ordinal, index in enumerate(selected_indices):
        iteration = int(arrays["iteration_id"][index])
        key = policy_key(arrays, index, policy["scope"])
        candidates = policy["map"].get(key, [])
        if iteration >= policy["observe"]:
            for slot, (expert, destination) in enumerate(candidates[:8]):
                candidate_experts[ordinal, slot] = expert
                candidate_destinations[ordinal, slot] = destination
            if candidates:
                block = (int(arrays["request_id"][index]), int(arrays["block_id"][index]))
                for budget in REPLICA_BUDGETS:
                    block_instances[budget][block].update(
                        (int(arrays["layer_id"][index]), e, d) for e, d in candidates[:budget]
                    )
    for budget in REPLICA_BUDGETS:
        candidate_hist, candidate_outgoing, candidate_incoming, moved_assignments = (
            batch_balance_and_traffic(
                histograms[selected_indices], arrays[f"assignment_matrix_ep{ep}"][selected_indices],
                arrays[f"unique_matrix_ep{ep}"][selected_indices],
                arrays["physical_rows"][selected_indices], candidate_experts,
                candidate_destinations, ep, budget,
            )
        )
        candidate_vectors = np.column_stack([
            model.predict_batch(candidate_hist[:, rank, :]) for rank in range(ep)
        ])
        candidate_ordered = np.sort(candidate_vectors, axis=1)
        candidate_max = candidate_ordered[:, -1]
        candidate_second = candidate_ordered[:, -2]
        candidate_mean = candidate_vectors.mean(axis=1)
        candidate_dispatch = np.asarray([
            communication.dispatch.latency(outgoing, incoming)
            for outgoing, incoming in zip(candidate_outgoing, candidate_incoming)
        ])
        candidate_combine = np.asarray([
            communication.combine.latency(incoming, outgoing)
            for outgoing, incoming in zip(candidate_outgoing, candidate_incoming)
        ])
        candidate_stage = candidate_dispatch + candidate_max + candidate_combine
        candidate_compute_only_stage = base_dispatch + candidate_max + base_combine
        for ordinal, index in enumerate(selected_indices):
            block = (int(arrays["request_id"][index]), int(arrays["block_id"][index]))
            block_savings[budget][block] += base_stage[ordinal] - candidate_stage[ordinal]
        def vector_stats(vectors):
            ordered = np.sort(vectors, axis=1); maximum = ordered[:, -1]
            second = ordered[:, -2]; mean = vectors.mean(axis=1)
            return {
                "max": maximum, "second": second, "mean": mean,
                "max_mean": maximum / mean, "max_second": maximum / second,
                "cv": vectors.std(axis=1) / mean,
                "wait": (ep * maximum - vectors.sum(axis=1)) / (ep * maximum),
            }
        records[budget] = {
            "baseline": vector_stats(base_vectors), "candidate": vector_stats(candidate_vectors),
            "baseline_dispatch": base_dispatch, "candidate_dispatch": candidate_dispatch,
            "baseline_combine": base_combine, "candidate_combine": candidate_combine,
            "baseline_stage": base_stage, "candidate_stage": candidate_stage,
            "candidate_compute_only_stage": candidate_compute_only_stage,
            "moved_assignments": moved_assignments,
        }
        print(f"replica evaluation EP{ep} {policy['scope']} budget={budget} complete", flush=True)
    output = {}
    for budget, row in records.items():
        baseline_stage = row["baseline_stage"]
        candidate_stage = row["candidate_stage"]
        compute_stage = row["candidate_compute_only_stage"]
        summary = {
            "aggregate_stage_reduction": float((baseline_stage.sum() - candidate_stage.sum()) / baseline_stage.sum()),
            "aggregate_compute_only_stage_reduction": float((baseline_stage.sum() - compute_stage.sum()) / baseline_stage.sum()),
            "per_invocation_stage_reduction": distribution(
                np.divide(baseline_stage - candidate_stage, baseline_stage,
                          out=np.zeros_like(baseline_stage), where=baseline_stage > 0)
            ),
            "baseline": {}, "candidate": {},
            "component_ms": {
                "baseline_dispatch": float(np.sum(row["baseline_dispatch"])),
                "candidate_dispatch": float(np.sum(row["candidate_dispatch"])),
                "baseline_expert": float(np.sum(row["baseline"]["max"])),
                "candidate_expert": float(np.sum(row["candidate"]["max"])),
                "baseline_combine": float(np.sum(row["baseline_combine"])),
                "candidate_combine": float(np.sum(row["candidate_combine"])),
                "baseline_stage": float(baseline_stage.sum()),
                "candidate_stage": float(candidate_stage.sum()),
            },
            "moved_assignment_fraction": float(
                np.sum(row["moved_assignments"]) / histograms[selected_indices].sum()
            ),
            "traffic_model": "aggregate A/U source-share and measured dedup-rate estimate",
            "block_stage_reduction": distribution([
                block_savings[budget][block] / baseline
                for block, baseline in block_baseline.items() if baseline > 0
            ]),
            "block_stage_totals": {
                f"{block[0]}:{block[1]}": {
                    "baseline_ms": baseline,
                    "candidate_ms": baseline - block_savings[budget][block],
                }
                for block, baseline in block_baseline.items()
            },
        }
        for prefix in ("baseline", "candidate"):
            summary[prefix] = {
                "max_rank_time_ms": distribution(row[prefix]["max"]),
                "second_max_rank_time_ms": distribution(row[prefix]["second"]),
                "mean_rank_time_ms": distribution(row[prefix]["mean"]),
                "max_mean": distribution(row[prefix]["max_mean"]),
                "max_second": distribution(row[prefix]["max_second"]),
                "cv": distribution(row[prefix]["cv"]),
                "wait_fraction": distribution(row[prefix]["wait"]),
            }
        summary["memory"] = {
            "extra_instances_per_layer": budget,
            "expert_bytes": EXPERT_WEIGHT_BYTES,
            "total_pool_bytes_all_19_layers": budget * EXPERT_WEIGHT_BYTES * ROUTED_LAYERS,
            "mean_pool_bytes_per_rank": budget * EXPERT_WEIGHT_BYTES * ROUTED_LAYERS / ep,
            "worst_case_pool_bytes_one_rank": budget * EXPERT_WEIGHT_BYTES * ROUTED_LAYERS,
        }
        if policy["observe"]:
            break_even_payload, break_even_charged = [], []
            feasible_payload, feasible_charged = [], []
            for block, saving in block_savings[budget].items():
                instances = block_instances[budget][block]
                if not instances:
                    continue
                outgoing = np.zeros(ep); incoming = np.zeros(ep)
                layers = set()
                for layer, expert, destination in instances:
                    owner = expert // (NUM_EXPERTS // ep)
                    outgoing[owner] += EXPERT_WEIGHT_BYTES
                    incoming[destination] += EXPERT_WEIGHT_BYTES
                    layers.add(layer)
                payload_ms = max(outgoing.max(), incoming.max()) / (
                    communication.dispatch.gb_per_s * 1e9
                ) * 1e3
                charged_ms = payload_ms + len(layers) * communication.dispatch.startup_ms
                # One refinement's saving is estimated from the realized
                # whole-block future sum divided by post-observation steps.
                request, block_id = block
                mask = ((arrays["request_id"] == request) & (arrays["block_id"] == block_id))
                future_refinements = max(1, len(np.unique(arrays["iteration_id"][mask])) - policy["observe"])
                per_refinement = saving / future_refinements
                break_even_payload.append(payload_ms / per_refinement if per_refinement > 0 else math.inf)
                break_even_charged.append(charged_ms / per_refinement if per_refinement > 0 else math.inf)
                feasible_payload.append(saving > payload_ms)
                feasible_charged.append(saving > charged_ms)
            finite_payload = [x for x in break_even_payload if np.isfinite(x)]
            finite_charged = [x for x in break_even_charged if np.isfinite(x)]
            summary["block_triggered_replication"] = {
                "payload_only_break_even_refinements": distribution(finite_payload),
                "setup_charged_break_even_refinements": distribution(finite_charged),
                "payload_only_feasible_block_fraction": float(np.mean(feasible_payload)) if feasible_payload else 0,
                "setup_charged_feasible_block_fraction": float(np.mean(feasible_charged)) if feasible_charged else 0,
                "setup_charge": "measured EP2 dispatch startup once per affected layer plus endpoint payload time",
            }
        output[f"budget_{budget}"] = summary
    return output


def h2_ep8_reanalysis(
    trace, models: dict[int, BatchComputeModel],
    communication: CommunicationScenario, heldout: set[int],
    rank_time_cache: dict[int, np.ndarray],
) -> dict:
    arrays = trace.arrays
    class_index = {name: i for i, name in enumerate(POSITION_CLASSES)}
    mask_held = np.isin(arrays["request_id"], list(heldout))
    same_position = {4: [], 8: []}
    transitions = {4: [], 8: []}
    groups = build_groups(arrays, heldout, layer_local=True)
    for values in groups.values():
        for (_, left), (_, right) in zip(values, values[1:]):
            changed = ((arrays["current_position_class"][left] == 2) &
                       np.isin(arrays["current_position_class"][right], [3, 4]))
            for position in np.flatnonzero(changed):
                left_set = set(map(int, arrays["current_expert_ids"][left, position]))
                right_set = set(map(int, arrays["current_expert_ids"][right, position]))
                for ep in (4, 8):
                    bucket = NUM_EXPERTS // ep
                    left_rank = {expert // bucket for expert in left_set}
                    right_rank = {expert // bucket for expert in right_set}
                    same_position[ep].append(len(left_rank & right_rank) / len(left_rank | right_rank))
                    transitions[ep].append(float(left_rank != right_rank))

    state_only = {}
    for ep in (4, 8):
        state_only[f"ep{ep}"] = {}
        for state in ("CURRENT_BLOCK_MASKED", "CURRENT_BLOCK_NEWLY_ACCEPTED", "CURRENT_BLOCK_DECODED"):
            state_id = class_index[state]
            max_mean, cvs, hottest, remote_bytes, fanout = [], [], [], [], []
            state_histograms = []
            for index in np.flatnonzero(mask_held):
                routes = arrays["current_expert_ids"][index][
                    arrays["current_position_class"][index] == state_id
                ]
                if not len(routes):
                    continue
                owners = routes // (NUM_EXPERTS // ep)
                loads = np.bincount(owners.reshape(-1), minlength=ep).astype(float)
                max_mean.append(loads.max() / loads.mean())
                cvs.append(loads.std() / loads.mean())
                hottest.append(int(np.argmax(loads)))
                source = arrays[f"current_source_rank_ep{ep}"][index][
                    arrays["current_position_class"][index] == state_id
                ]
                unique = np.zeros((ep, ep), dtype=int)
                for destination in range(ep):
                    hit = np.any(owners == destination, axis=1)
                    unique[:, destination] = np.bincount(source[hit], minlength=ep)
                remote = unique.copy(); np.fill_diagonal(remote, 0)
                remote_bytes.append(float(remote.sum() * HIDDEN * 2))
                fanout.extend([len(set(map(int, row))) for row in owners])
                histogram = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
                state_histograms.append(histogram)
            state_histograms = np.asarray(state_histograms)
            if len(state_histograms):
                predicted = rank_times(state_histograms, ep, models[ep])
                expert_time = predicted.max(axis=1)
            else:
                expert_time = []
            state_only[f"ep{ep}"][state] = {
                "max_mean": distribution(max_mean), "cv": distribution(cvs),
                "hottest_rank_distribution": np.bincount(hottest, minlength=ep).astype(int).tolist(),
                "remote_bytes": distribution(remote_bytes), "fanout": distribution(fanout),
                "predicted_critical_expert_time_ms": distribution(expert_time),
            }

    # Full-workload counterfactual: replace current masked route sets with a
    # deterministic cycle of observed decoded route sets in the same invocation.
    full_effect = {4: defaultdict(list), 8: defaultdict(list)}
    effect_indices = []
    candidate_histograms = []
    candidate_endpoint = {4: [], 8: []}
    for index in np.flatnonzero(mask_held):
        classes = arrays["current_position_class"][index]
        masked_rows = arrays["current_expert_ids"][index][classes == class_index["CURRENT_BLOCK_MASKED"]]
        decoded_rows = arrays["current_expert_ids"][index][classes == class_index["CURRENT_BLOCK_DECODED"]]
        if not len(masked_rows) or not len(decoded_rows):
            continue
        replacement = decoded_rows[np.arange(len(masked_rows)) % len(decoded_rows)]
        base_hist = arrays["expert_counts_by_class"][index].sum(axis=0).astype(float)
        candidate_hist = base_hist.copy()
        candidate_hist -= np.bincount(masked_rows.reshape(-1), minlength=NUM_EXPERTS)
        candidate_hist += np.bincount(replacement.reshape(-1), minlength=NUM_EXPERTS)
        effect_indices.append(index)
        candidate_histograms.append(candidate_hist)
        for ep in (4, 8):
            base_load = arrays[f"rank_load_ep{ep}"][index].astype(float)
            candidate_load = candidate_hist.reshape(ep, NUM_EXPERTS // ep).sum(axis=1)
            full_effect[ep]["max_mean"].append(abs(
                candidate_load.max() / candidate_load.mean() - base_load.max() / base_load.mean()
            ) / (base_load.max() / base_load.mean()))
            # Current block rows are exact and each is an independent token,
            # so their U contribution can be subtracted/added from full U.
            source_mask = arrays[f"current_source_rank_ep{ep}"][index][classes == class_index["CURRENT_BLOCK_MASKED"]]
            base_current_owner = masked_rows // (NUM_EXPERTS // ep)
            repl_owner = replacement // (NUM_EXPERTS // ep)
            unique = arrays[f"unique_matrix_ep{ep}"][index].astype(float).copy()
            for destination in range(ep):
                old_hit = np.any(base_current_owner == destination, axis=1)
                new_hit = np.any(repl_owner == destination, axis=1)
                unique[:, destination] -= np.bincount(source_mask[old_hit], minlength=ep)
                unique[:, destination] += np.bincount(source_mask[new_hit], minlength=ep)
            unique = np.maximum(unique, 0)
            remote = unique.copy(); np.fill_diagonal(remote, 0)
            outgoing = remote.sum(axis=1) * HIDDEN * 2
            incoming = remote.sum(axis=0) * HIDDEN * 2
            base_out = arrays[f"outgoing_bytes_ep{ep}"][index]
            base_in = arrays[f"incoming_bytes_ep{ep}"][index]
            base_remote = float(base_out.sum())
            candidate_remote = float(outgoing.sum())
            full_effect[ep]["remote_bytes"].append(
                abs(candidate_remote - base_remote) / base_remote if base_remote else 0
            )
            candidate_endpoint[ep].append((base_out.copy(), base_in.copy(), outgoing, incoming))
    if candidate_histograms:
        candidate_histograms = np.asarray(candidate_histograms)
        effect_indices_array = np.asarray(effect_indices, dtype=int)
        for ep in (4, 8):
            candidate_times = rank_times(candidate_histograms, ep, models[ep])
            base_times = rank_time_cache[ep][effect_indices_array]
            critical_change = np.abs(candidate_times.max(axis=1) - base_times.max(axis=1)) / base_times.max(axis=1)
            full_effect[ep]["critical_expert_time"].extend(critical_change.tolist())
            for ordinal, (base_out, base_in, outgoing, incoming) in enumerate(candidate_endpoint[ep]):
                base_stage = (communication.dispatch.latency(base_out, base_in) +
                              base_times[ordinal].max() + communication.combine.latency(base_in, base_out))
                candidate_stage = (communication.dispatch.latency(outgoing, incoming) +
                                   candidate_times[ordinal].max() + communication.combine.latency(incoming, outgoing))
                full_effect[ep]["routed_moe_stage"].append(abs(candidate_stage - base_stage) / base_stage)
    return {
        "same_position": {
            "ep4_destination_rank_set_jaccard": distribution(same_position[4]),
            "ep8_destination_rank_set_jaccard": distribution(same_position[8]),
            "ep4_rank_transition_rate": float(np.mean(transitions[4])),
            "ep8_rank_transition_rate": float(np.mean(transitions[8])),
            "samples": len(same_position[8]),
        },
        "state_only": state_only,
        "full_workload_composition_effect": {
            f"ep{ep}": {metric: distribution(values) for metric, values in full_effect[ep].items()}
            for ep in (4, 8)
        },
        "counterfactual": "replace current MASKED route sets with deterministic same-invocation DECODED route-set cycle",
    }


def make_figures(summary: dict, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # 1/2: time imbalance and wait CDFs.
    for metric, filename, xlabel in (
        ("max_mean", "time_imbalance_cdf.png", "predicted expert-time max/mean"),
        ("synchronization_wait_fraction", "wait_fraction_cdf.png", "synchronization wait fraction"),
    ):
        plt.figure(figsize=(5.8, 3.6))
        for ep in (4, 8):
            values = np.load(root.parent / f"ep{ep}_{metric}.npy")
            ordered = np.sort(values); y = np.linspace(0, 1, len(ordered), endpoint=False)
            plt.plot(ordered, y, label=f"EP{ep}")
        plt.xlabel(xlabel); plt.ylabel("CDF"); plt.legend(); plt.tight_layout()
        plt.savefig(root / filename, dpi=170); plt.close()
    # 3/4 persistence.
    labels = ["EP2", "EP4", "EP8"]
    adjacent = [summary["h1"][f"ep{ep}"]["persistence"]["layer_local"]["adjacent_critical_rank_match"] for ep in (2,4,8)]
    random = [summary["h1"][f"ep{ep}"]["persistence"]["layer_local"]["matched_random_critical_rank_match"] for ep in (2,4,8)]
    global_b = [summary["h1"][f"ep{ep}"]["persistence"]["layer_local"]["global_majority_accuracy"] for ep in (2,4,8)]
    x = np.arange(3); plt.figure(figsize=(6, 3.6))
    plt.bar(x-.25, adjacent, .25, label="adjacent"); plt.bar(x, random, .25, label="matched random")
    plt.bar(x+.25, global_b, .25, label="global majority"); plt.xticks(x, labels)
    plt.ylabel("critical-rank match"); plt.legend(); plt.tight_layout()
    plt.savefig(root / "critical_rank_match_controls.png", dpi=170); plt.close()
    quantiles = ("p50", "p75", "p90", "p95", "p99", "max")
    plt.figure(figsize=(5.8, 3.6))
    for ep in (4, 8):
        run = summary["h1"][f"ep{ep}"]["persistence"]["layer_local"]["run_length"]
        plt.plot(range(len(quantiles)), [run[key] for key in quantiles], marker="o", label=f"EP{ep}")
    plt.xticks(range(len(quantiles)), [key.upper() for key in quantiles])
    plt.ylabel("critical-rank run length (refinements)"); plt.legend(); plt.tight_layout()
    plt.savefig(root / "critical_rank_run_length.png", dpi=170); plt.close()
    # 5 top-k concentration.
    plt.figure(figsize=(6, 3.6))
    for ep in (4,8):
        vals=[summary["h1"][f"ep{ep}"]["concentration"]["top_k_excess_mass_fraction"][f"top_{k}"]["p50"] for k in (1,2,4,8)]
        plt.plot((1,2,4,8), vals, marker="o", label=f"EP{ep}")
    plt.xlabel("top-K experts"); plt.ylabel("P50 excess-mass fraction"); plt.legend(); plt.tight_layout()
    plt.savefig(root / "topk_excess_contribution.png", dpi=170); plt.close()
    # 6 early recall.
    plt.figure(figsize=(6, 3.6))
    for ep in (4,8):
        vals=[summary["h1"][f"ep{ep}"]["predictability"]["early_1_2"][f"k{k}"]["excess_mass_recall_mean"] for k in (1,2,4,8)]
        base=[summary["h1"][f"ep{ep}"]["predictability"]["global_hot"][f"k{k}"]["excess_mass_recall_mean"] for k in (1,2,4,8)]
        plt.plot((1,2,4,8), vals, marker="o", label=f"EP{ep} early1-2")
        plt.plot((1,2,4,8), base, linestyle="--", label=f"EP{ep} global")
    plt.xlabel("K"); plt.ylabel("future excess-mass recall"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(root / "early_hot_expert_recall.png", dpi=170); plt.close()
    # 7 perfect balance.
    vals=[100*summary["h1"][f"ep{ep}"]["perfect_balance"]["aggregate_routed_moe_stage_reduction"] for ep in (4,8)]
    plt.figure(figsize=(4.5,3.4)); plt.bar((0,1),vals); plt.xticks((0,1),("EP4","EP8")); plt.ylabel("routed-MoE reduction (%)"); plt.tight_layout()
    plt.savefig(root / "perfect_balance_headroom.png",dpi=170); plt.close()
    # 8-11 replicas (absent in the fast structural smoke mode).
    if "replication" in summary["h1"]["ep4"]:
        for metric, filename, ylabel in (
            ("aggregate_stage_reduction","replica_budget_stage.png","routed-MoE reduction (%)"),
            ("candidate_max_mean_p50","replica_budget_maxmean.png","candidate max/mean P50"),
        ):
            plt.figure(figsize=(6.4,3.8))
            for ep in (4,8):
                for policy in ("global_static","early_1_2","full_future_block"):
                    values=[]
                    for budget in REPLICA_BUDGETS:
                        row=summary["h1"][f"ep{ep}"]["replication"][policy][f"budget_{budget}"]
                        values.append(100*row[metric] if metric=="aggregate_stage_reduction" else row["candidate"]["max_mean"]["p50"])
                    plt.plot(REPLICA_BUDGETS,values,marker="o",label=f"EP{ep} {policy}")
            plt.xlabel("replica instances/layer"); plt.ylabel(ylabel); plt.legend(fontsize=7); plt.tight_layout()
            plt.savefig(root/filename,dpi=170); plt.close()
        plt.figure(figsize=(5.8,3.6))
        for ep in (4,8):
            vals=[]
            for budget in REPLICA_BUDGETS:
                row=summary["h1"][f"ep{ep}"]["replication"]["early_1_2"][f"budget_{budget}"]
                vals.append(row.get("block_triggered_replication",{}).get("setup_charged_feasible_block_fraction",0))
            plt.plot(REPLICA_BUDGETS,vals,marker="o",label=f"EP{ep}")
        plt.xlabel("replica instances/layer"); plt.ylabel("blocks amortizing setup"); plt.legend(); plt.tight_layout()
        plt.savefig(root/"replica_break_even.png",dpi=170); plt.close()
        plt.figure(figsize=(6.2,3.6))
        names=("global_static","early_1","early_1_2","full_future_oracle_envelope")
        labels=("global static","early 1","early 1–2","full-future envelope")
        x=np.arange(len(names)); width=.36
        for offset,ep in ((-.18,4),(.18,8)):
            values=[]
            for policy in names:
                values.append(100*summary["h1"][f"ep{ep}"]["replication"][policy]["budget_8"]["aggregate_stage_reduction"])
            plt.bar(x+offset,values,width,label=f"EP{ep}")
        plt.xticks(x,labels,rotation=15);plt.ylabel("budget-8 stage reduction (%)");plt.legend();plt.tight_layout()
        plt.savefig(root/"replication_policy_comparison.png",dpi=170);plt.close()
    # H2 figures 12/13.
    h2=summary["h2"]
    j=[h2["same_position"]["ep4_destination_rank_set_jaccard"]["p50"],h2["same_position"]["ep8_destination_rank_set_jaccard"]["p50"]]
    plt.figure(figsize=(4.5,3.4)); plt.bar((0,1),j);plt.xticks((0,1),("EP4","EP8"));plt.ylabel("destination-set Jaccard P50");plt.tight_layout();plt.savefig(root/"h2_destination_jaccard.png",dpi=170);plt.close()
    effects=[]
    for ep in (4,8): effects.append(100*h2["full_workload_composition_effect"][f"ep{ep}"]["routed_moe_stage"]["p95"])
    plt.figure(figsize=(4.5,3.4));plt.bar((0,1),effects);plt.xticks((0,1),("EP4","EP8"));plt.ylabel("full workload stage effect P95 (%)");plt.tight_layout();plt.savefig(root/"h2_full_workload_effect.png",dpi=170);plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--compute-ep2", type=Path, required=True)
    parser.add_argument("--compute-ep4", type=Path, required=True)
    parser.add_argument("--compute-ep8", type=Path, required=True)
    parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-replication", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = load_discovery_trace(args.trace)
    arrays = trace.arrays
    histograms = arrays["expert_counts_by_class"].sum(axis=1, dtype=np.int32)
    train, heldout = set(range(64)), set(range(64,128))
    models = {
        2: BatchComputeModel(args.compute_ep2),
        4: BatchComputeModel(args.compute_ep4),
        8: BatchComputeModel(args.compute_ep8),
    }
    communication = CommunicationScenario.load(args.communication, "ep2_calibrated_base")
    times = {}
    for ep in (2,4,8):
        cache = args.output / f"rank_times_ep{ep}.npy"
        if cache.exists():
            times[ep] = np.load(cache)
        else:
            times[ep] = rank_times(histograms, ep, models[ep])
            np.save(cache, times[ep])
        held_mask = np.isin(arrays["request_id"], list(heldout))
        selected = times[ep][held_mask]
        maximum, mean = selected.max(axis=1), selected.mean(axis=1)
        np.save(args.output / f"ep{ep}_max_mean.npy", maximum / mean)
        np.save(args.output / f"ep{ep}_synchronization_wait_fraction.npy",
                (ep * maximum - selected.sum(axis=1)) / (ep * maximum))
    summary = {
        "evidence": {
            "trace": str(args.trace), "invocations": len(arrays["request_id"]),
            "refinement_forwards": int(len(arrays["request_id"]) // ROUTED_LAYERS),
            "requests": len(set(map(int, arrays["request_id"]))),
            "accuracy": "118/128 = 92.19%", "train_requests": "0-63",
            "heldout_requests": "64-127", "physical_gpus_used_for_this_analysis": [],
            "ep4_label": "SIMULATED-EP4-EP2-CALIBRATED",
            "ep8_label": "SIMULATED-EP8-EP2-CALIBRATED",
        },
        "h1": {},
    }
    held_mask = np.isin(arrays["request_id"], list(heldout))
    for ep in (2,4,8):
        summary["h1"][f"ep{ep}"] = {
            "assignment_severity": imbalance_metrics(arrays[f"rank_load_ep{ep}"][held_mask]),
            "time_severity": imbalance_metrics(times[ep][held_mask]),
            "persistence": persistence_analysis(arrays, times[ep], ep, train, heldout),
            "perfect_balance": perfect_balance(arrays, times[ep], ep, heldout, communication),
        }
        if ep in (4,8):
            summary["h1"][f"ep{ep}"]["concentration"] = concentration_analysis(
                histograms, times[ep], arrays["request_id"], heldout, ep
            )
            summary["h1"][f"ep{ep}"]["predictability"] = hot_expert_predictability(
                arrays, histograms, times[ep], ep, train, heldout
            )
            if not args.skip_replication:
                policies = build_replica_policies(arrays, histograms, times[ep], ep, train, heldout)
                replication = {
                    name: evaluate_replica_policy(
                        arrays, histograms, times[ep], models[ep], communication,
                        ep, heldout, policy,
                    ) for name, policy in policies.items()
                }
                # Full-future information may choose either the block-greedy
                # candidates or the globally hot list.  This restricted
                # block-level envelope makes the oracle no worse than a
                # feasible static policy without inventing per-invocation
                # switching.
                envelope = {}
                for budget in REPLICA_BUDGETS:
                    name = f"budget_{budget}"
                    global_blocks = replication["global_static"][name]["block_stage_totals"]
                    future_blocks = replication["full_future_block"][name]["block_stage_totals"]
                    baseline_sum = 0.0; candidate_sum = 0.0; future_selected = 0
                    reductions = []
                    for key, global_row in global_blocks.items():
                        future_row = future_blocks[key]
                        baseline = global_row["baseline_ms"]
                        candidate = min(global_row["candidate_ms"], future_row["candidate_ms"])
                        baseline_sum += baseline; candidate_sum += candidate
                        reductions.append((baseline - candidate) / baseline)
                        future_selected += int(future_row["candidate_ms"] < global_row["candidate_ms"])
                    envelope[name] = {
                        "aggregate_stage_reduction": (baseline_sum - candidate_sum) / baseline_sum,
                        "block_stage_reduction": distribution(reductions),
                        "future_greedy_selected_block_fraction": future_selected / len(global_blocks),
                        "candidate_pool": "min(global-static, full-future-greedy) selected once per block",
                    }
                replication["full_future_oracle_envelope"] = envelope
                summary["h1"][f"ep{ep}"]["replication"] = replication
                (args.output / f"partial_ep{ep}.json").write_text(
                    json.dumps(summary["h1"][f"ep{ep}"], indent=2) + "\n"
                )
    summary["h2"] = h2_ep8_reanalysis(trace, models, communication, heldout, times)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figures(summary, args.output / "figures")
    print(json.dumps({
        "ep4_perfect_stage": summary["h1"]["ep4"]["perfect_balance"]["aggregate_routed_moe_stage_reduction"],
        "ep8_perfect_stage": summary["h1"]["ep8"]["perfect_balance"]["aggregate_routed_moe_stage_reduction"],
        "h2_ep8_p95": summary["h2"]["full_workload_composition_effect"]["ep8"]["routed_moe_stage"]["p95"],
    }, indent=2))


if __name__ == "__main__":
    main()
