#!/usr/bin/env python3
"""Analyze four dLLM--MoE--EP discovery hypotheses from aggregate trace v2."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from virtual_ep.comm_model import CommunicationScenario
from virtual_ep.compute_model import ComputeModel
from virtual_ep.discovery_trace import DiscoveryTrace, POSITION_CLASSES


HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 8
EXPERT_WEIGHT_BYTES = (2048 * 1024 + 512 * 2048) * 2


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def percentiles(values) -> dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {name: float("nan") for name in ("p50", "p90", "p95", "p99", "max")}
    return {
        "p50": float(np.percentile(values, 50)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def paired_bootstrap(a: dict, b: dict, seed: int = 20260917, repeats: int = 2000) -> dict:
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"difference": float("nan"), "ci95": [float("nan"), float("nan")], "units": 0}
    delta = np.asarray([float(a[key]) - float(b[key]) for key in keys])
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for i in range(repeats):
        samples[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    return {
        "difference": float(delta.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "units": len(keys),
    }


def jsd(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2.0) ** 2)


def tv(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


class CostModel:
    def __init__(self, ep: int, compute_path: Path, communication_path: Path):
        self.ep = ep
        self.compute = ComputeModel.from_csv(compute_path)
        self._compute_tree = cKDTree(self.compute._features / self.compute._scale)
        self._rank_cache: dict[bytes, np.ndarray] = {}
        self.communication = CommunicationScenario.load(communication_path, "ep2_calibrated_base")

    def _predict_histogram(self, histogram: np.ndarray) -> float:
        counts = np.asarray(histogram, dtype=np.float64)
        nonzero = counts[counts > 0]
        if nonzero.size:
            query = np.asarray([
                len(nonzero), nonzero.sum(), nonzero.max(),
                nonzero.std() / nonzero.mean(),
            ])
        else:
            query = np.zeros(4)
        _distance, index = self._compute_tree.query(query / self.compute._scale)
        return float(self.compute.samples[int(index)].latency_ms)

    def routes(self, routes: np.ndarray, sources: np.ndarray | None = None,
               owner: np.ndarray | None = None) -> dict:
        routes = np.asarray(routes, dtype=np.int16).reshape(-1, TOP_K)
        if owner is None:
            owner = np.arange(NUM_EXPERTS, dtype=np.int16) // (NUM_EXPERTS // self.ep)
        if sources is None:
            base, remainder = divmod(len(routes), self.ep)
            sizes = np.full(self.ep, base, dtype=int); sizes[:remainder] += 1
            sources = np.repeat(np.arange(self.ep), sizes)
        owners = owner[routes]
        rank_load = np.bincount(owners.reshape(-1), minlength=self.ep)
        matrix = np.zeros((self.ep, self.ep), dtype=np.int64)
        unique = np.zeros_like(matrix)
        np.add.at(matrix, (np.repeat(sources, TOP_K), owners.reshape(-1)), 1)
        for destination in range(self.ep):
            hit = np.any(owners == destination, axis=1)
            unique[:, destination] = np.bincount(sources[hit], minlength=self.ep)
        remote = unique.copy(); np.fill_diagonal(remote, 0)
        outgoing = remote.sum(axis=1) * HIDDEN * 2
        incoming = remote.sum(axis=0) * HIDDEN * 2
        histogram = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
        rank_times = []
        for rank in range(self.ep):
            rank_times.append(self._predict_histogram(histogram[owner == rank]))
        dispatch = self.communication.dispatch.latency(outgoing, incoming)
        combine = self.communication.combine.latency(incoming, outgoing)
        return {
            "rank_load": rank_load, "assignment_matrix": matrix, "unique_matrix": unique,
            "remote_bytes": int(remote.sum() * HIDDEN * 2),
            "fanout": float(np.mean([len(set(row)) for row in owners])) if len(routes) else 0.0,
            "rank_times_ms": np.asarray(rank_times), "critical_rank": int(np.argmax(rank_times)),
            "dispatch_ms": dispatch, "expert_ms": float(max(rank_times, default=0)),
            "combine_ms": combine, "stage_ms": float(dispatch + max(rank_times, default=0) + combine),
        }

    def rank_times_from_histogram(self, histogram: np.ndarray) -> np.ndarray:
        histogram = np.asarray(histogram)
        cache_key = histogram.tobytes()
        cached = self._rank_cache.get(cache_key)
        if cached is not None:
            return cached
        owner = np.arange(NUM_EXPERTS) // (NUM_EXPERTS // self.ep)
        values = []
        for rank in range(self.ep):
            values.append(self._predict_histogram(histogram[owner == rank]))
        result = np.asarray(values)
        self._rank_cache[cache_key] = result
        return result


def key_rows(trace: DiscoveryTrace):
    a = trace.arrays
    for i in range(len(a["request_id"])):
        yield (int(a["request_id"][i]), int(a["block_id"][i]),
               int(a["iteration_id"][i]), int(a["layer_id"][i])), i


def current_generated_routes(trace: DiscoveryTrace, index: int) -> np.ndarray:
    """Exact current-block generated rows, excluding prompt tail in block zero."""
    classes = trace.arrays["current_position_class"][index]
    return trace.arrays["current_expert_ids"][index][classes >= 2]


def current_generated_sources(trace: DiscoveryTrace, index: int, ep: int = 4) -> np.ndarray:
    classes = trace.arrays["current_position_class"][index]
    return trace.arrays[f"current_source_rank_ep{ep}"][index][classes >= 2]


def load_groups(trace: DiscoveryTrace, ep: int, current_only: bool = False):
    a = trace.arrays
    grouped = defaultdict(list)
    if current_only:
        loads = np.stack([
            np.bincount(
                (current_generated_routes(trace, i) // (NUM_EXPERTS // ep)).reshape(-1),
                minlength=ep,
            )
            for i in range(len(a["request_id"]))
        ])
    else:
        loads = a[f"rank_load_ep{ep}"]
    for i in range(len(loads)):
        grouped[(int(a["request_id"][i]), int(a["block_id"][i]), int(a["layer_id"][i]))].append(
            (int(a["iteration_id"][i]), i, loads[i].astype(np.float64))
        )
    for values in grouped.values():
        values.sort()
    return grouped


def analyze_h1(trace: DiscoveryTrace, train: set[int], heldout: set[int],
               models: dict[int, CostModel]) -> dict:
    a = trace.arrays
    output = {"label": "H1_BLOCK_PERSISTENT_STRAGGLER", "targets": {}}
    for ep in (2, 4, 8):
        model = models[ep]
        compute_vectors = np.stack([
            model.rank_times_from_histogram(histogram)
            for histogram in a["expert_counts_by_class"].sum(axis=1)
        ])
        grouped = defaultdict(list)
        for i in range(len(a["request_id"])):
            grouped[(int(a["request_id"][i]), int(a["block_id"][i]),
                     int(a["layer_id"][i]))].append(
                (int(a["iteration_id"][i]), i, compute_vectors[i]))
        for values in grouped.values():
            values.sort()
        same_block = defaultdict(list); random_control = defaultdict(list)
        same_cosine = defaultdict(list); random_cosine = defaultdict(list)
        adjacent_matches = []; adjacent_cosines = []; adjacent_spearman = []
        run_lengths = []; margins = []
        rng = np.random.default_rng(20260917 + ep)
        pools = defaultdict(list)
        for key, values in grouped.items():
            if key[0] in heldout:
                pools[(key[1], key[2])].extend([(key[0], v[2]) for v in values])
        for key, values in grouped.items():
            if key[0] not in heldout:
                continue
            block_key = (key[0], key[1])
            critical_sequence = [int(np.argmax(value[2])) for value in values]
            if critical_sequence:
                run = 1
                for left_rank, right_rank in zip(critical_sequence, critical_sequence[1:]):
                    if left_rank == right_rank:
                        run += 1
                    else:
                        run_lengths.append(run); run = 1
                run_lengths.append(run)
            for value in values:
                ordered = np.sort(value[2])
                margins.append(float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0)
            for left, right in zip(values, values[1:]):
                match = float(np.argmax(left[2]) == np.argmax(right[2]))
                sim = cosine(left[2], right[2])
                same_block[block_key].append(match)
                same_cosine[block_key].append(sim)
                adjacent_matches.append(match); adjacent_cosines.append(sim)
                adjacent_spearman.append(float(spearmanr(left[2], right[2]).statistic))
                pool = [candidate for candidate in pools[(key[1], key[2])]
                        if candidate[0] != key[0]]
                if not pool:
                    pool = pools[(key[1], key[2])]
                candidate = pool[int(rng.integers(len(pool)))][1]
                random_control[block_key].append(float(np.argmax(left[2]) == np.argmax(candidate)))
                random_cosine[block_key].append(cosine(left[2], candidate))
        sb = {key: float(np.mean(values)) for key, values in same_block.items()}
        rc = {key: float(np.mean(random_control[key])) for key in sb}
        sc = {key: float(np.mean(values)) for key, values in same_cosine.items()}
        rcos = {key: float(np.mean(random_cosine[key])) for key in sc}

        forward = defaultdict(lambda: np.zeros(ep, dtype=np.float64))
        for i in range(len(a["request_id"])):
            request = int(a["request_id"][i])
            if request in heldout:
                forward[(request, int(a["block_id"][i]), int(a["iteration_id"][i]))] += compute_vectors[i]
        forward_groups = defaultdict(list)
        for (request, block, iteration), vector in forward.items():
            forward_groups[(request, block)].append((iteration, vector))
        forward_match = []; forward_cosine = []; early1 = []; early2 = []
        early2_cosine = []; early2_traffic_cosine = []
        cross_block = []
        by_request = defaultdict(dict)
        for block_key, values in forward_groups.items():
            values.sort()
            by_request[block_key[0]][block_key[1]] = values
            for left, right in zip(values, values[1:]):
                forward_match.append(float(np.argmax(left[1]) == np.argmax(right[1])))
                forward_cosine.append(cosine(left[1], right[1]))
            if len(values) > 1:
                first = values[0][1]
                first_two = np.mean([v[1] for v in values[:2]], axis=0)
                for _, later in values[1:]:
                    early1.append(float(np.argmax(first) == np.argmax(later)))
                    early2.append(float(np.argmax(first_two) == np.argmax(later)))
                    early2_cosine.append(cosine(first_two, later))
            # Traffic prediction uses the first two full-forward assignment matrices.
            matrix_by_iteration = defaultdict(lambda: np.zeros((ep, ep)))
            mask = ((a["request_id"] == block_key[0]) & (a["block_id"] == block_key[1]))
            for index in np.flatnonzero(mask):
                matrix_by_iteration[int(a["iteration_id"][index])] += a[f"assignment_matrix_ep{ep}"][index]
            ordered_matrices = [matrix_by_iteration[i].reshape(-1) for i in sorted(matrix_by_iteration)]
            if len(ordered_matrices) > 1:
                early_matrix = np.mean(ordered_matrices[:2], axis=0)
                early2_traffic_cosine.extend(cosine(early_matrix, later) for later in ordered_matrices[1:])
        for request, blocks in by_request.items():
            ordered = sorted(blocks)
            for left, right in zip(ordered, ordered[1:]):
                cross_block.append(float(np.argmax(blocks[left][-1][1]) == np.argmax(blocks[right][0][1])))
        train_critical = []
        for i in range(len(a["request_id"])):
            if int(a["request_id"][i]) in train:
                train_critical.append(int(np.argmax(compute_vectors[i])))
        majority_rank = int(np.argmax(np.bincount(train_critical, minlength=ep)))
        majority_accuracy = float(np.mean([
            majority_rank == int(np.argmax(value[1]))
            for values in forward_groups.values() for value in values
        ]))
        previous_block_accuracy = []
        for request, blocks in by_request.items():
            ordered = sorted(blocks)
            for previous, current in zip(ordered, ordered[1:]):
                prediction = int(np.argmax(np.mean([row[1] for row in blocks[previous][-2:]], axis=0)))
                previous_block_accuracy.extend(prediction == int(np.argmax(row[1])) for row in blocks[current])
        current_matches = []; current_cosines = []
        for key, values in load_groups(trace, ep, current_only=True).items():
            if key[0] not in heldout:
                continue
            for left, right in zip(values, values[1:]):
                current_matches.append(float(np.argmax(left[2]) == np.argmax(right[2])))
                current_cosines.append(cosine(left[2], right[2]))
        output["targets"][f"ep{ep}"] = {
            "layer_local_adjacent_critical_rank_match": float(np.mean(adjacent_matches)),
            "layer_local_adjacent_load_cosine": float(np.mean(adjacent_cosines)),
            "layer_local_adjacent_spearman": float(np.nanmean(adjacent_spearman)),
            "critical_rank_run_length": percentiles(run_lengths),
            "critical_rank_compute_margin_ms": percentiles(margins),
            "matched_random_critical_rank_match": float(np.mean([x for v in random_control.values() for x in v])),
            "same_block_minus_random_bootstrap": paired_bootstrap(sb, rc),
            "same_block_cosine_minus_random_bootstrap": paired_bootstrap(sc, rcos),
            "forward_adjacent_critical_rank_match": float(np.mean(forward_match)),
            "forward_adjacent_rank_time_cosine": float(np.mean(forward_cosine)),
            "current_generated_rows_adjacent_critical_rank_match": float(np.mean(current_matches)),
            "current_generated_rows_adjacent_load_cosine": float(np.mean(current_cosines)),
            "same_request_cross_block_boundary_match": float(np.mean(cross_block)),
            "early1_future_critical_rank_accuracy": float(np.mean(early1)),
            "early2_future_critical_rank_accuracy": float(np.mean(early2)),
            "early2_future_rank_time_cosine": float(np.mean(early2_cosine)),
            "early2_future_traffic_cosine": float(np.mean(early2_traffic_cosine)),
            "global_majority_rank_accuracy": majority_accuracy,
            "previous_block_rank_accuracy": float(np.mean(previous_block_accuracy)),
            "uniform_rank_reference": 1.0 / ep,
            "pairs": len(adjacent_matches), "blocks": len(sb),
        }
    ep4 = output["targets"]["ep4"]
    ci = ep4["same_block_minus_random_bootstrap"]["ci95"]
    match_delta = ep4["same_block_minus_random_bootstrap"]["difference"]
    cosine_delta = ep4["same_block_cosine_minus_random_bootstrap"]["difference"]
    early_baseline = max(ep4["global_majority_rank_accuracy"], ep4["previous_block_rank_accuracy"])
    strong = ci[0] > 0 and (match_delta >= .15 or cosine_delta >= .10) and (
        ep4["early2_future_critical_rank_accuracy"] > early_baseline)
    output["verdict_pre_ep2"] = "GO" if strong else "HOLD" if ci[1] > 0 else "NO-GO"
    output["ar_control"] = "Cross-block and matched-random controls represent non-repeated/generic routing; only excess same-block persistence is dLLM-specific."
    return output


def analyze_h2(trace: DiscoveryTrace, heldout: set[int]) -> dict:
    a = trace.arrays
    class_index = {name: i for i, name in enumerate(POSITION_CLASSES)}
    pairs = [
        ("CURRENT_BLOCK_MASKED", "CURRENT_BLOCK_NEWLY_ACCEPTED"),
        ("CURRENT_BLOCK_MASKED", "CURRENT_BLOCK_DECODED"),
        ("PROMPT_PREFIX", "PRIOR_GENERATED_BLOCKS"),
    ]
    metrics = {}
    per_block = defaultdict(lambda: defaultdict(list))
    for left_name, right_name in pairs:
        left_id, right_id = class_index[left_name], class_index[right_name]
        values = []
        for i in range(len(a["request_id"])):
            request = int(a["request_id"][i])
            if request not in heldout:
                continue
            left = a["expert_counts_by_class"][i, left_id]
            right = a["expert_counts_by_class"][i, right_id]
            if left.sum() and right.sum():
                row = (jsd(left, right), tv(left, right), cosine(left, right))
                values.append(row)
                per_block[(request, int(a["block_id"][i]))][f"{left_name}__{right_name}"].append(row[0])
        metrics[f"{left_name}__{right_name}"] = {
            "jsd": percentiles(x[0] for x in values), "tv": percentiles(x[1] for x in values),
            "cosine": percentiles(x[2] for x in values), "comparisons": len(values),
        }
    state_load = {}
    for name in ("CURRENT_BLOCK_MASKED", "CURRENT_BLOCK_NEWLY_ACCEPTED", "CURRENT_BLOCK_DECODED"):
        class_id = class_index[name]
        ratios = []; cvs = []; hottest = []
        for i in range(len(a["request_id"])):
            if int(a["request_id"][i]) not in heldout:
                continue
            hist = a["expert_counts_by_class"][i, class_id].astype(np.float64)
            if not hist.sum():
                continue
            rank = hist.reshape(4, 64).sum(axis=1)
            ratios.append(rank.max() / rank.mean()); cvs.append(rank.std() / rank.mean())
            hottest.append(int(np.argmax(rank)))
        state_load[name] = {
            "max_mean": percentiles(ratios), "rank_cv": percentiles(cvs),
            "hottest_rank_distribution": np.bincount(hottest, minlength=4).tolist(),
        }

    max_iteration = defaultdict(int)
    for i in range(len(a["request_id"])):
        key = (int(a["request_id"][i]), int(a["block_id"][i]))
        max_iteration[key] = max(max_iteration[key], int(a["iteration_id"][i]))
    phase_jsd = defaultdict(list); mask_ratios = []; total_max_mean = []
    composition_max_mean_change = []; composition_by_phase = defaultdict(list)
    for i in range(len(a["request_id"])):
        request = int(a["request_id"][i])
        if request not in heldout:
            continue
        block = int(a["block_id"][i]); iteration = int(a["iteration_id"][i])
        progress = iteration / max(max_iteration[(request, block)], 1)
        phase = "early" if progress < 1/3 else "middle" if progress < 2/3 else "late"
        masked_hist = a["expert_counts_by_class"][i, class_index["CURRENT_BLOCK_MASKED"]]
        decoded_hist = a["expert_counts_by_class"][i, class_index["CURRENT_BLOCK_DECODED"]]
        value = jsd(masked_hist, decoded_hist)
        if not np.isnan(value):
            phase_jsd[phase].append(value)
        mask_ratios.append(float(a["masked_current_block"][i]) / 32.0)
        total_max_mean.append(float(a["max_mean_ep4"][i]))
        masked_rank = masked_hist.reshape(4, 64).sum(axis=1).astype(np.float64)
        decoded_rank = decoded_hist.reshape(4, 64).sum(axis=1).astype(np.float64)
        if masked_rank.sum() and decoded_rank.sum():
            total_rank = a["rank_load_ep4"][i].astype(np.float64)
            counterfactual = total_rank - masked_rank + masked_rank.sum() * decoded_rank / decoded_rank.sum()
            baseline_ratio = total_rank.max() / total_rank.mean()
            counterfactual_ratio = counterfactual.max() / counterfactual.mean()
            effect = abs(counterfactual_ratio - baseline_ratio) / baseline_ratio
            composition_max_mean_change.append(effect)
            composition_by_phase[phase].append(effect)
    mask_load_spearman = float(spearmanr(mask_ratios, total_max_mean).statistic)

    transition_jaccard = []; transition_destination = []
    groups = defaultdict(list)
    for key, i in key_rows(trace):
        if key[0] in heldout:
            groups[(key[0], key[1], key[3])].append((key[2], i))
    for values in groups.values():
        values.sort()
        for (_, left), (_, right) in zip(values, values[1:]):
            left_class = a["current_position_class"][left]
            right_class = a["current_position_class"][right]
            changed = (left_class == 2) & np.isin(right_class, [3, 4])
            for position in np.flatnonzero(changed):
                x = set(map(int, a["current_expert_ids"][left, position]))
                y = set(map(int, a["current_expert_ids"][right, position]))
                transition_jaccard.append(len(x & y) / len(x | y))
                xr = {item // 64 for item in x}; yr = {item // 64 for item in y}
                transition_destination.append(len(xr & yr) / len(xr | yr))
    masked = state_load["CURRENT_BLOCK_MASKED"]["max_mean"]["p50"]
    decoded = state_load["CURRENT_BLOCK_DECODED"]["max_mean"]["p50"]
    relative_load_gap = abs(masked - decoded) / max(masked, decoded) if max(masked, decoded) else 0
    d_llm_jsd = metrics["CURRENT_BLOCK_MASKED__CURRENT_BLOCK_DECODED"]["jsd"]["p50"]
    generic_jsd = metrics["PROMPT_PREFIX__PRIOR_GENERATED_BLOCKS"]["jsd"]["p50"]
    dllm_key = "CURRENT_BLOCK_MASKED__CURRENT_BLOCK_DECODED"
    generic_key = "PROMPT_PREFIX__PRIOR_GENERATED_BLOCKS"
    dllm_blocks = {key: float(np.mean(values[dllm_key])) for key, values in per_block.items()
                   if values[dllm_key]}
    generic_blocks = {key: float(np.mean(values[generic_key])) for key, values in per_block.items()
                      if values[generic_key]}
    state_vs_generic_bootstrap = paired_bootstrap(dllm_blocks, generic_blocks)
    practical_effect = float(np.median(composition_max_mean_change)) if composition_max_mean_change else 0.0
    verdict = "GO" if (d_llm_jsd >= 0.05 and practical_effect >= 0.05 and
                        state_vs_generic_bootstrap["ci95"][0] > 0) else (
        "HOLD" if d_llm_jsd >= 0.02 and max(composition_max_mean_change, default=0) >= .05 else "NO-GO")
    return {
        "label": "H2_MASK_STATE_SPECIALIZATION", "distribution_metrics": metrics,
        "ep4_state_load": state_load, "median_relative_ep4_max_mean_gap": relative_load_gap,
        "same_position_mask_to_decoded_topk_jaccard": percentiles(transition_jaccard),
        "same_position_mask_to_decoded_destination_jaccard": percentiles(transition_destination),
        "transition_samples": len(transition_jaccard), "verdict": verdict,
        "state_jsd_minus_generic_bootstrap": state_vs_generic_bootstrap,
        "masked_vs_decoded_jsd_by_phase": {name: percentiles(values) for name, values in phase_jsd.items()},
        "mask_ratio_vs_ep4_max_mean_spearman": mask_load_spearman,
        "actual_full_workload_ep4_max_mean_change_from_state_composition": percentiles(composition_max_mean_change),
        "composition_effect_by_phase": {name: percentiles(values) for name, values in composition_by_phase.items()},
        "ar_control": "PROMPT_PREFIX vs PRIOR_GENERATED_BLOCKS measures generic token-age/type specialization; MASK-state excess and same-position transitions are dLLM-specific.",
    }


def graph_from_routes(routes) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(routes, list):
        routes = np.concatenate([np.asarray(item).reshape(-1, TOP_K) for item in routes if len(item)], axis=0)
    routes = np.asarray(routes, dtype=np.int16).reshape(-1, TOP_K)
    graph = np.zeros((NUM_EXPERTS, NUM_EXPERTS), dtype=np.float64)
    for left in range(TOP_K):
        for right in range(left + 1, TOP_K):
            np.add.at(graph, (routes[:, left], routes[:, right]), 1)
            np.add.at(graph, (routes[:, right], routes[:, left]), 1)
    frequency = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS).astype(np.float64)
    return graph, frequency


def source_affinity(routes, sources, ep: int = 4) -> np.ndarray:
    affinity = np.zeros((NUM_EXPERTS, ep), dtype=np.float64)
    if not isinstance(routes, list):
        routes, sources = [routes], [sources]
    for selected, source in zip(routes, sources):
        selected = np.asarray(selected).reshape(-1, TOP_K)
        source = np.asarray(source).reshape(-1)
        np.add.at(affinity, (selected.reshape(-1), np.repeat(source, TOP_K)), 1)
    return affinity


def optimize_placement(graph: np.ndarray, frequency: np.ndarray, ep: int = 4,
                       steps: int = 3000, seed: int = 0,
                       affinity: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    owner = np.arange(NUM_EXPERTS, dtype=np.int16) // (NUM_EXPERTS // ep)
    total_edges = max(graph.sum() / 2, 1.0)
    affinity_total = max(affinity.sum(), 1.0) if affinity is not None else 1.0
    cut_raw = graph[np.not_equal.outer(owner, owner)].sum() / 2
    loads = np.bincount(owner, weights=frequency, minlength=ep)
    local = (affinity[np.arange(NUM_EXPERTS), owner].sum()
             if affinity is not None else 0.0)

    def objective(cut_value, load_value, local_value):
        cut = cut_value / total_edges
        balance = load_value.std() / max(load_value.mean(), 1.0)
        remote = ((affinity_total - local_value) / affinity_total
                  if affinity is not None else 0.0)
        return 0.5 * cut + 0.5 * remote + 0.75 * balance

    current = objective(cut_raw, loads, local); best = current; best_owner = owner.copy()
    temperature = 0.02
    for step in range(steps):
        a = int(rng.integers(NUM_EXPERTS)); candidates = np.flatnonzero(owner != owner[a])
        b = int(rng.choice(candidates))
        rank_a, rank_b = int(owner[a]), int(owner[b])
        other = np.ones(NUM_EXPERTS, dtype=bool); other[[a, b]] = False
        old_incident = (
            graph[a, other][rank_a != owner[other]].sum() +
            graph[b, other][rank_b != owner[other]].sum()
        )
        new_incident = (
            graph[a, other][rank_b != owner[other]].sum() +
            graph[b, other][rank_a != owner[other]].sum()
        )
        proposal_cut = cut_raw + new_incident - old_incident
        proposal_loads = loads.copy()
        proposal_loads[rank_a] += frequency[b] - frequency[a]
        proposal_loads[rank_b] += frequency[a] - frequency[b]
        proposal_local = local
        if affinity is not None:
            proposal_local += (
                affinity[a, rank_b] + affinity[b, rank_a] -
                affinity[a, rank_a] - affinity[b, rank_b]
            )
        value = objective(proposal_cut, proposal_loads, proposal_local)
        if value < current or rng.random() < math.exp((current - value) / max(temperature, 1e-6)):
            owner[a], owner[b] = owner[b], owner[a]
            cut_raw, loads, local = proposal_cut, proposal_loads, proposal_local
            current = value
            if value < best:
                best = value; best_owner = owner.copy()
        temperature *= 0.999
    return best_owner


def migration_cost(owner: np.ndarray, bandwidth_gbps: float = 85.51,
                   layers: int = 19) -> dict:
    baseline = np.arange(NUM_EXPERTS) // 64
    changed = np.flatnonzero(owner != baseline)
    outgoing = np.zeros(4); incoming = np.zeros(4)
    for expert in changed:
        outgoing[baseline[expert]] += EXPERT_WEIGHT_BYTES * layers
        incoming[owner[expert]] += EXPERT_WEIGHT_BYTES * layers
    return {
        "moved_expert_instances": int(len(changed) * layers),
        "moved_expert_ids_per_layer": int(len(changed)),
        "weight_bytes": int(len(changed) * EXPERT_WEIGHT_BYTES * layers),
        "endpoint_lower_bound_ms": float(max(outgoing.max(), incoming.max()) / (bandwidth_gbps * 1e9) * 1e3),
    }


def evaluate_routes(indices: list[int], trace: DiscoveryTrace, model: CostModel,
                    owner: np.ndarray) -> dict:
    a = trace.arrays
    baseline = np.arange(NUM_EXPERTS, dtype=np.int16) // 64
    result = {"baseline": defaultdict(float), "candidate": defaultdict(float)}
    for i in indices:
        valid = a["current_position_class"][i] >= 2
        routes = a["current_expert_ids"][i][valid]
        sources = a["current_source_rank_ep4"][i][valid]
        for name, mapping in (("baseline", baseline), ("candidate", owner)):
            cost = model.routes(routes, sources, mapping)
            for metric in ("remote_bytes", "fanout", "expert_ms", "stage_ms"):
                result[name][metric] += float(cost[metric])
            result[name]["max_load"] += float(cost["rank_load"].max())
    output = {}
    for metric in result["baseline"]:
        before = result["baseline"][metric]; after = result["candidate"][metric]
        output[metric] = {"baseline": before, "candidate": after,
                          "reduction_fraction": (before - after) / before if before else 0.0}
    return output


def full_stage_for_indices(indices: list[int], trace: DiscoveryTrace, model: CostModel) -> float:
    a = trace.arrays; total = 0.0
    for i in indices:
        rank_times = model.rank_times_from_histogram(a["expert_counts_by_class"][i].sum(axis=0))
        outgoing = a["outgoing_bytes_ep4"][i]; incoming = a["incoming_bytes_ep4"][i]
        total += (model.communication.dispatch.latency(outgoing, incoming) + rank_times.max() +
                  model.communication.combine.latency(incoming, outgoing))
    return float(total)


def analyze_h3(trace: DiscoveryTrace, train: set[int], heldout: set[int], model: CostModel,
               max_oracle_blocks: int | None = None) -> dict:
    a = trace.arrays
    train_idx = [i for i, request in enumerate(a["request_id"]) if int(request) in train]
    held_idx = [i for i, request in enumerate(a["request_id"]) if int(request) in heldout]
    train_routes = [current_generated_routes(trace, i) for i in train_idx]
    train_sources = [current_generated_sources(trace, i) for i in train_idx]
    token_graph, frequency = graph_from_routes(train_routes)
    forward_graph = np.zeros_like(token_graph)
    for invocation in train_routes:
        active = np.unique(invocation)
        forward_graph[np.ix_(active, active)] += 1
    np.fill_diagonal(forward_graph, 0)
    graph_similarity = cosine(token_graph.reshape(-1), forward_graph.reshape(-1))
    global_affinity = source_affinity(train_routes, train_sources)
    global_owner = optimize_placement(
        token_graph, frequency, steps=5000, seed=7, affinity=global_affinity
    )
    static_eval = evaluate_routes(held_idx, trace, model, global_owner)
    static_full_denominator = full_stage_for_indices(held_idx, trace, model)
    static_eval["full_physical_stage_denominator_ms"] = static_full_denominator
    static_eval["full_physical_stage_mapped_reduction_fraction"] = (
        (static_eval["stage_ms"]["baseline"] - static_eval["stage_ms"]["candidate"])
        / static_full_denominator
    )
    blocks = defaultdict(list)
    for i in held_idx:
        blocks[(int(a["request_id"][i]), int(a["block_id"][i]))].append(i)
    selected_blocks = sorted(blocks)
    if max_oracle_blocks:
        selected_blocks = selected_blocks[:max_oracle_blocks]
    block_records = []
    for ordinal, block in enumerate(selected_blocks):
        indices = blocks[block]
        full_graph, full_frequency = graph_from_routes([current_generated_routes(trace, i) for i in indices])
        full_affinity = source_affinity(
            [current_generated_routes(trace, i) for i in indices],
            [current_generated_sources(trace, i) for i in indices],
        )
        early_indices = [i for i in indices if int(a["iteration_id"][i]) < 2]
        early_graph, early_frequency = graph_from_routes([current_generated_routes(trace, i) for i in early_indices])
        early_affinity = source_affinity(
            [current_generated_routes(trace, i) for i in early_indices],
            [current_generated_sources(trace, i) for i in early_indices],
        )
        oracle_owner = optimize_placement(
            full_graph, full_frequency, steps=700, seed=1000 + ordinal,
            affinity=full_affinity,
        )
        early_owner = optimize_placement(
            early_graph, early_frequency, steps=700, seed=2000 + ordinal,
            affinity=early_affinity,
        )
        oracle_eval = evaluate_routes(indices, trace, model, oracle_owner)
        early_eval = evaluate_routes(indices, trace, model, early_owner)
        full_denominator = full_stage_for_indices(indices, trace, model)
        oracle_migration = migration_cost(oracle_owner); early_migration = migration_cost(early_owner)
        oracle_saving = oracle_eval["stage_ms"]["baseline"] - oracle_eval["stage_ms"]["candidate"]
        early_saving = early_eval["stage_ms"]["baseline"] - early_eval["stage_ms"]["candidate"]
        refinements = len(set(int(a["iteration_id"][i]) for i in indices))
        oracle_per_refinement = oracle_saving / max(refinements, 1)
        early_per_refinement = early_saving / max(refinements, 1)
        block_records.append({
            "request": block[0], "block": block[1], "refinements": refinements,
            "oracle_stage_reduction": oracle_eval["stage_ms"]["reduction_fraction"],
            "early_stage_reduction": early_eval["stage_ms"]["reduction_fraction"],
            "oracle_full_stage_mapped_reduction": oracle_saving / full_denominator,
            "early_full_stage_mapped_reduction": early_saving / full_denominator,
            "oracle_remote_reduction": oracle_eval["remote_bytes"]["reduction_fraction"],
            "early_remote_reduction": early_eval["remote_bytes"]["reduction_fraction"],
            "oracle_migration_ms": oracle_migration["endpoint_lower_bound_ms"],
            "early_migration_ms": early_migration["endpoint_lower_bound_ms"],
            "oracle_break_even_refinements": (oracle_migration["endpoint_lower_bound_ms"] /
                                                  oracle_per_refinement
                                                  if oracle_per_refinement > 0 else float("inf")),
            "early_break_even_refinements": (early_migration["endpoint_lower_bound_ms"] /
                                                 early_per_refinement
                                                 if early_per_refinement > 0 else float("inf")),
            "early_moved_expert_instances": early_migration["moved_expert_instances"],
        })
    oracle_gain = float(np.mean([r["oracle_full_stage_mapped_reduction"] for r in block_records]))
    early_gain = float(np.mean([r["early_full_stage_mapped_reduction"] for r in block_records]))
    migration_feasible = float(np.mean([r["early_break_even_refinements"] <= r["refinements"] for r in block_records]))
    verdict = "GO" if early_gain >= .05 and migration_feasible >= .5 else (
        "HOLD" if oracle_gain >= .05 else "NO-GO")
    return {
        "label": "H3_COACTIVATION_PLACEMENT", "heldout_global_static": static_eval,
        "block_local_oracle": {
            "stage_reduction": percentiles(r["oracle_stage_reduction"] for r in block_records),
            "full_stage_mapped_reduction": percentiles(r["oracle_full_stage_mapped_reduction"] for r in block_records),
            "remote_reduction": percentiles(r["oracle_remote_reduction"] for r in block_records),
            "break_even_refinements": percentiles(r["oracle_break_even_refinements"] for r in block_records),
        },
        "early_predicted": {
            "stage_reduction": percentiles(r["early_stage_reduction"] for r in block_records),
            "full_stage_mapped_reduction": percentiles(r["early_full_stage_mapped_reduction"] for r in block_records),
            "remote_reduction": percentiles(r["early_remote_reduction"] for r in block_records),
            "break_even_refinements": percentiles(r["early_break_even_refinements"] for r in block_records),
            "migration_feasible_block_fraction": migration_feasible,
        },
        "blocks": len(block_records), "global_migration": migration_cost(global_owner),
        "coactivation_graphs": {
            "same_token_edge_weight": float(token_graph.sum() / 2),
            "same_forward_edge_weight": float(forward_graph.sum() / 2),
            "graph_cosine": graph_similarity,
            "kept_separate": True,
        },
        "verdict": verdict,
        "oracle_only": oracle_gain >= .05 and early_gain < .05,
        "ar_control": "Global static placement is generic MoE; only a held-out early-refinement signature that predicts later block co-activation is dLLM-specific.",
    }


def block_signatures(trace: DiscoveryTrace, requests: set[int]):
    a = trace.arrays
    blocks = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    block_indices = defaultdict(list)
    for i in range(len(a["request_id"])):
        request = int(a["request_id"][i])
        if request not in requests:
            continue
        key = (request, int(a["block_id"][i]))
        iteration = int(a["iteration_id"][i])
        load = a["rank_load_ep4"][i]
        blocks[key][iteration] += load
        block_indices[key].append(i)
    signatures = {}
    for key, iterations in blocks.items():
        early = [iterations[i] for i in sorted(iterations)[:2]]
        signatures[key] = np.sum(early, axis=0)
    return signatures, block_indices


def make_groups(signatures: dict, size: int, policy: str, seed: int) -> list[list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    remaining = list(signatures)
    if policy == "random":
        rng.shuffle(remaining)
        groups = []
        while len(remaining) >= size:
            group = [remaining.pop()]
            while len(group) < size:
                valid = [i for i, key in enumerate(remaining)
                         if key[0] not in {member[0] for member in group}]
                chosen = int(rng.choice(valid)) if valid else len(remaining) - 1
                group.append(remaining.pop(chosen))
            groups.append(group)
        return groups
    groups = []
    while len(remaining) >= size:
        anchor = remaining.pop(0)
        group = [anchor]
        while len(group) < size:
            aggregate = sum((signatures[key] for key in group), np.zeros(4))
            valid = [i for i, key in enumerate(remaining)
                     if key[0] not in {member[0] for member in group}]
            candidates = valid or list(range(len(remaining)))
            if policy == "similar":
                score = [cosine(aggregate, signatures[remaining[i]]) for i in candidates]
                chosen = candidates[int(np.argmax(score))]
            else:
                score = [np.max(aggregate + signatures[remaining[i]]) for i in candidates]
                chosen = candidates[int(np.argmin(score))]
            group.append(remaining.pop(chosen))
        groups.append(group)
    return groups


def group_cost(groups, block_indices, trace: DiscoveryTrace, model: CostModel) -> tuple[float, list[dict]]:
    a = trace.arrays
    total = 0.0; details = []
    for group in groups:
        by_step = defaultdict(list)
        for block in group:
            for i in block_indices[block]:
                if int(a["iteration_id"][i]) >= 2:
                    by_step[(int(a["iteration_id"][i]), int(a["layer_id"][i]))].append(i)
        group_total = 0.0
        for indices in by_step.values():
            # Preserve each full physical wave's calibrated rank-local time,
            # then compose rank critical paths. This avoids extrapolating the
            # compute model outside its measured full-row envelope.
            rank_times = np.zeros(4)
            unique = np.zeros((4, 4), dtype=np.int64)
            for i in indices:
                histogram = a["expert_counts_by_class"][i].sum(axis=0)
                rank_times += model.rank_times_from_histogram(histogram)
                unique += a["unique_matrix_ep4"][i]
            remote = unique.copy(); np.fill_diagonal(remote, 0)
            outgoing = remote.sum(axis=1) * HIDDEN * 2
            incoming = remote.sum(axis=0) * HIDDEN * 2
            dispatch = model.communication.dispatch.latency(outgoing, incoming)
            combine = model.communication.combine.latency(incoming, outgoing)
            group_total += float(dispatch + rank_times.max() + combine)
        total += group_total
        details.append({"blocks": [list(key) for key in group], "stage_ms": group_total})
    return total, details


def analyze_h4(trace: DiscoveryTrace, heldout: set[int], model: CostModel) -> tuple[dict, list[dict]]:
    signatures, block_indices = block_signatures(trace, heldout)
    a = trace.arrays
    iteration_loads = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for block, indices in block_indices.items():
        for i in indices:
            iteration_loads[block][int(a["iteration_id"][i])] += a["rank_load_ep4"][i]
    rng = np.random.default_rng(31415)
    signature_cosines = []; shuffled_cosines = []; critical_matches = []
    lag_cosines = defaultdict(list)
    block_keys = list(signatures)
    for block in block_keys:
        future = iteration_loads[block]
        other = block_keys[int(rng.integers(len(block_keys)))]
        if other == block and len(block_keys) > 1:
            other = block_keys[(block_keys.index(block) + 1) % len(block_keys)]
        for iteration, load in future.items():
            if iteration < 2:
                continue
            signature_cosines.append(cosine(signatures[block], load))
            shuffled_cosines.append(cosine(signatures[other], load))
            critical_matches.append(float(np.argmax(signatures[block]) == np.argmax(load)))
            lag_cosines[iteration].append(cosine(signatures[block], load))
    results = {}; selected_cases = []
    for size in (2, 4):
        policy_costs = {}
        policy_details = {}
        random_values = []
        random_details = None
        for seed in range(10):
            groups = make_groups(signatures, size, "random", 700 + seed)
            cost, details = group_cost(groups, block_indices, trace, model)
            random_values.append(cost)
            if seed == 0:
                random_details = details
        random_cost = float(np.median(random_values))
        for policy in ("similar", "complementary"):
            groups = make_groups(signatures, size, policy, 9)
            cost, details = group_cost(groups, block_indices, trace, model)
            policy_costs[policy] = cost; policy_details[policy] = details
        results[f"group{size}"] = {
            "random_stage_ms_median": random_cost,
            "random_stage_ms": percentiles(random_values),
            "similar_stage_ms": policy_costs["similar"],
            "complementary_stage_ms": policy_costs["complementary"],
            "complementary_reduction_vs_random": (random_cost - policy_costs["complementary"]) / random_cost,
            "similar_increase_vs_random": (policy_costs["similar"] - random_cost) / random_cost,
            "groups": len(policy_details["complementary"]),
        }
        if size == 2:
            all_details = {**policy_details, "random": random_details or []}
            for policy in ("random", "similar", "complementary"):
                ordered = sorted(all_details[policy], key=lambda row: row["stage_ms"], reverse=(policy == "similar"))
                selected_cases.extend({"policy": policy, **row} for row in ordered[:3])
    gain = results["group2"]["complementary_reduction_vs_random"]
    predictability_delta = float(np.mean(signature_cosines) - np.mean(shuffled_cosines))
    verdict = "GO" if gain >= .05 and predictability_delta > 0 else "HOLD" if gain >= .02 else "NO-GO"
    return ({
        "label": "H4_COMPLEMENTARY_BATCHING", "ep4": results, "verdict_pre_ep2": verdict,
        "signature": "first two refinements only; no future information",
        "early_signature_predictability": {
            "future_load_cosine": percentiles(signature_cosines),
            "shuffled_history_cosine": percentiles(shuffled_cosines),
            "mean_cosine_advantage": predictability_delta,
            "future_critical_rank_accuracy": float(np.mean(critical_matches)),
            "cosine_by_iteration": {str(key): float(np.mean(values)) for key, values in sorted(lag_cosines.items())},
        },
        "ar_control": "Load-aware batching is generic; only persistence of an early block signature across later refinements is dLLM-specific.",
    }, selected_cases)


def save_replay_cases(path: Path, trace: DiscoveryTrace, h1: dict, h4_cases: list[dict], heldout: set[int]):
    a = trace.arrays; labels = []; route_chunks = []
    # H1: highest-persistence held-out blocks, layer 10, adjacent refinements.
    groups = defaultdict(list)
    for key, i in key_rows(trace):
        if key[0] in heldout and key[3] == 10:
            groups[(key[0], key[1])].append((key[2], i))
    scored = []
    for block, values in groups.items():
        values.sort()
        matches = [np.argmax(a["rank_load_ep2"][x[1]]) == np.argmax(a["rank_load_ep2"][y[1]])
                   for x, y in zip(values, values[1:])]
        if matches:
            imbalance = float(np.mean([
                a["max_mean_ep2"][index] for _, index in values
            ]))
            scored.append((np.mean(matches), imbalance, block, values))
    ordered = sorted(scored)
    representatives = []
    if ordered:
        representatives.extend([
            ("low_persistence", ordered[0]),
            ("high_persistence", ordered[-1]),
            ("near_balanced", min(ordered, key=lambda row: row[1])),
            ("high_imbalance", max(ordered, key=lambda row: row[1])),
        ])
    seen = set()
    for category, (_, _imbalance, block, values) in representatives:
        if (category, block) in seen:
            continue
        seen.add((category, block))
        for iteration, index in values[:3]:
            routes = current_generated_routes(trace, index)
            route_chunks.append(routes)
            labels.append({"hypothesis": "H1", "request": block[0], "block": block[1],
                           "category": category, "iteration": iteration, "layer": 10,
                           "structural_critical_rank_ep2": int(np.argmax(np.bincount(
                               (routes // 128).reshape(-1), minlength=2)))})
    # H4: representative pair at iteration 2/layer 10, exact current-block routes.
    lookup = {(key[0], key[1], key[2], key[3]): i for key, i in key_rows(trace)}
    for case in h4_cases:
        chunks = []
        for request, block in case["blocks"]:
            index = lookup.get((request, block, 2, 10))
            if index is not None:
                chunks.append(current_generated_routes(trace, index))
        if len(chunks) == 2:
            routes = np.concatenate(chunks)
            route_chunks.append(routes)
            loads = np.bincount((routes // 128).reshape(-1), minlength=2)
            labels.append({"hypothesis": "H4", "policy": case["policy"],
                           "blocks": case["blocks"], "iteration": 2, "layer": 10,
                           "structural_critical_rank_ep2": int(np.argmax(loads))})
    offsets = [0]; flat = []
    for routes in route_chunks:
        flat.append(routes); offsets.append(offsets[-1] + len(routes))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, expert_ids=np.concatenate(flat), offsets=np.asarray(offsets),
                        labels_json=np.asarray(json.dumps(labels), dtype=np.str_))


def figures(trace: DiscoveryTrace, h1, h2, h3, h4, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("h1", "h2", "h3", "h4"):
        (root / name).mkdir(exist_ok=True)
    # H1 target comparison.
    eps = [2, 4, 8]
    adjacent = [h1["targets"][f"ep{x}"]["layer_local_adjacent_critical_rank_match"] for x in eps]
    random = [h1["targets"][f"ep{x}"]["matched_random_critical_rank_match"] for x in eps]
    plt.figure(figsize=(5, 3)); x = np.arange(3)
    plt.bar(x-.18, adjacent, .36, label="same block adjacent"); plt.bar(x+.18, random, .36, label="matched random")
    plt.xticks(x, [f"EP{e}" for e in eps]); plt.ylabel("critical-rank match"); plt.legend(); plt.tight_layout()
    plt.savefig(root / "h1" / "critical_rank_persistence.png", dpi=160); plt.close()
    # H2 state load.
    names = ["CURRENT_BLOCK_MASKED", "CURRENT_BLOCK_NEWLY_ACCEPTED", "CURRENT_BLOCK_DECODED"]
    vals = [h2["ep4_state_load"][name]["max_mean"]["p50"] for name in names]
    plt.figure(figsize=(6, 3)); plt.bar(range(3), vals); plt.xticks(range(3), ["masked", "newly accepted", "decoded"])
    plt.ylabel("EP4 max/mean (P50)"); plt.tight_layout(); plt.savefig(root / "h2" / "state_rank_load.png", dpi=160); plt.close()
    # H3 headroom.
    vals = [h3["heldout_global_static"]["full_physical_stage_mapped_reduction_fraction"],
            h3["early_predicted"]["full_stage_mapped_reduction"]["p50"],
            h3["block_local_oracle"]["full_stage_mapped_reduction"]["p50"]]
    plt.figure(figsize=(6, 3)); plt.bar(range(3), np.asarray(vals)*100); plt.xticks(range(3), ["global static", "early predicted", "full oracle"])
    plt.ylabel("full physical EP4 stage reduction (%)"); plt.tight_layout(); plt.savefig(root / "h3" / "placement_headroom.png", dpi=160); plt.close()
    # H4 policy.
    row = h4["ep4"]["group2"]; baseline = row["random_stage_ms_median"]
    vals = [baseline, row["similar_stage_ms"], row["complementary_stage_ms"]]
    plt.figure(figsize=(5, 3)); plt.bar(range(3), np.asarray(vals)/baseline); plt.xticks(range(3), ["random", "similar", "complementary"])
    plt.ylabel("normalized EP4 stage"); plt.tight_layout(); plt.savefig(root / "h4" / "group2_policy.png", dpi=160); plt.close()


def common_ep4_tails(trace: DiscoveryTrace, heldout: set[int], model: CostModel) -> dict:
    a = trace.arrays
    values = defaultdict(list)
    for i in range(len(a["request_id"])):
        if int(a["request_id"][i]) not in heldout:
            continue
        histogram = a["expert_counts_by_class"][i].sum(axis=0)
        rank_times = []
        for rank in range(4):
            latency, _, _ = model.compute.predict(histogram[rank * 64:(rank + 1) * 64])
            rank_times.append(latency)
        outgoing = a["outgoing_bytes_ep4"][i]
        incoming = a["incoming_bytes_ep4"][i]
        dispatch = model.communication.dispatch.latency(outgoing, incoming)
        combine = model.communication.combine.latency(incoming, outgoing)
        values["max_mean_rank_load"].append(float(a["max_mean_ep4"][i]))
        values["critical_rank_compute_ms"].append(float(max(rank_times)))
        values["remote_bytes"].append(float(outgoing.sum()))
        values["fanout"].append(float(a["mean_fanout_ep4"][i]))
        values["simulated_moe_stage_ms"].append(float(dispatch + max(rank_times) + combine))
    return {name: percentiles(rows) for name, rows in values.items()}


def apply_ep2_validation(summaries: dict, path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"status": "PENDING", "reason": "selected-state true EP2 replay not supplied"}
    document = json.loads(path.read_text())
    h1_cases = [row for row in document["cases"] if row["label"]["hypothesis"] == "H1"]
    h4_cases = [row for row in document["cases"] if row["label"]["hypothesis"] == "H4"]
    h1_match = float(np.mean([row["structural_rank_match"] for row in h1_cases])) if h1_cases else float("nan")
    h1_groups = defaultdict(list)
    for row in h1_cases:
        label = row["label"]
        h1_groups[(label["category"], label["request"], label["block"])].append(
            (label["iteration"], row["measured_slow_rank"])
        )
    measured_persistence = defaultdict(list)
    for (category, _request, _block), rows in h1_groups.items():
        rows.sort()
        measured_persistence[category].extend(
            float(left[1] == right[1]) for left, right in zip(rows, rows[1:])
        )
    measured_persistence = {
        key: float(np.mean(rows)) for key, rows in measured_persistence.items() if rows
    }
    persistence_contrast = (
        measured_persistence.get("high_persistence", float("nan")) -
        measured_persistence.get("low_persistence", float("nan"))
    )
    policy = defaultdict(list)
    for row in h4_cases:
        stage = row["critical_dispatch_ms"] + row["critical_expert_ms"] + row["critical_combine_ms"]
        policy[row["label"]["policy"]].append(stage)
    policy_mean = {name: float(np.mean(rows)) for name, rows in policy.items()}
    h4_direction = ("complementary" in policy_mean and "random" in policy_mean and
                    policy_mean["complementary"] < policy_mean["random"])
    summaries["h1"]["true_ep2_support"] = {
        "cases": len(h1_cases), "structural_critical_rank_match_rate": h1_match,
        "measured_adjacent_persistence_by_category": measured_persistence,
        "high_minus_low_measured_persistence": persistence_contrast,
        "direction_confirmed": bool(h1_match >= .6 and persistence_contrast >= .05),
    }
    if summaries["h1"]["verdict_pre_ep2"] == "GO":
        summaries["h1"]["verdict"] = "GO" if summaries["h1"]["true_ep2_support"]["direction_confirmed"] else "HOLD"
    else:
        summaries["h1"]["verdict"] = summaries["h1"]["verdict_pre_ep2"]
    summaries["h4"]["true_ep2_support"] = {
        "cases": len(h4_cases), "policy_stage_ms": policy_mean,
        "complementary_better_than_random": bool(h4_direction),
    }
    if summaries["h4"]["verdict_pre_ep2"] == "GO":
        summaries["h4"]["verdict"] = "GO" if h4_direction else "HOLD"
    else:
        summaries["h4"]["verdict"] = summaries["h4"]["verdict_pre_ep2"]
    return {"status": "COMPLETE", "source": str(path), "h1_match": h1_match,
            "h4_policy_stage_ms": policy_mean, "h4_direction": bool(h4_direction)}


def markdown_reports(report_dir: Path, summaries: dict, cohort: dict, common_tails: dict,
                     ep2_validation: dict):
    report_dir.mkdir(parents=True, exist_ok=True)
    h1, h2, h3, h4 = (summaries[f"h{i}"] for i in range(1, 5))
    common = (f"Cohort: GSM8K {cohort['requests']} requests at threshold 0.95; "
              f"discovery/train={cohort['train']}, held-out={cohort['heldout']}. "
              "Statistical units are requests/blocks, not token-layer rows. EP4/EP8 are EP2-calibrated simulations.\n\n")
    documents = {
        "discovery_h1_block_persistent_straggler.md": "# H1 — Block-Persistent EP Straggler\n\n" + common +
            f"EP4 adjacent critical-rank match: {h1['targets']['ep4']['layer_local_adjacent_critical_rank_match']:.3f}; matched random: {h1['targets']['ep4']['matched_random_critical_rank_match']:.3f}.\n\n"
            f"Early-two future accuracy: {h1['targets']['ep4']['early2_future_critical_rank_accuracy']:.3f}. Verdict: **{h1.get('verdict', h1['verdict_pre_ep2'])}**.\n\nAR control: {h1['ar_control']}\n",
        "discovery_h2_mask_state_specialization.md": "# H2 — Mask-State Expert Specialization\n\n" + common +
            f"Masked↔decoded expert JSD P50: {h2['distribution_metrics']['CURRENT_BLOCK_MASKED__CURRENT_BLOCK_DECODED']['jsd']['p50']:.4f}. "
            f"Isolated-state relative EP4 max/mean difference: {100*h2['median_relative_ep4_max_mean_gap']:.2f}%; "
            f"actual full-workload composition effect P50: {100*h2['actual_full_workload_ep4_max_mean_change_from_state_composition']['p50']:.2f}%. "
            f"Verdict: **{h2['verdict']}**.\n\nAR control: {h2['ar_control']}\n",
        "discovery_h3_coactivation_placement.md": "# H3 — Co-activation and Placement\n\n" + common +
            f"Held-out global-static full-stage-mapped reduction: {100*h3['heldout_global_static']['full_physical_stage_mapped_reduction_fraction']:.2f}%. "
            f"Early-predicted full-stage P50: {100*h3['early_predicted']['full_stage_mapped_reduction']['p50']:.2f}%; "
            f"full-future oracle P50: {100*h3['block_local_oracle']['full_stage_mapped_reduction']['p50']:.2f}%. "
            f"Verdict: **{h3['verdict']}**.\n\nWeight migration is included in break-even analysis. AR control: {h3['ar_control']}\n",
        "discovery_h4_complementary_batching.md": "# H4 — Complementary EP Batching\n\n" + common +
            f"Held-out EP4 pair-size-2 complementary reduction versus random: {100*h4['ep4']['group2']['complementary_reduction_vs_random']:.2f}%. "
            f"Similar-hot change versus random: {100*h4['ep4']['group2']['similar_increase_vs_random']:.2f}%. Verdict: **{h4.get('verdict', h4['verdict_pre_ep2'])}**.\n\nAR control: {h4['ar_control']}\n",
    }
    for name, text in documents.items():
        (report_dir / name).write_text(text)
    verdicts = {"h1": h1.get("verdict", h1["verdict_pre_ep2"]), "h2": h2["verdict"],
                "h3": h3["verdict"], "h4": h4.get("verdict", h4["verdict_pre_ep2"])}
    table = f"""# Four dLLM–MoE–EP Hypotheses: Discovery Summary

{common}
## Evidence boundary

- Existing trace had proved average structural imbalance and calibrated true-EP2 component timing only.
- EP4/EP8 remain `SIMULATED-EP4/8-EP2-CALIBRATED`; no physical EP4/EP8 claim is made.
- Persistent critical-rank behavior was **not** established before this PoC.
- Replicated-state bridge all-gather is excluded from every EP cost and speedup.

## Summary

| Hypothesis | dLLM-specific signal | EP-specific signal | True EP2 support | Virtual EP4 headroom | Verdict |
|---|---|---|---|---|---|
| H1 Block-persistent straggler | same-block vs cross/random persistence | critical-rank/load persistence | {h1.get('true_ep2_support', {}).get('direction_confirmed', 'pending')} | early2 accuracy {h1['targets']['ep4']['early2_future_critical_rank_accuracy']:.3f} | {verdicts['h1']} |
| H2 Mask-state specialization | same-position MASK→decoded transition | full-load composition effect {100*h2['actual_full_workload_ep4_max_mean_change_from_state_composition']['p50']:.2f}% | structural only | JSD P50 {h2['distribution_metrics']['CURRENT_BLOCK_MASKED__CURRENT_BLOCK_DECODED']['jsd']['p50']:.4f} | {verdicts['h2']} |
| H3 Co-activation placement | early refinements predict block graph | remote/fanout/stage placement | not required for oracle gate | full-stage early P50 {100*h3['early_predicted']['full_stage_mapped_reduction']['p50']:.2f}% | {verdicts['h3']} |
| H4 Complementary batching | early signature reused over later refinements | combined critical EP rank | {h4.get('true_ep2_support', {}).get('complementary_better_than_random', 'pending')} | pair gain {100*h4['ep4']['group2']['complementary_reduction_vs_random']:.2f}% | {verdicts['h4']} |

## Common EP4 tails

```json
{json.dumps(common_tails, indent=2)}
```

## Physical validation

```json
{json.dumps(ep2_validation, indent=2)}
```

The reports rank candidates by measured/simulated headroom, but no method is implemented automatically.
"""
    (report_dir / "discovery_four_hypotheses_summary.md").write_text(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures/discovery_ep"))
    parser.add_argument("--communication-model", type=Path, required=True)
    parser.add_argument("--compute-ep2", type=Path, required=True)
    parser.add_argument("--compute-ep4", type=Path, required=True)
    parser.add_argument("--compute-ep8", type=Path, required=True)
    parser.add_argument("--replay-cases", type=Path, required=True)
    parser.add_argument("--ep2-validation", type=Path)
    parser.add_argument("--stage0", action="store_true")
    args = parser.parse_args()
    trace = DiscoveryTrace.load(args.trace)
    requests = sorted(set(map(int, trace.arrays["request_id"])))
    if args.stage0:
        midpoint = max(1, len(requests) // 2)
        train, heldout = set(requests[:midpoint]), set(requests[midpoint:])
    else:
        train, heldout = set(range(64)), set(range(64, 128))
        missing = (train | heldout) - set(requests)
        if missing:
            raise ValueError(f"discovery cohort missing request IDs: {sorted(missing)[:10]}")
    models = {
        2: CostModel(2, args.compute_ep2, args.communication_model),
        4: CostModel(4, args.compute_ep4, args.communication_model),
        8: CostModel(8, args.compute_ep8, args.communication_model),
    }
    model = models[4]
    h1 = analyze_h1(trace, train, heldout, models)
    h2 = analyze_h2(trace, heldout)
    h3 = analyze_h3(trace, train, heldout, model, max_oracle_blocks=8 if args.stage0 else None)
    h4, selected = analyze_h4(trace, heldout, model)
    summaries = {"h1": h1, "h2": h2, "h3": h3, "h4": h4}
    cohort = {"requests": len(requests), "train": len(train), "heldout": len(heldout),
              "trace": str(args.trace), "stage0_smoke_only": args.stage0}
    save_replay_cases(args.replay_cases, trace, h1, selected, heldout)
    ep2_validation = apply_ep2_validation(summaries, args.ep2_validation)
    tails = common_ep4_tails(trace, heldout, model)
    figures(trace, h1, h2, h3, h4, args.figures)
    markdown_reports(args.report_dir, summaries, cohort, tails, ep2_validation)
    for number in range(1, 5):
        (args.report_dir / f"discovery_h{number}_summary.json").write_text(
            json.dumps(summaries[f"h{number}"], indent=2, allow_nan=True) + "\n")
    overall = {"cohort": cohort, "hypotheses": summaries,
               "common_ep4_tails": tails, "true_ep2_selected_validation": ep2_validation,
               "accounting": "routed-MoE only; replicated-state bridge excluded",
               "ep4_label": "SIMULATED-EP4-EP2-CALIBRATED",
               "ep8_label": "SIMULATED-EP8-EP2-CALIBRATED"}
    (args.report_dir / "discovery_four_hypotheses_summary.json").write_text(
        json.dumps(overall, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"cohort": cohort, "verdicts": {key: value.get("verdict", value.get("verdict_pre_ep2")) for key, value in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
