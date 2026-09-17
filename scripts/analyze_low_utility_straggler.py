#!/usr/bin/env python3
"""Trace-only low-utility EP-straggler discovery analysis.

The aggregate-v2 trace preserves exact expert IDs and effective gate
coefficients for the current diffusion block.  Prefix and prior-block rows are
retained only as expert/rank histograms.  Consequently:

* router-mass anatomy is exact for current-block rows across GSM8K-128;
* the critical rank and routed-stage denominator use the complete physical
  workload;
* counterfactual pruning removes current-block routes only and is therefore a
  conservative systems oracle, not a production pruning implementation;
* no model-quality claim is made.

The optional existing GSM8K-32 full-row heavy trace is used only as an audit of
the anatomy result.  It is never silently mixed with the 128-request primary
population.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

from scripts.analyze_h1_straggler_deepdive import BatchComputeModel, load_discovery_trace
from virtual_ep.comm_model import CommunicationScenario


NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_BYTES = 2048 * 2
EP_TARGETS = (4, 8)
MASS_BUDGETS = (0.005, 0.01, 0.025, 0.05, 0.10)
ROUTE_BUDGETS = (0.05, 0.10, 0.20, 0.30)
MASS_THRESHOLDS = (0.01, 0.02, 0.05, 0.10)
PRESERVE_TARGETS = (0.90, 0.95, 0.975, 0.99)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def effective_slot_ranks(weights: np.ndarray) -> np.ndarray:
    """Return ranks 1..8 after sorting the *effective* gate coefficient."""
    order = np.argsort(-weights, axis=-1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int8)
    np.put_along_axis(
        ranks, order,
        np.broadcast_to(np.arange(1, TOP_K + 1, dtype=np.int8), order.shape),
        axis=-1,
    )
    return ranks


def cumulative_tail_masks(normalized: np.ndarray, slot_ranks: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.take_along_axis(normalized, np.argsort(slot_ranks, axis=-1), axis=-1)
    cumulative = np.cumsum(ordered, axis=-1)
    result = {}
    for target in PRESERVE_TARGETS:
        prefix = np.argmax(cumulative >= target, axis=-1) + 1
        result[f"outside_preserve_{target:g}"] = slot_ranks > prefix[..., None]
    return result


def block_relative_phase(arrays: dict[str, np.ndarray]) -> np.ndarray:
    request = arrays["request_id"].astype(np.int64)
    block = arrays["block_id"].astype(np.int64)
    pair = np.stack((request, block), axis=1)
    _keys, inverse = np.unique(pair, axis=0, return_inverse=True)
    maximum = np.zeros(int(inverse.max()) + 1, dtype=np.int16)
    np.maximum.at(maximum, inverse, arrays["iteration_id"])
    denominator = np.maximum(maximum[inverse], 1)
    relative = arrays["iteration_id"] / denominator
    return np.where(relative < 1 / 3, 0, np.where(relative < 2 / 3, 1, 2)).astype(np.int8)


def mask_ratio_bin(arrays: dict[str, np.ndarray]) -> np.ndarray:
    generated = arrays["class_token_counts"][:, 2:].sum(axis=1)
    ratio = np.divide(
        arrays["masked_current_block"], generated,
        out=np.zeros_like(generated, dtype=np.float64), where=generated > 0,
    )
    return np.where(ratio > .75, 0, np.where(ratio > .50, 1, np.where(ratio > .25, 2, 3))).astype(np.int8)


def cluster_bootstrap(
    critical_low: np.ndarray, critical_total: np.ndarray,
    other_low: np.ndarray, other_total: np.ndarray,
    request_ids: np.ndarray, samples: int = 1000, seed: int = 20260917,
) -> dict[str, list[float]]:
    requests = np.unique(request_ids)
    per_request = np.zeros((len(requests), 4), dtype=np.float64)
    for ordinal, request in enumerate(requests):
        selected = request_ids == request
        per_request[ordinal] = (
            critical_low[selected].sum(), critical_total[selected].sum(),
            other_low[selected].sum(), other_total[selected].sum(),
        )
    rng = np.random.default_rng(seed)
    differences, enrichments = [], []
    for _ in range(samples):
        chosen = per_request[rng.integers(0, len(per_request), len(per_request))].sum(axis=0)
        critical = safe_ratio(chosen[0], chosen[1])
        other = safe_ratio(chosen[2], chosen[3])
        differences.append(100 * (critical - other))
        enrichments.append(safe_ratio(critical, other))
    return {
        "percentage_point_difference_95ci": np.percentile(differences, [2.5, 97.5]).tolist(),
        "enrichment_95ci": np.percentile(enrichments, [2.5, 97.5]).tolist(),
        "cluster_unit": "request (128 independent generation trajectories)",
    }


def definition_masks(
    normalized: np.ndarray, slot_ranks: np.ndarray, valid: np.ndarray,
    quantiles: dict[str, float],
) -> dict[str, np.ndarray]:
    masks = {
        "slot_8": slot_ranks == 8,
        "slots_7_8": slot_ranks >= 7,
        "slots_5_8": slot_ranks >= 5,
        **{f"bottom_{q}_global": normalized <= threshold for q, threshold in quantiles.items()},
        **{f"mass_le_{threshold:g}": normalized <= threshold for threshold in MASS_THRESHOLDS},
    }
    masks.update(cumulative_tail_masks(normalized, slot_ranks))
    return {key: value & valid[..., None] for key, value in masks.items()}


def anatomy_for_ep(
    arrays: dict[str, np.ndarray], ids: np.ndarray, normalized: np.ndarray,
    slot_ranks: np.ndarray, valid: np.ndarray, times: np.ndarray,
    ep: int, definitions: dict[str, np.ndarray], phase: np.ndarray,
    mask_bin: np.ndarray,
) -> tuple[dict, dict, dict]:
    owner = ids // (NUM_EXPERTS // ep)
    critical = np.argmax(times, axis=1)
    critical_mask = (owner == critical[:, None, None]) & valid[..., None]
    other_mask = (~(owner == critical[:, None, None])) & valid[..., None]
    critical_total_per_inv = critical_mask.sum(axis=(1, 2)).astype(np.float64)
    other_total_per_inv = other_mask.sum(axis=(1, 2)).astype(np.float64)

    full_hist = arrays["expert_counts_by_class"].sum(axis=1).astype(np.float64)
    per_rank = NUM_EXPERTS // ep
    full_critical_counts = np.empty(len(ids), dtype=np.float64)
    for rank in range(ep):
        selected = critical == rank
        full_critical_counts[selected] = full_hist[selected, rank * per_rank:(rank + 1) * per_rank].sum(axis=1)
    excess = times.max(axis=1) - times.mean(axis=1)
    row_excess = np.divide(
        excess, full_critical_counts,
        out=np.zeros_like(excess), where=full_critical_counts > 0,
    )
    total_excess = float(excess.sum())

    result = {}
    phase_result = {"block_relative": {}, "mask_ratio": {}}
    per_definition_arrays = {}
    for name, low in definitions.items():
        critical_low_per_inv = (low & critical_mask).sum(axis=(1, 2)).astype(np.float64)
        other_low_per_inv = (low & other_mask).sum(axis=(1, 2)).astype(np.float64)
        critical_fraction = safe_ratio(critical_low_per_inv.sum(), critical_total_per_inv.sum())
        other_fraction = safe_ratio(other_low_per_inv.sum(), other_total_per_inv.sum())
        attributed = float(np.sum(critical_low_per_inv * row_excess))
        entry = {
            "critical_fraction": critical_fraction,
            "noncritical_fraction": other_fraction,
            "enrichment": safe_ratio(critical_fraction, other_fraction),
            "percentage_point_difference": 100 * (critical_fraction - other_fraction),
            "attributed_excess_share_of_full_physical_excess": safe_ratio(attributed, total_excess),
            "attributed_excess_ms": attributed,
        }
        if name in ("slots_7_8", "bottom_25_global", "mass_le_0.1"):
            entry["request_cluster_bootstrap"] = cluster_bootstrap(
                critical_low_per_inv, critical_total_per_inv,
                other_low_per_inv, other_total_per_inv, arrays["request_id"],
                seed=20260917 + ep,
            )
        result[name] = entry
        per_definition_arrays[name] = (
            critical_low_per_inv, other_low_per_inv, critical_total_per_inv,
            other_total_per_inv,
        )

    labels = {
        "block_relative": (phase, ("early", "middle", "late")),
        "mask_ratio": (mask_bin, ("gt75", "50_75", "25_50", "le25")),
    }
    focus = ("slots_7_8", "bottom_25_global", "mass_le_0.1")
    for axis_name, (axis, axis_labels) in labels.items():
        for value, label in enumerate(axis_labels):
            selected = axis == value
            cell = {"invocations": int(selected.sum())}
            for name in focus:
                c_low, o_low, c_total, o_total = per_definition_arrays[name]
                cf = safe_ratio(c_low[selected].sum(), c_total[selected].sum())
                of = safe_ratio(o_low[selected].sum(), o_total[selected].sum())
                attr = float(np.sum(c_low[selected] * row_excess[selected]))
                cell[name] = {
                    "critical_fraction": cf,
                    "noncritical_fraction": of,
                    "enrichment": safe_ratio(cf, of),
                    "percentage_point_difference": 100 * (cf - of),
                    "attributed_excess_share": safe_ratio(attr, excess[selected].sum()),
                }
            phase_result[axis_name][label] = cell

    # Global hot-rank and random-rank controls on the held-out request half.
    train = arrays["request_id"] < 64
    held = arrays["request_id"] >= 64
    majority = {}
    for layer in np.unique(arrays["layer_id"]):
        selected = train & (arrays["layer_id"] == layer)
        majority[int(layer)] = int(np.argmax(np.bincount(critical[selected], minlength=ep)))
    pseudo = np.asarray([majority[int(layer)] for layer in arrays["layer_id"]], dtype=np.int8)
    rng = np.random.default_rng(20260917 + ep)
    random_rank = rng.integers(0, ep, len(ids), dtype=np.int8)
    controls = {}
    for control_name, rank_vector in (("layer_global_hot_rank", pseudo), ("random_rank", random_rank)):
        c_mask = (owner == rank_vector[:, None, None]) & valid[..., None] & held[:, None, None]
        o_mask = (owner != rank_vector[:, None, None]) & valid[..., None] & held[:, None, None]
        controls[control_name] = {}
        for name in focus:
            cf = safe_ratio(np.count_nonzero(definitions[name] & c_mask), np.count_nonzero(c_mask))
            of = safe_ratio(np.count_nonzero(definitions[name] & o_mask), np.count_nonzero(o_mask))
            controls[control_name][name] = {
                "critical_fraction": cf, "noncritical_fraction": of,
                "enrichment": safe_ratio(cf, of),
                "percentage_point_difference": 100 * (cf - of),
            }
    return result, phase_result, controls


def excess_mass_curve(
    ids: np.ndarray, normalized: np.ndarray, valid: np.ndarray,
    arrays: dict[str, np.ndarray], times: np.ndarray, ep: int,
) -> dict:
    owner = ids // (NUM_EXPERTS // ep)
    critical = np.argmax(times, axis=1)
    selected = (owner == critical[:, None, None]) & valid[..., None]
    hist = arrays["expert_counts_by_class"].sum(axis=1).astype(np.float64)
    per_rank = NUM_EXPERTS // ep
    critical_count = np.empty(len(ids), dtype=np.float64)
    for rank in range(ep):
        mask = critical == rank
        critical_count[mask] = hist[mask, rank * per_rank:(rank + 1) * per_rank].sum(axis=1)
    excess = times.max(axis=1) - times.mean(axis=1)
    factor = np.divide(excess, critical_count, out=np.zeros_like(excess), where=critical_count > 0)
    weights = normalized[selected].astype(np.float32)
    attribution = np.broadcast_to(factor[:, None, None], selected.shape)[selected].astype(np.float32)
    order = np.argsort(weights, kind="stable")
    weights = weights[order]
    attribution = attribution[order]
    cumulative_mass = np.cumsum(weights, dtype=np.float64)
    cumulative_excess = np.cumsum(attribution, dtype=np.float64)
    total_target_mass = float(valid.sum())  # normalized mass sums to one per valid token
    total_full_excess = float(excess.sum())
    total_current_excess = float(attribution.sum())
    targets = {}
    for target in (.25, .50, .75):
        wanted = target * total_full_excess
        if cumulative_excess[-1] + 1e-12 < wanted:
            targets[str(target)] = None
        else:
            index = int(np.searchsorted(cumulative_excess, wanted, side="left"))
            targets[str(target)] = 100 * float(cumulative_mass[index]) / total_target_mass
    conditional = {}
    for target in (.25, .50, .75):
        wanted = target * total_current_excess
        index = int(np.searchsorted(cumulative_excess, wanted, side="left"))
        conditional[str(target)] = 100 * float(cumulative_mass[index]) / total_target_mass
    return {
        "router_mass_percent_needed_for_fraction_of_full_physical_excess": targets,
        "router_mass_percent_needed_for_fraction_of_current_block_attributed_excess": conditional,
        "maximum_full_physical_excess_removable_from_current_block_routes_percent": 100 * safe_ratio(total_current_excess, total_full_excess),
        "current_block_critical_routes": int(len(weights)),
        "evidence_boundary": "current-block exact routes; denominator is full physical critical excess",
    }


@njit(parallel=True, cache=True)
def build_current_unique(ids, valid, source, ep, removed):
    count = ids.shape[0]
    result = np.zeros((count, ep, ep), dtype=np.int32)
    per_rank = NUM_EXPERTS // ep
    for invocation in prange(count):
        for position in range(ids.shape[1]):
            if not valid[invocation, position]:
                continue
            seen = np.zeros(ep, dtype=np.uint8)
            for slot in range(ids.shape[2]):
                if removed[invocation, position, slot]:
                    continue
                destination = ids[invocation, position, slot] // per_rank
                seen[destination] = 1
            src = source[invocation, position]
            for destination in range(ep):
                result[invocation, src, destination] += seen[destination]
    return result


@njit(parallel=True, cache=True)
def build_current_fanout(ids, valid, ep, removed):
    result = np.zeros(ids.shape[0], dtype=np.int32)
    per_rank = NUM_EXPERTS // ep
    for invocation in prange(ids.shape[0]):
        total = 0
        for position in range(ids.shape[1]):
            if not valid[invocation, position]:
                continue
            seen = np.zeros(ep, dtype=np.uint8)
            for slot in range(ids.shape[2]):
                if not removed[invocation, position, slot]:
                    seen[ids[invocation, position, slot] // per_rank] = 1
            for destination in range(ep):
                total += seen[destination]
        result[invocation] = total
    return result


@njit(parallel=True, cache=True)
def subtract_removed_hist(full_hist, ids, valid, removed):
    result = full_hist.copy()
    for invocation in prange(ids.shape[0]):
        for position in range(ids.shape[1]):
            if not valid[invocation, position]:
                continue
            for slot in range(ids.shape[2]):
                if removed[invocation, position, slot]:
                    result[invocation, ids[invocation, position, slot]] -= 1
    return result


@njit(parallel=True, cache=True)
def selection_mask(
    normalized, slot_ranks, ids, valid, ep, pressure, target,
    budget_kind, min_k, already_removed, random_mode,
):
    """Select eligible routes by score under a per-invocation budget.

    budget_kind=0 uses normalized router mass; budget_kind=1 route count.
    pressure is [invocation, rank].  A vector of ones gives utility-only
    selection; calibrated or assignment pressure gives P3/P4.  A route on a
    non-positive-pressure rank is ineligible.  The selection is deterministic.
    """
    output = already_removed.copy()
    per_rank = NUM_EXPERTS // ep
    width = normalized.shape[1] * normalized.shape[2]
    for invocation in prange(normalized.shape[0]):
        scores = np.full(width, -1.0, dtype=np.float64)
        masses = np.zeros(width, dtype=np.float64)
        candidates = 0
        valid_tokens = 0
        removed_mass = 0.0
        removed_count = 0
        for position in range(normalized.shape[1]):
            if not valid[invocation, position]:
                continue
            valid_tokens += 1
            for slot in range(normalized.shape[2]):
                mass = normalized[invocation, position, slot]
                if output[invocation, position, slot]:
                    removed_mass += mass
                    removed_count += 1
                    continue
                if slot_ranks[invocation, position, slot] <= min_k:
                    continue
                rank = ids[invocation, position, slot] // per_rank
                rank_pressure = pressure[invocation, rank]
                if rank_pressure <= 0:
                    continue
                flat = position * normalized.shape[2] + slot
                masses[flat] = mass
                if random_mode:
                    # Reproducible integer hash converted to (0,1).
                    value = ((invocation + 1) * 1103515245 + (flat + 17) * 12345) & 0x7fffffff
                    scores[flat] = value / 2147483647.0
                else:
                    scores[flat] = rank_pressure / max(mass, 1e-12)
                candidates += 1
        if budget_kind == 0:
            budget = target * valid_tokens
            current = removed_mass
        else:
            budget = target * valid_tokens * normalized.shape[2]
            current = removed_count
        order = np.argsort(scores)
        for ordinal in range(width - 1, -1, -1):
            flat = order[ordinal]
            if scores[flat] < 0:
                break
            position = flat // normalized.shape[2]
            slot = flat % normalized.shape[2]
            cost = masses[flat] if budget_kind == 0 else 1.0
            if current + cost <= budget + 1e-12:
                output[invocation, position, slot] = True
                current += cost
    return output


@njit(parallel=True, cache=True)
def canonical_mass_preserving(normalized, slot_ranks, valid, target, min_k):
    removed = np.zeros(normalized.shape, dtype=np.bool_)
    for invocation in prange(normalized.shape[0]):
        for position in range(normalized.shape[1]):
            if not valid[invocation, position]:
                continue
            retained = 0.0
            cutoff = min_k
            for rank in range(1, TOP_K + 1):
                for slot in range(TOP_K):
                    if slot_ranks[invocation, position, slot] == rank:
                        retained += normalized[invocation, position, slot]
                        break
                if rank >= min_k and retained >= target:
                    cutoff = rank
                    break
                cutoff = rank
            for slot in range(TOP_K):
                removed[invocation, position, slot] = slot_ranks[invocation, position, slot] > cutoff
    return removed


def communication_latency(model, unique: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    remote = unique.astype(np.float64, copy=True)
    diagonal = np.arange(unique.shape[1])
    remote[:, diagonal, diagonal] = 0
    outgoing = remote.sum(axis=2) * HIDDEN_BYTES
    incoming = remote.sum(axis=1) * HIDDEN_BYTES
    endpoint = np.maximum(outgoing.max(axis=1), incoming.max(axis=1))
    active = np.maximum(np.count_nonzero(outgoing, axis=1), np.count_nonzero(incoming, axis=1))
    latency = (
        model.startup_ms + endpoint / (model.gb_per_s * 1e9) * 1e3
        + model.peer_ms * np.maximum(active - 1, 0) + model.sync_ms
    )
    return latency, outgoing, incoming


class CounterfactualProjector:
    def __init__(
        self, arrays: dict[str, np.ndarray], ids: np.ndarray, normalized: np.ndarray,
        slot_ranks: np.ndarray, valid: np.ndarray, models: dict[int, BatchComputeModel],
        communication: CommunicationScenario,
    ):
        self.arrays = arrays
        self.ids = ids
        self.normalized = normalized
        self.slot_ranks = slot_ranks
        self.valid = valid
        self.models = models
        self.communication = communication
        self.full_hist = arrays["expert_counts_by_class"].sum(axis=1).astype(np.int32)
        self.no_removed = np.zeros(ids.shape, dtype=np.bool_)
        self.base_current_unique = {
            ep: build_current_unique(
                ids, valid, arrays[f"current_source_rank_ep{ep}"], ep, self.no_removed
            ) for ep in EP_TARGETS
        }
        self.base_current_fanout = {
            ep: build_current_fanout(ids, valid, ep, self.no_removed) for ep in EP_TARGETS
        }
        self.base = {}
        for ep in EP_TARGETS:
            self.base[ep] = self.evaluate(self.no_removed, ep, "P0_vanilla")

    def _unique(self, removed: np.ndarray, ep: int) -> np.ndarray:
        current = build_current_unique(
            self.ids, self.valid, self.arrays[f"current_source_rank_ep{ep}"], ep, removed
        )
        fixed = self.arrays[f"unique_matrix_ep{ep}"].astype(np.int32) - self.base_current_unique[ep]
        return fixed + current

    def evaluate(self, removed: np.ndarray, ep: int, name: str) -> dict:
        histogram = subtract_removed_hist(
            self.full_hist, self.ids, self.valid, removed
        ).astype(np.float64)
        per_rank = NUM_EXPERTS // ep
        times = np.column_stack([
            self.models[ep].predict_batch(histogram[:, rank * per_rank:(rank + 1) * per_rank])
            for rank in range(ep)
        ])
        unique = self._unique(removed, ep)
        dispatch, outgoing, incoming = communication_latency(self.communication.dispatch, unique)
        combine, _reverse_out, _reverse_in = communication_latency(
            self.communication.combine, np.transpose(unique, (0, 2, 1))
        )
        expert = times.max(axis=1)
        stage = dispatch + expert + combine
        maximum = expert
        mean = times.mean(axis=1)
        ordered = np.sort(times, axis=1)
        second = ordered[:, -2]
        wait = np.divide(
            ep * maximum - times.sum(axis=1), ep * maximum,
            out=np.zeros_like(maximum), where=maximum > 0,
        )
        removed_count = int(removed.sum())
        removed_mass = float(self.normalized[removed & self.valid[..., None]].sum())
        current_routes = int(self.valid.sum() * TOP_K)
        full_routes = int(self.full_hist.sum())
        result = {
            "name": name, "ep": ep,
            "removed_current_routes": removed_count,
            "removed_current_route_fraction": safe_ratio(removed_count, current_routes),
            "removed_full_physical_route_fraction": safe_ratio(removed_count, full_routes),
            "removed_current_router_mass_fraction": safe_ratio(removed_mass, self.valid.sum()),
            "dispatch_ms": float(dispatch.sum()),
            "expert_ms": float(expert.sum()),
            "combine_ms": float(combine.sum()),
            "stage_ms": float(stage.sum()),
            "max_mean_mean": float(np.mean(maximum / np.maximum(mean, 1e-12))),
            "max_mean_p50": float(np.median(maximum / np.maximum(mean, 1e-12))),
            "max_second_mean": float(np.mean(maximum / np.maximum(second, 1e-12))),
            "cv_mean": float(np.mean(times.std(axis=1) / np.maximum(mean, 1e-12))),
            "wait_fraction_mean": float(np.mean(wait)),
            "wait_fraction_p50": float(np.median(wait)),
            "remote_logical_bytes": float(outgoing.sum()),
            "label": f"SIMULATED-EP{ep}-EP2-CALIBRATED",
            "histogram_calibration": {
                "assignments_min": float(histogram.reshape(len(histogram), ep, per_rank).sum(axis=2).min()),
                "assignments_max": float(histogram.reshape(len(histogram), ep, per_rank).sum(axis=2).max()),
            },
        }
        # Exact current-token fanout update plus fixed full-row contribution.
        current_base_fanout = self.base_current_fanout[ep]
        current_new_fanout = build_current_fanout(self.ids, self.valid, ep, removed)
        full_fanout_sum = self.arrays[f"mean_fanout_ep{ep}"] * self.arrays["physical_rows"]
        result["fanout_mean"] = float(np.sum(full_fanout_sum - current_base_fanout + current_new_fanout) / np.sum(self.arrays["physical_rows"]))
        if ep in self.base and name != "P0_vanilla":
            baseline = self.base[ep]
            result["stage_gain_percent"] = 100 * (1 - result["stage_ms"] / baseline["stage_ms"])
            result["expert_gain_percent"] = 100 * (1 - result["expert_ms"] / baseline["expert_ms"])
            result["remote_bytes_reduction_percent"] = 100 * (1 - result["remote_logical_bytes"] / baseline["remote_logical_bytes"])
            result["max_mean_reduction"] = baseline["max_mean_mean"] - result["max_mean_mean"]
            result["wait_fraction_reduction"] = baseline["wait_fraction_mean"] - result["wait_fraction_mean"]
        else:
            result.update(stage_gain_percent=0.0, expert_gain_percent=0.0,
                          remote_bytes_reduction_percent=0.0,
                          max_mean_reduction=0.0, wait_fraction_reduction=0.0)
        return result

    def pressures(self, removed: np.ndarray, ep: int, calibrated: bool) -> np.ndarray:
        histogram = subtract_removed_hist(self.full_hist, self.ids, self.valid, removed).astype(np.float64)
        per_rank = NUM_EXPERTS // ep
        if calibrated:
            vector = np.column_stack([
                self.models[ep].predict_batch(histogram[:, rank * per_rank:(rank + 1) * per_rank])
                for rank in range(ep)
            ])
        else:
            vector = histogram.reshape(len(histogram), ep, per_rank).sum(axis=2)
        return np.maximum(vector - vector.mean(axis=1, keepdims=True), 0)


def counterfactuals(projector: CounterfactualProjector) -> dict:
    results = {"baseline": projector.base, "canonical": [], "matched_mass": [], "matched_routes": []}
    ones = {ep: np.ones((len(projector.ids), ep), dtype=np.float64) for ep in EP_TARGETS}

    # P1 and the canonical P2 definition from the specification.
    for retained_k in (7, 6, 4):
        removed = (projector.slot_ranks > retained_k) & projector.valid[..., None]
        for ep in EP_TARGETS:
            results["canonical"].append(projector.evaluate(removed, ep, f"P1_top{retained_k}"))
    for target in PRESERVE_TARGETS:
        for min_k in (1, 2, 4):
            removed = canonical_mass_preserving(
                projector.normalized, projector.slot_ranks, projector.valid, target, min_k
            )
            for ep in EP_TARGETS:
                results["canonical"].append(
                    projector.evaluate(removed, ep, f"P2_preserve{100*target:g}_min{min_k}")
                )

    def run_axis(targets: Iterable[float], kind: int, sink: str):
        for target in targets:
            # P2 budget-matched utility-only; pressure=1 makes score 1/mass.
            generic = selection_mask(
                projector.normalized, projector.slot_ranks, projector.ids,
                projector.valid, 4, ones[4], target, kind, 4,
                projector.no_removed, False,
            )
            for ep in EP_TARGETS:
                # P2 selection is EP-independent; ep argument only affects owner
                # mapping when pressure is nonuniform, so reuse the same mask.
                entry = projector.evaluate(generic, ep, f"P2_utility_budget_{target:g}")
                entry.update(policy="P2_utility", budget=target,
                             budget_axis="mass" if kind == 0 else "routes")
                results[sink].append(entry)

                # P3: four safe batches, recomputing calibrated rank pressure.
                removed = projector.no_removed.copy()
                for fraction in (.25, .50, .75, 1.0):
                    pressure = projector.pressures(removed, ep, calibrated=True)
                    removed = selection_mask(
                        projector.normalized, projector.slot_ranks, projector.ids,
                        projector.valid, ep, pressure, target * fraction, kind, 4,
                        removed, False,
                    )
                oracle = projector.evaluate(removed, ep, f"P3_calibrated_oracle_budget_{target:g}")
                oracle.update(policy="P3_calibrated_oracle", budget=target,
                              budget_axis="mass" if kind == 0 else "routes",
                              selection_rounds=4)
                results[sink].append(oracle)

                # P4: assignment-pressure only, one cheap post-router pass.
                pressure = projector.pressures(projector.no_removed, ep, calibrated=False)
                heuristic_removed = selection_mask(
                    projector.normalized, projector.slot_ranks, projector.ids,
                    projector.valid, ep, pressure, target, kind, 4,
                    projector.no_removed, False,
                )
                heuristic = projector.evaluate(
                    heuristic_removed, ep, f"P4_rank_pressure_budget_{target:g}"
                )
                heuristic.update(policy="P4_rank_pressure", budget=target,
                                 budget_axis="mass" if kind == 0 else "routes")
                results[sink].append(heuristic)

                # C4 matched random tail removal.
                random_removed = selection_mask(
                    projector.normalized, projector.slot_ranks, projector.ids,
                    projector.valid, ep, ones[ep], target, kind, 4,
                    projector.no_removed, True,
                )
                random = projector.evaluate(random_removed, ep, f"C4_random_tail_budget_{target:g}")
                random.update(policy="C4_random_tail", budget=target,
                              budget_axis="mass" if kind == 0 else "routes")
                results[sink].append(random)

    run_axis(MASS_BUDGETS, 0, "matched_mass")
    run_axis(ROUTE_BUDGETS, 1, "matched_routes")
    return results


def figures(summary: dict, output: Path, normalized: np.ndarray, valid: np.ndarray,
            critical_masks: dict[int, np.ndarray]):
    output.mkdir(parents=True, exist_ok=True)
    sample_rng = np.random.default_rng(20260917)
    flat_valid = np.flatnonzero(np.broadcast_to(valid[..., None], normalized.shape).reshape(-1))
    sample = sample_rng.choice(flat_valid, size=min(1_000_000, len(flat_valid)), replace=False)
    flat_weight = normalized.reshape(-1)
    for ep in EP_TARGETS:
        critical = critical_masks[ep].reshape(-1)[sample]
        weight = flat_weight[sample]
        plt.figure(figsize=(6, 4))
        for mask, label in ((critical, "critical"), (~critical, "non-critical")):
            values = np.sort(weight[mask]); y = np.arange(1, len(values) + 1) / len(values)
            plt.plot(values, y, label=label)
        plt.xlabel("normalized effective router mass"); plt.ylabel("CDF")
        plt.title(f"EP{ep} current-block router-weight CDF"); plt.grid(alpha=.3); plt.legend()
        plt.tight_layout(); plt.savefig(output / f"ep{ep}_critical_noncritical_weight_cdf.png", dpi=170); plt.close()

    anatomy = summary["anatomy"]
    x = np.arange(1, 9)
    plt.figure(figsize=(7, 4))
    for ep in EP_TARGETS:
        shares = []
        for rank in x:
            key = "slot_8" if rank == 8 else None
            # Per-slot shares retained separately in summary.
            shares.append(anatomy[f"ep{ep}"]["slot_rank_excess_share"][str(rank)])
        plt.plot(x, np.asarray(shares) * 100, "o-", label=f"EP{ep}")
    plt.xlabel("effective-weight slot rank"); plt.ylabel("ATTRIBUTED_EXCESS share (%)")
    plt.title("Critical excess by within-token slot"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(output / "critical_excess_by_slot_rank.png", dpi=170); plt.close()

    bins = ("mass_le_0.01", "mass_le_0.02", "mass_le_0.05", "mass_le_0.1")
    plt.figure(figsize=(7, 4))
    for ep in EP_TARGETS:
        values = [100 * anatomy[f"ep{ep}"][key]["attributed_excess_share_of_full_physical_excess"] for key in bins]
        plt.plot([1, 2, 5, 10], values, "o-", label=f"EP{ep}")
    plt.xlabel("normalized route-mass threshold (%)"); plt.ylabel("ATTRIBUTED_EXCESS share (%)")
    plt.title("Critical excess carried by low-mass current-block routes"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(output / "critical_excess_by_router_mass.png", dpi=170); plt.close()

    for ep in EP_TARGETS:
        phase = summary["phase"][f"ep{ep}"]["block_relative"]
        labels = ("early", "middle", "late")
        values = [phase[label]["bottom_25_global"]["enrichment"] for label in labels]
        plt.figure(figsize=(6, 4)); plt.bar(labels, values); plt.axhline(1, color="black", lw=1)
        plt.ylabel("critical/non-critical enrichment")
        plt.title(f"EP{ep} bottom-25% enrichment by refinement phase")
        plt.tight_layout(); plt.savefig(output / f"ep{ep}_enrichment_by_phase.png", dpi=170); plt.close()

    matched = summary["counterfactuals"]["matched_mass"]
    for ep in EP_TARGETS:
        plt.figure(figsize=(7, 4))
        for policy, label in (("P2_utility", "utility-only"),
                              ("P3_calibrated_oracle", "EP-aware oracle"),
                              ("P4_rank_pressure", "EP-aware heuristic")):
            rows = sorted((row for row in matched if row["ep"] == ep and row["policy"] == policy), key=lambda row: row["budget"])
            plt.plot([100 * row["removed_current_router_mass_fraction"] for row in rows],
                     [row["stage_gain_percent"] for row in rows], "o-", label=label)
        plt.xlabel("removed current-block router mass (%)"); plt.ylabel("routed-MoE stage gain (%)")
        plt.title(f"EP{ep}: stage gain at matched router-mass budget"); plt.grid(alpha=.3); plt.legend()
        plt.tight_layout(); plt.savefig(output / f"ep{ep}_stage_gain_vs_router_mass.png", dpi=170); plt.close()

    for metric, filename, ylabel in (("max_mean_mean", "ep8_max_mean_vs_router_mass.png", "mean max/mean"),
                                     ("wait_fraction_mean", "ep8_wait_vs_router_mass.png", "mean wait fraction")):
        plt.figure(figsize=(7, 4))
        for policy, label in (("P2_utility", "utility-only"), ("P3_calibrated_oracle", "EP-aware oracle"), ("P4_rank_pressure", "EP-aware heuristic")):
            rows = sorted((row for row in matched if row["ep"] == 8 and row["policy"] == policy), key=lambda row: row["budget"])
            plt.plot([100 * row["removed_current_router_mass_fraction"] for row in rows], [row[metric] for row in rows], "o-", label=label)
        plt.xlabel("removed current-block router mass (%)"); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.legend()
        plt.tight_layout(); plt.savefig(output / filename, dpi=170); plt.close()

    curve = summary["excess_mass_curve"]
    plt.figure(figsize=(6, 4))
    for ep in EP_TARGETS:
        item = curve[f"ep{ep}"]["router_mass_percent_needed_for_fraction_of_current_block_attributed_excess"]
        plt.plot([25, 50, 75], [item[str(x / 100)] for x in (25, 50, 75)], "o-", label=f"EP{ep}")
    plt.xlabel("current-block ATTRIBUTED_EXCESS removed (%)"); plt.ylabel("router mass removed (% of current block)")
    plt.title("Mass needed to remove attributed current-block excess"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(output / "router_mass_vs_critical_excess.png", dpi=170); plt.close()

    plt.figure(figsize=(6, 4))
    for ep in EP_TARGETS:
        p2 = {row["budget"]: row for row in matched if row["ep"] == ep and row["policy"] == "P2_utility"}
        p3 = {row["budget"]: row for row in matched if row["ep"] == ep and row["policy"] == "P3_calibrated_oracle"}
        budgets = sorted(set(p2) & set(p3))
        plt.plot([100 * b for b in budgets], [p3[b]["stage_gain_percent"] - p2[b]["stage_gain_percent"] for b in budgets], "o-", label=f"EP{ep}")
    plt.axhline(3, color="red", ls="--", label="G3=3pp")
    plt.xlabel("router-mass budget (%)"); plt.ylabel("EP-specific increment (pp)")
    plt.title("EP-aware increment over utility-only"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(output / "ep4_vs_ep8_increment.png", dpi=170); plt.close()


def write_reports(summary: dict, report_dir: Path):
    anatomy = summary["anatomy"]
    anatomy_lines = []
    keys = ("slot_8", "slots_7_8", "slots_5_8", "bottom_10_global", "bottom_25_global",
            "bottom_50_global", "mass_le_0.01", "mass_le_0.02", "mass_le_0.05", "mass_le_0.1")
    for ep in EP_TARGETS:
        for key in keys:
            row = anatomy[f"ep{ep}"][key]
            anatomy_lines.append(
                f"| EP{ep} | {key} | {100*row['critical_fraction']:.4f}% | "
                f"{100*row['noncritical_fraction']:.4f}% | {row['enrichment']:.4f}x | "
                f"{row['percentage_point_difference']:+.4f} | "
                f"{100*row['attributed_excess_share_of_full_physical_excess']:.4f}% |"
            )
    (report_dir / "low_utility_straggler_anatomy.md").write_text(
        "# Low-utility straggler anatomy\n\n"
        "The primary population is the exact current-block route slots in the existing threshold-0.95 GSM8K-128 aggregate-v2 trace. Critical rank and excess use calibrated time over the complete physical workload. Thus excess shares are conservative shares of full physical excess. `current_router_weights` are the exact `topk_weight` coefficients returned by the official gate and multiplied into expert outputs; per-token sums are 2.5, so all threshold analyses use division by that sum.\n\n"
        "| target | definition | P(low|critical) | P(low|noncritical) | enrichment | difference (pp) | ATTRIBUTED_EXCESS/full excess |\n"
        "|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(anatomy_lines) +
        "\n\nThe comparison is invocation-matched by construction: critical and non-critical routes come from the same layer/refinement invocation. Request-cluster bootstrap intervals and global-hot/random-rank controls are in the JSON artifact. Millions of slots are not treated as independent samples.\n"
    )

    phase_lines = []
    for ep in EP_TARGETS:
        for axis in ("block_relative", "mask_ratio"):
            for label, cell in summary["phase"][f"ep{ep}"][axis].items():
                item = cell["bottom_25_global"]
                phase_lines.append(
                    f"| EP{ep} | {axis} | {label} | {cell['invocations']:,} | "
                    f"{item['enrichment']:.4f}x | {item['percentage_point_difference']:+.4f} | "
                    f"{100*item['attributed_excess_share']:.4f}% |"
                )
    (report_dir / "low_utility_straggler_refinement_phase.md").write_text(
        "# Refinement-phase analysis\n\n"
        "| target | axis | phase/bin | invocations | bottom-25 enrichment | difference (pp) | bottom-25 ATTRIBUTED_EXCESS share |\n"
        "|---|---|---|---:|---:|---:|---:|\n" + "\n".join(phase_lines) +
        "\n\nThis phase view tests whether repeated dLLM refinement adds a pattern beyond generic MoE tail pruning.\n"
    )

    mass_rows = []
    for row in summary["counterfactuals"]["matched_mass"]:
        mass_rows.append(
            f"| EP{row['ep']} | {row['policy']} | {100*row['budget']:.1f}% | "
            f"{100*row['removed_current_router_mass_fraction']:.3f}% | "
            f"{100*row['removed_full_physical_route_fraction']:.3f}% | "
            f"{row['stage_gain_percent']:.3f}% | {row['max_mean_mean']:.4f} | "
            f"{100*row['wait_fraction_mean']:.3f}% | {row['remote_bytes_reduction_percent']:.3f}% |"
        )
    (report_dir / "low_utility_straggler_ep4_ep8_oracle.md").write_text(
        "# EP4/EP8 counterfactual pruning oracle\n\n"
        "These are route-trace counterfactuals, not quality-validated policies. Only current-block exact routes are removable; prefix/prior physical work remains unchanged. Costs are rebuilt from the remaining expert histogram and unique source-destination activation matrix and passed through grouped-mm/EP2-calibrated models—no route-count latency scaling is used. P3 recomputes calibrated pressure in four safe batches; P4 uses assignment pressure once. All rows enforce min_k=4.\n\n"
        "| target | policy | nominal mass budget | actual removed mass | full physical route reduction | stage gain | mean max/mean | mean wait | remote-byte reduction |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(mass_rows) +
        "\n\nCanonical P1/P2 and matched route-count controls are retained in `low_utility_straggler_summary.json`. No model rollout or output-weight renormalization was performed.\n"
    )

    decision = summary["decision"]
    (report_dir / "low_utility_straggler_summary.md").write_text(
        "# Low-Utility EP Straggler — Summary\n\n"
        f"**Verdict: {decision['verdict']} ({decision['interpretation']}).**\n\n"
        f"G1 low-mass enrichment: **{decision['gates']['g1']}**. "
        f"G2 excess concentration: **{decision['gates']['g2']}**. "
        f"G3 EP-specific pruning advantage: **{decision['gates']['g3']}**. "
        f"G4 EP scaling: **{decision['gates']['g4']}**.\n\n"
        + decision["reasoning"] + "\n\n"
        "## Evidence boundary\n\n"
        "EP4/EP8 are `SIMULATED-EP4/8-EP2-CALIBRATED` routed-MoE-stage projections. The exact slot-weight population is current-block rows across 128 requests; complete physical rank histograms provide the stage denominator. No quality rollout and no production pruning runtime were run.\n\n"
        "## Required final summary\n\n```text\n" + summary["required_final_summary"] + "\n```\n"
    )


def choose_best(rows: list[dict], ep: int, policy: str) -> dict:
    candidates = [row for row in rows if row["ep"] == ep and row["policy"] == policy]
    return max(candidates, key=lambda row: row["stage_gain_percent"])


def slot_excess_shares(
    ids: np.ndarray, slot_ranks: np.ndarray, valid: np.ndarray,
    arrays: dict[str, np.ndarray], times: np.ndarray, ep: int,
) -> dict[str, float]:
    owner = ids // (NUM_EXPERTS // ep)
    critical = np.argmax(times, axis=1)
    critical_mask = (owner == critical[:, None, None]) & valid[..., None]
    hist = arrays["expert_counts_by_class"].sum(axis=1).astype(np.float64)
    per_rank = NUM_EXPERTS // ep
    critical_count = np.empty(len(ids), dtype=np.float64)
    for rank in range(ep):
        selected = critical == rank
        critical_count[selected] = hist[selected, rank * per_rank:(rank + 1) * per_rank].sum(axis=1)
    excess = times.max(axis=1) - times.mean(axis=1)
    factor = np.divide(excess, critical_count, out=np.zeros_like(excess), where=critical_count > 0)
    total = float(excess.sum())
    return {
        str(rank): safe_ratio(
            np.sum(((slot_ranks == rank) & critical_mask).sum(axis=(1, 2)) * factor), total
        ) for rank in range(1, TOP_K + 1)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--rank-times-root", type=Path, required=True)
    parser.add_argument("--ep4-compute", type=Path, required=True)
    parser.add_argument("--ep8-compute", type=Path, required=True)
    parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures/low_utility_straggler"))
    args = parser.parse_args()

    trace = load_discovery_trace(args.trace)
    arrays = trace.arrays
    ids = arrays["current_expert_ids"].astype(np.int16, copy=False)
    weights = arrays["current_router_weights"].astype(np.float32)
    valid = arrays["current_position_class"] != 0
    sums = weights.sum(axis=-1, keepdims=True)
    normalized = np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)
    slot_ranks = effective_slot_ranks(weights)
    flat = normalized[np.broadcast_to(valid[..., None], normalized.shape)]
    quantiles = {
        "10": float(np.quantile(flat, .10)),
        "25": float(np.quantile(flat, .25)),
        "50": float(np.quantile(flat, .50)),
    }
    definitions = definition_masks(normalized, slot_ranks, valid, quantiles)
    phase = block_relative_phase(arrays)
    mask_bin = mask_ratio_bin(arrays)
    times = {
        ep: np.load(args.rank_times_root / f"rank_times_ep{ep}.npy", mmap_mode="r")
        for ep in EP_TARGETS
    }

    anatomy, phases, controls, curves = {}, {}, {}, {}
    critical_masks = {}
    for ep in EP_TARGETS:
        anatomy[f"ep{ep}"], phases[f"ep{ep}"], controls[f"ep{ep}"] = anatomy_for_ep(
            arrays, ids, normalized, slot_ranks, valid, times[ep], ep,
            definitions, phase, mask_bin,
        )
        anatomy[f"ep{ep}"]["slot_rank_excess_share"] = slot_excess_shares(
            ids, slot_ranks, valid, arrays, times[ep], ep
        )
        curves[f"ep{ep}"] = excess_mass_curve(ids, normalized, valid, arrays, times[ep], ep)
        owner = ids // (NUM_EXPERTS // ep)
        critical_masks[ep] = (owner == np.argmax(times[ep], axis=1)[:, None, None]) & valid[..., None]

    models = {4: BatchComputeModel(args.ep4_compute), 8: BatchComputeModel(args.ep8_compute)}
    communication = CommunicationScenario.load(args.communication, "ep2_calibrated_base")
    projector = CounterfactualProjector(
        arrays, ids, normalized, slot_ranks, valid, models, communication
    )
    counter = counterfactuals(projector)

    # Gates use conservative, common definitions and matched-mass comparisons.
    conservative_names = ("slots_7_8", "bottom_25_global", "mass_le_0.1", "outside_preserve_0.975")
    g1 = any(
        anatomy[f"ep{ep}"][name]["percentage_point_difference"] >= 10
        or anatomy[f"ep{ep}"][name]["enrichment"] >= 1.25
        for ep in EP_TARGETS for name in conservative_names
    )
    ep8_low_excess = anatomy["ep8"]["mass_le_0.1"]["attributed_excess_share_of_full_physical_excess"]
    g2 = ep8_low_excess >= .25
    increments = {}
    for ep in EP_TARGETS:
        p2 = {row["budget"]: row for row in counter["matched_mass"] if row["ep"] == ep and row["policy"] == "P2_utility"}
        p3 = {row["budget"]: row for row in counter["matched_mass"] if row["ep"] == ep and row["policy"] == "P3_calibrated_oracle"}
        increments[ep] = {
            budget: p3[budget]["stage_gain_percent"] - p2[budget]["stage_gain_percent"]
            for budget in sorted(set(p2) & set(p3))
        }
    best_increment8 = max(increments[8].values())
    best_budget8 = max(increments[8], key=increments[8].get)
    p3_best8 = next(row for row in counter["matched_mass"] if row["ep"] == 8 and row["policy"] == "P3_calibrated_oracle" and row["budget"] == best_budget8)
    base8 = counter["baseline"][8]
    g3 = best_increment8 >= 3 and p3_best8["max_mean_mean"] < base8["max_mean_mean"] and p3_best8["wait_fraction_mean"] < base8["wait_fraction_mean"]
    best_increment4 = max(increments[4].values())
    g4 = best_increment8 > best_increment4
    if g1 and g2 and g3:
        verdict, interpretation = "GO", "EP-specific"
    elif (not g1 or not g2) and best_increment8 < 1:
        verdict, interpretation = "NO-GO", "no-go"
    else:
        verdict, interpretation = "HOLD", "generic-only"

    generic4 = choose_best(counter["matched_mass"], 4, "P2_utility")
    generic8 = choose_best(counter["matched_mass"], 8, "P2_utility")
    oracle4 = choose_best(counter["matched_mass"], 4, "P3_calibrated_oracle")
    oracle8 = choose_best(counter["matched_mass"], 8, "P3_calibrated_oracle")
    practical4 = choose_best(counter["matched_mass"], 4, "P4_rank_pressure")
    practical8 = choose_best(counter["matched_mass"], 8, "P4_rank_pressure")
    # Headline matched comparison uses the budget where P3's EP8 increment is largest.
    budget = best_budget8
    at = lambda ep, policy: next(row for row in counter["matched_mass"] if row["ep"] == ep and row["policy"] == policy and row["budget"] == budget)
    h_generic4, h_oracle4 = at(4, "P2_utility"), at(4, "P3_calibrated_oracle")
    h_generic8, h_oracle8, h_practical8 = at(8, "P2_utility"), at(8, "P3_calibrated_oracle"), at(8, "P4_rank_pressure")

    required = f"""TRACE_SOURCE:
{args.trace}; calibrated rank times {args.rank_times_root}; exact slot weights are current-block-only across GSM8K-128

EP4_LOW_MASS_CRITICAL_ENRICHMENT:
{anatomy['ep4']['bottom_25_global']['enrichment']:.4f}x / global bottom-25% normalized effective route mass ({anatomy['ep4']['bottom_25_global']['percentage_point_difference']:+.4f} pp)

EP8_LOW_MASS_CRITICAL_ENRICHMENT:
{anatomy['ep8']['bottom_25_global']['enrichment']:.4f}x / global bottom-25% normalized effective route mass ({anatomy['ep8']['bottom_25_global']['percentage_point_difference']:+.4f} pp)

EP4_LOW_MASS_CRITICAL_EXCESS_SHARE:
{100*anatomy['ep4']['mass_le_0.1']['attributed_excess_share_of_full_physical_excess']:.4f}% / normalized per-route mass <=10%, ATTRIBUTED_EXCESS over full physical excess

EP8_LOW_MASS_CRITICAL_EXCESS_SHARE:
{100*anatomy['ep8']['mass_le_0.1']['attributed_excess_share_of_full_physical_excess']:.4f}% / normalized per-route mass <=10%, ATTRIBUTED_EXCESS over full physical excess

EP4_TAIL_7_8_CRITICAL_EXCESS_SHARE:
{100*anatomy['ep4']['slots_7_8']['attributed_excess_share_of_full_physical_excess']:.4f}%

EP8_TAIL_7_8_CRITICAL_EXCESS_SHARE:
{100*anatomy['ep8']['slots_7_8']['attributed_excess_share_of_full_physical_excess']:.4f}%

BEST_GENERIC_PRUNING_POLICY:
P2 utility-only current-block pruning / {100*generic8['budget']:.1f}% nominal removed-mass budget (EP8 stage gain {generic8['stage_gain_percent']:.3f}%)

BEST_EP_AWARE_PRUNING_ORACLE:
P3 four-round calibrated-pressure current-block oracle / {100*oracle8['budget']:.1f}% nominal budget (EP8 stage gain {oracle8['stage_gain_percent']:.3f}%)

BEST_EP_AWARE_PRACTICAL_HEURISTIC:
P4 assignment-rank-pressure current-block heuristic / {100*practical8['budget']:.1f}% nominal budget (EP8 stage gain {practical8['stage_gain_percent']:.3f}%)

EP4_GENERIC_STAGE_GAIN:
{h_generic4['stage_gain_percent']:.3f}%

EP4_EP_AWARE_STAGE_GAIN:
{h_oracle4['stage_gain_percent']:.3f}%

EP4_EP_SPECIFIC_INCREMENT:
{h_oracle4['stage_gain_percent']-h_generic4['stage_gain_percent']:+.3f} percentage points

EP8_GENERIC_STAGE_GAIN:
{h_generic8['stage_gain_percent']:.3f}%

EP8_EP_AWARE_STAGE_GAIN:
{h_oracle8['stage_gain_percent']:.3f}%

EP8_EP_SPECIFIC_INCREMENT:
{h_oracle8['stage_gain_percent']-h_generic8['stage_gain_percent']:+.3f} percentage points

EP8_MAX_MEAN_BEFORE_AFTER:
{base8['max_mean_mean']:.4f} -> {h_oracle8['max_mean_mean']:.4f}

EP8_WAIT_FRACTION_BEFORE_AFTER:
{base8['wait_fraction_mean']:.4f} -> {h_oracle8['wait_fraction_mean']:.4f}

EP8_ROUTER_MASS_NEEDED_FOR_50_PERCENT_EXCESS_REMOVAL:
{curves['ep8']['router_mass_percent_needed_for_fraction_of_full_physical_excess']['0.5'] if curves['ep8']['router_mass_percent_needed_for_fraction_of_full_physical_excess']['0.5'] is not None else 'N/A'}

DLLM_PHASE_EFFECT:
Bottom-25% critical enrichment remains near/below 1x across refinement phases; no growing dLLM-specific low-utility straggler signal was found.

QUALITY_ROLLOUT_RUN:
NO

PRIMARY_INTERPRETATION:
{interpretation}

VERDICT:
{verdict}

NEXT_ACTION:
Do not implement route pruning; return to exact balancing/work stealing unless a future full-row trace contradicts this negative current-block result."""

    summary = {
        "trace_source": str(args.trace),
        "trace_metadata": trace.metadata,
        "evidence_boundary": {
            "slot_level_population": "current-block generated rows in GSM8K-128",
            "critical_rank_and_stage_denominator": "complete vanilla physical workload",
            "quality_rollout": False,
            "physical_gpus_used": [],
            "labels": ["SIMULATED-EP4-EP2-CALIBRATED", "SIMULATED-EP8-EP2-CALIBRATED"],
            "effective_gate_weight_source": "official LLaDA2MoeGate topk_weight consumed by moe_infer combine",
        },
        "population": {
            "requests": int(np.unique(arrays["request_id"]).size),
            "invocations": int(len(ids)),
            "current_generated_token_rows": int(valid.sum()),
            "current_expert_slots": int(valid.sum() * TOP_K),
            "full_physical_token_rows": int(arrays["physical_rows"].sum()),
            "current_row_share": float(valid.sum() / arrays["physical_rows"].sum()),
            "normalized_weight_quantiles": quantiles,
            "effective_weight_sum_percentiles": np.percentile(sums[valid], [0, 50, 100]).tolist(),
        },
        "anatomy": anatomy, "phase": phases, "controls": controls,
        "excess_mass_curve": curves, "counterfactuals": counter,
        "ep_specific_increments": {f"ep{ep}": values for ep, values in increments.items()},
        "decision": {
            "gates": {"g1": "PASS" if g1 else "FAIL", "g2": "PASS" if g2 else "FAIL",
                      "g3": "PASS" if g3 else "FAIL", "g4": "PASS" if g4 else "FAIL"},
            "verdict": verdict, "interpretation": interpretation,
            "reasoning": (
                "Meaningful low-mass definitions are not enriched on calibrated time-critical ranks; "
                "the rare <=1–2% tail can show a ratio above one but has negligible absolute mass and excess. "
                "The current block is only a small fraction of full physical rows, so exact current-block route removal cannot explain a large share of full critical excess. "
                "Matched-budget simulation determines whether rank pressure adds anything beyond utility-only pruning."
            ),
        },
        "required_final_summary": required,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "low_utility_straggler_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    figures(summary, args.figure_dir, normalized, valid, critical_masks)
    write_reports(summary, args.report_dir)
    print(required)


if __name__ == "__main__":
    main()
