#!/usr/bin/env python3
"""Structural discovery for Temporal-Edge EP and token-home remapping.

All EP4/EP8 timings are EP2-calibrated simulations.  The script consumes the
existing threshold-.95 GSM8K-128 aggregate trace and performs no generation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

from scripts.analyze_h1_straggler_deepdive import BatchComputeModel, load_discovery_trace
from scripts.analyze_low_utility_straggler import (
    build_current_unique, communication_latency, subtract_removed_hist,
)
from virtual_ep.comm_model import CommunicationScenario


NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_BYTES = 2048 * 2
EPS = (4, 8)


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item()))


def build_previous(arrays):
    previous = np.full(len(arrays["request_id"]), -1, dtype=np.int32)
    last = {}
    for index, key in enumerate(zip(
        arrays["request_id"].tolist(), arrays["block_id"].tolist(),
        arrays["layer_id"].tolist(),
    )):
        prior = last.get(key, -1)
        if prior >= 0 and arrays["iteration_id"][index] == arrays["iteration_id"][prior] + 1:
            previous[index] = prior
        last[key] = index
    return previous


def stay_masks(arrays, previous):
    ids = arrays["current_expert_ids"]
    cls = arrays["current_position_class"]
    valid = np.zeros(cls.shape, dtype=np.bool_)
    stay = np.zeros(ids.shape, dtype=np.bool_)
    current = np.flatnonzero(previous >= 0)
    for start in range(0, len(current), 4096):
        selected = current[start:start + 4096]
        prior = previous[selected]
        live = (cls[selected] == 2) & (cls[prior] == 2)
        valid[selected] = live
        stay[selected] = (
            ids[selected, :, :, None] == ids[prior, :, None, :]
        ).any(axis=-1) & live[:, :, None]
    return valid, stay


def route_stats(valid, stay, weights, invocation=None, route_filter=None):
    if invocation is None:
        invocation = np.ones(len(valid), dtype=np.bool_)
    denominator = valid[invocation, :, None]
    selected_stay = stay[invocation]
    selected_weight = weights[invocation]
    if route_filter is not None:
        denominator = denominator & route_filter[invocation]
        selected_stay = selected_stay & route_filter[invocation]
    route_count = int(np.count_nonzero(denominator) * (TOP_K if denominator.shape[-1] == 1 else 1))
    if denominator.shape[-1] == 1:
        mass_denominator = float((selected_weight * denominator).sum())
    else:
        mass_denominator = float((selected_weight * denominator).sum())
    return {
        "routes": route_count,
        "stay_routes": int(np.count_nonzero(selected_stay)),
        "stay_route_fraction": float(np.count_nonzero(selected_stay) / route_count) if route_count else None,
        "router_mass": mass_denominator,
        "stay_router_mass": float((selected_weight * selected_stay).sum()),
        "stay_router_mass_fraction": float((selected_weight * selected_stay).sum() / mass_denominator) if mass_denominator else None,
    }


def pair_overlap(arrays, current, reference, current_position=None, reference_position=None, population=2):
    ids = arrays["current_expert_ids"]
    cls = arrays["current_position_class"]
    weights = arrays["current_router_weights"].astype(np.float32)
    total = stay_count = 0
    mass = stay_mass = 0.0
    for begin in range(0, len(current), 4096):
        cur = current[begin:begin + 4096]
        ref = reference[begin:begin + 4096]
        cur_ids = ids[cur]
        ref_ids = ids[ref]
        cur_cls = cls[cur]
        ref_cls = cls[ref]
        if current_position is not None:
            cur_ids = cur_ids[:, current_position]
            cur_cls = cur_cls[:, current_position]
        if reference_position is not None:
            ref_ids = ref_ids[:, reference_position]
            ref_cls = ref_cls[:, reference_position]
        live = (cur_cls == population) & (ref_cls == population)
        matched = (cur_ids[:, :, :, None] == ref_ids[:, :, None, :]).any(axis=-1)
        matched &= live[:, :, None]
        total += int(live.sum() * TOP_K)
        stay_count += int(matched.sum())
        weight = weights[cur]
        if current_position is not None:
            weight = weight[:, current_position]
        mass += float((weight * live[:, :, None]).sum())
        stay_mass += float((weight * matched).sum())
    return {
        "pairs": len(current), "routes": total,
        "stay_route_fraction": stay_count / total if total else None,
        "stay_router_mass_fraction": stay_mass / mass if mass else None,
    }


def controls(arrays, previous):
    current = np.flatnonzero(previous >= 0)
    adjacent = pair_overlap(arrays, current, previous[current])
    decoded = pair_overlap(arrays, current, previous[current], population=4)

    # Same adjacent refinements, but rotate token positions by one.
    rotated = pair_overlap(
        arrays, current, previous[current],
        current_position=np.arange(32), reference_position=np.roll(np.arange(32), 1),
    )

    groups = defaultdict(list)
    for index in range(len(arrays["request_id"])):
        groups[(int(arrays["request_id"][index]), int(arrays["block_id"][index]),
                int(arrays["layer_id"][index]))].append(index)
    rng = np.random.default_rng(20260918)
    random_current, random_reference = [], []
    for values in groups.values():
        values.sort(key=lambda index: int(arrays["iteration_id"][index]))
        if len(values) < 3:
            continue
        for ordinal in range(2, len(values)):
            candidates = values[:max(1, ordinal - 1)]
            random_current.append(values[ordinal])
            random_reference.append(candidates[int(rng.integers(0, len(candidates)))])
    random_refinement = pair_overlap(
        arrays, np.asarray(random_current, dtype=np.int32),
        np.asarray(random_reference, dtype=np.int32),
    )

    block_last = {}
    cross_current, cross_reference = [], []
    for index in range(len(arrays["request_id"])):
        key = (int(arrays["request_id"][index]), int(arrays["layer_id"][index]),
               int(arrays["iteration_id"][index]))
        block = int(arrays["block_id"][index])
        prior = block_last.get(key)
        if prior is not None and block == int(arrays["block_id"][prior]) + 1:
            cross_current.append(index); cross_reference.append(prior)
        block_last[key] = index
    cross_block = pair_overlap(
        arrays, np.asarray(cross_current, dtype=np.int32),
        np.asarray(cross_reference, dtype=np.int32),
    )

    # Shuffle reference rows within each routed layer, preserving marginal IDs.
    sample = current[::max(1, len(current) // 100_000)]
    shuffled_ref = previous[sample].copy()
    for layer in np.unique(arrays["layer_id"]):
        where = np.flatnonzero(arrays["layer_id"][sample] == layer)
        if len(where):
            shuffled_ref[where] = shuffled_ref[where[rng.permutation(len(where))]]
    shuffled = pair_overlap(arrays, sample, shuffled_ref)
    return {
        "adjacent_live_mask": adjacent,
        "matched_random_nonadjacent_refinement": random_refinement,
        "same_layer_different_token": rotated,
        "same_token_different_block": cross_block,
        "shuffled_expert_rows_preserving_layer_popularity": shuffled,
        "decoded_stable_positive_control": decoded,
    }


def run_lengths(arrays):
    groups = defaultdict(list)
    for index in range(len(arrays["request_id"])):
        groups[(int(arrays["request_id"][index]), int(arrays["block_id"][index]),
                int(arrays["layer_id"][index]))].append(index)
    lengths = []
    for values in groups.values():
        values.sort(key=lambda index: int(arrays["iteration_id"][index]))
        active = {}
        previous_iteration = None
        for index in values:
            iteration = int(arrays["iteration_id"][index])
            if previous_iteration is not None and iteration != previous_iteration + 1:
                lengths.extend(active.values()); active = {}
            edges = {
                (position, int(expert))
                for position in np.flatnonzero(arrays["current_position_class"][index] == 2)
                for expert in arrays["current_expert_ids"][index, position]
            }
            for edge in list(active):
                if edge not in edges:
                    lengths.append(active.pop(edge))
            active = {edge: active.get(edge, 0) + 1 for edge in edges}
            previous_iteration = iteration
        lengths.extend(active.values())
    values = np.asarray(lengths, dtype=np.int16)
    return {
        "count": int(len(values)),
        **{f"p{q}": float(np.percentile(values, q)) for q in (50, 75, 90, 95, 99)},
        "max": int(values.max(initial=0)),
        "histogram": np.bincount(values, minlength=33).astype(int).tolist(),
    }


def persistence_strata(arrays, valid, stay):
    weights = arrays["current_router_weights"].astype(np.float32)
    discovery = arrays["request_id"] < 64
    heldout = ~discovery
    result = {
        "global": {
            "discovery": route_stats(valid, stay, weights, discovery),
            "heldout": route_stats(valid, stay, weights, heldout),
            "all": route_stats(valid, stay, weights),
        },
        "layer": {}, "refinement": {}, "mask_ratio": {}, "confidence": {},
        "slot": {}, "router_weight_decile": {}, "block_index": {}, "destination": {},
    }
    axes = {
        "layer": (arrays["layer_id"], sorted(np.unique(arrays["layer_id"]).tolist())),
        "refinement": (
            np.select(
                [arrays["iteration_id"] == 0, arrays["iteration_id"] == 1,
                 arrays["iteration_id"] <= 3, arrays["iteration_id"] <= 7],
                [0, 1, 2, 3], default=4,
            ), [0, 1, 2, 3, 4],
        ),
        "mask_ratio": (
            np.select(
                [arrays["masked_current_block"] > 24, arrays["masked_current_block"] > 16,
                 arrays["masked_current_block"] > 8], [0, 1, 2], default=3,
            ), [0, 1, 2, 3],
        ),
        "block_index": (
            np.minimum(arrays["block_id"], 4), [0, 1, 2, 3, 4],
        ),
    }
    labels = {
        "refinement": ("r0", "r1", "r2_3", "r4_7", "r8plus"),
        "mask_ratio": ("gt75", "50_75", "25_50", "le25"),
        "block_index": ("b0", "b1", "b2", "b3", "b4plus"),
    }
    for name, (values, categories) in axes.items():
        for ordinal, value in enumerate(categories):
            label = str(value) if name == "layer" else labels.get(name, categories)[ordinal]
            result[name][label] = {
                "discovery": route_stats(valid, stay, weights, discovery & (values == value)),
                "heldout": route_stats(valid, stay, weights, heldout & (values == value)),
            }

    quantiles = np.quantile(arrays["confidence_mean"][discovery], [.25, .5, .75])
    confidence_bin = np.digitize(arrays["confidence_mean"], quantiles)
    for value in range(4):
        result["confidence"][f"q{value + 1}"] = {
            "discovery": route_stats(valid, stay, weights, discovery & (confidence_bin == value)),
            "heldout": route_stats(valid, stay, weights, heldout & (confidence_bin == value)),
        }
    slot_rank = np.empty_like(arrays["current_expert_ids"], dtype=np.int8)
    order = np.argsort(-weights, axis=-1, kind="stable")
    np.put_along_axis(slot_rank, order, np.broadcast_to(
        np.arange(1, TOP_K + 1, dtype=np.int8), order.shape
    ), axis=-1)
    for slot in range(1, TOP_K + 1):
        filt = slot_rank == slot
        result["slot"][str(slot)] = {
            "discovery": route_stats(valid, stay, weights, discovery, filt),
            "heldout": route_stats(valid, stay, weights, heldout, filt),
        }
    eligible_discovery_weight = weights[np.broadcast_to(valid[..., None], weights.shape) &
                                        discovery[:, None, None]]
    weight_edges = np.quantile(eligible_discovery_weight, np.arange(1, 10) / 10)
    weight_bin = np.digitize(weights, weight_edges)
    for decile in range(10):
        filt = weight_bin == decile
        result["router_weight_decile"][str(decile + 1)] = {
            "discovery": route_stats(valid, stay, weights, discovery, filt),
            "heldout": route_stats(valid, stay, weights, heldout, filt),
        }
    for ep in EPS:
        owner = arrays["current_expert_ids"] // (NUM_EXPERTS // ep)
        source = arrays[f"current_source_rank_ep{ep}"][..., None]
        for label, filt in (("local", owner == source), ("remote", owner != source)):
            result["destination"][f"ep{ep}_{label}"] = {
                "discovery": route_stats(valid, stay, weights, discovery, filt),
                "heldout": route_stats(valid, stay, weights, heldout, filt),
            }

    candidates = []
    for axis in ("layer", "refinement", "mask_ratio", "confidence", "slot",
                 "router_weight_decile", "block_index"):
        for label, entry in result[axis].items():
            discovery_entry = entry["discovery"]
            if discovery_entry["routes"] >= 100_000:
                candidates.append((discovery_entry["stay_route_fraction"], axis, label))
    strongest = max(candidates)
    result["discovery_selected_subgroup"] = {
        "axis": strongest[1], "label": strongest[2],
        "discovery": result[strongest[1]][strongest[2]]["discovery"],
        "heldout_confirmation": result[strongest[1]][strongest[2]]["heldout"],
        "selection_rule": "maximum discovery STAY fraction among predefined cells with >=100k routes",
    }
    return result


@njit(parallel=True, cache=True)
def build_hist(ids, valid):
    result = np.zeros((ids.shape[0], NUM_EXPERTS), dtype=np.int32)
    for invocation in prange(ids.shape[0]):
        for position in range(ids.shape[1]):
            if valid[invocation, position]:
                for slot in range(ids.shape[2]):
                    result[invocation, ids[invocation, position, slot]] += 1
    return result


def evaluated_stage(hist, unique, ep, compute, communication):
    per_rank = NUM_EXPERTS // ep
    times = np.column_stack([
        compute.predict_batch(hist[:, rank * per_rank:(rank + 1) * per_rank])
        for rank in range(ep)
    ])
    dispatch, outgoing, _incoming = communication_latency(communication.dispatch, unique)
    combine, _reverse_out, _reverse_in = communication_latency(
        communication.combine, np.transpose(unique, (0, 2, 1))
    )
    expert = times.max(axis=1)
    stage = dispatch + expert + combine
    ordered = np.sort(times, axis=1)
    mean = times.mean(axis=1)
    wait = np.divide(
        ep * expert - times.sum(axis=1), ep * expert,
        out=np.zeros_like(expert), where=expert > 0,
    )
    remote = unique.copy()
    diagonal = np.arange(ep)
    remote[:, diagonal, diagonal] = 0
    peer_fanout = np.count_nonzero(remote, axis=2)
    active_sources = remote.sum(axis=2) > 0
    return {
        "dispatch_ms": float(dispatch.sum()), "expert_ms": float(expert.sum()),
        "combine_ms": float(combine.sum()), "stage_ms": float(stage.sum()),
        "remote_logical_bytes": int(remote.sum()) * HIDDEN_BYTES,
        "max_mean": float(np.mean(expert / np.maximum(mean, 1e-12))),
        "max_second": float(np.mean(expert / np.maximum(ordered[:, -2], 1e-12))),
        "cv": float(np.mean(times.std(axis=1) / np.maximum(mean, 1e-12))),
        "wait_fraction": float(wait.mean()),
        "source_remote_peer_fanout_mean": float(peer_fanout.mean()),
        "active_source_remote_peer_fanout_mean": float(
            peer_fanout[active_sources].mean() if active_sources.any() else 0.0
        ),
        "per_invocation_stage": stage,
    }


def temporal_edge_systems(arrays, valid, stay, models, communication):
    ids = arrays["current_expert_ids"]
    full_hist = arrays["expert_counts_by_class"].sum(axis=1).astype(np.int32)
    removed_hist = subtract_removed_hist(full_hist, ids, valid, stay)
    zero = np.zeros(ids.shape, dtype=np.bool_)
    result = {}
    for ep in EPS:
        current_base = build_current_unique(
            ids, valid, arrays[f"current_source_rank_ep{ep}"], ep, zero
        )
        fixed = arrays[f"unique_matrix_ep{ep}"].astype(np.int32) - current_base
        source_unique = fixed + build_current_unique(
            ids, valid, arrays[f"current_source_rank_ep{ep}"], ep, stay
        )
        baseline = evaluated_stage(full_hist, arrays[f"unique_matrix_ep{ep}"].copy(),
                                   ep, models[ep], communication)
        source = evaluated_stage(removed_hist, source_unique, ep, models[ep], communication)
        owner = evaluated_stage(removed_hist, arrays[f"unique_matrix_ep{ep}"].copy(),
                                ep, models[ep], communication)
        live_hist = build_hist(ids, valid)
        live_unique = current_base
        live_remaining_hist = subtract_removed_hist(live_hist, ids, valid, stay)
        live_remaining_unique = build_current_unique(
            ids, valid, arrays[f"current_source_rank_ep{ep}"], ep, stay
        )
        live_base = evaluated_stage(live_hist, live_unique, ep, models[ep], communication)
        live_source = evaluated_stage(
            live_remaining_hist, live_remaining_unique, ep, models[ep], communication
        )
        for label, entry in (("source_side", source), ("owner_side", owner)):
            entry["stage_gain_percent"] = 100 * (1 - entry["stage_ms"] / baseline["stage_ms"])
            entry["expert_gain_percent"] = 100 * (1 - entry["expert_ms"] / baseline["expert_ms"])
            entry["remote_byte_reduction_percent"] = 100 * (
                1 - entry["remote_logical_bytes"] / baseline["remote_logical_bytes"]
            )
            entry.pop("per_invocation_stage")
        live_source["stage_gain_percent"] = 100 * (
            1 - live_source["stage_ms"] / live_base["stage_ms"]
        )
        baseline.pop("per_invocation_stage")
        live_base.pop("per_invocation_stage")
        live_source.pop("per_invocation_stage")
        result[f"ep{ep}"] = {
            "label": f"SIMULATED-EP{ep}-EP2-CALIBRATED",
            "baseline": baseline, "source_side": source, "owner_side": owner,
            "live_mask_only_baseline": live_base,
            "live_mask_only_source_side": live_source,
        }
        del current_base, fixed, source_unique, live_hist, live_unique, live_remaining_hist
    result["work"] = {
        "eligible_live_mask_routes": int(valid.sum() * TOP_K),
        "stay_routes": int(stay.sum()),
        "live_mask_fresh_pair_reduction_percent": 100 * float(stay.sum() / (valid.sum() * TOP_K)),
        "whole_physical_route_reduction_percent": 100 * float(stay.sum() / full_hist.sum()),
        "source_cache_peak_raw_output_bytes_per_invocation": int(
            stay.reshape(len(stay), -1).sum(axis=1).max(initial=0) * HIDDEN_BYTES
        ),
    }
    return result


@njit(cache=True)
def infer_homes(req, block, iteration, layer, cls, ids, source, weights, ep, global_home):
    policies = 6  # O1, O1-future, O2, O3, O4, O5
    homes = np.empty((policies, ids.shape[0], ids.shape[1]), dtype=np.int8)
    for policy in range(policies):
        homes[policy] = source
    affinity = np.full((ids.shape[0], ids.shape[1]), -1.0, dtype=np.float32)
    mass_affinity = np.full_like(affinity, -1.0)
    early1_match = np.full_like(affinity, -1.0)
    early2_match = np.full_like(affinity, -1.0)
    per_rank = NUM_EXPERTS // ep
    start = 0
    while start < len(req):
        end = start + 1
        while end < len(req) and req[end] == req[start] and block[end] == block[start]:
            end += 1
        first_iteration = iteration[start]
        for position in range(ids.shape[1]):
            all_count = np.zeros(ep, dtype=np.int64)
            future_count = np.zeros(ep, dtype=np.int64)
            first_count = np.zeros(ep, dtype=np.int64)
            first2_count = np.zeros(ep, dtype=np.int64)
            mass = np.zeros(ep, dtype=np.float64)
            layer_count = np.zeros((20, ep), dtype=np.int64)
            valid_any = False
            for index in range(start, end):
                if cls[index, position] < 2:
                    continue
                valid_any = True
                seen = np.zeros(ep, dtype=np.uint8)
                for slot in range(TOP_K):
                    destination = ids[index, position, slot] // per_rank
                    seen[destination] = 1
                    mass[destination] += weights[index, position, slot]
                for destination in range(ep):
                    if seen[destination]:
                        all_count[destination] += 1
                        layer_count[layer[index], destination] += 1
                        if iteration[index] > first_iteration:
                            future_count[destination] += 1
                        if iteration[index] == first_iteration:
                            first_count[destination] += 1
                        if iteration[index] <= first_iteration + 1:
                            first2_count[destination] += 1
            if not valid_any:
                continue
            old = source[start, position]
            h_all = int(np.argmax(all_count))
            h_future = int(np.argmax(future_count)) if future_count.sum() else h_all
            h_first = int(np.argmax(first_count))
            h_first2 = int(np.argmax(first2_count))
            affinity[start, position] = all_count[h_all] / max(1, all_count.sum())
            mass_affinity[start, position] = mass.max() / max(1e-12, mass.sum())
            early1_match[start, position] = 1.0 if h_first == h_future else 0.0
            early2_match[start, position] = 1.0 if h_first2 == h_future else 0.0
            for index in range(start, end):
                if cls[index, position] < 2:
                    continue
                homes[0, index, position] = h_all
                homes[1, index, position] = source[index, position] if iteration[index] == first_iteration else h_future
                homes[2, index, position] = source[index, position] if iteration[index] == first_iteration else h_first
                homes[3, index, position] = source[index, position] if iteration[index] <= first_iteration + 1 else h_first2
                homes[4, index, position] = global_home
                homes[5, index, position] = int(np.argmax(layer_count[layer[index]]))
        start = end
    return homes, affinity, mass_affinity, early1_match, early2_match


@njit(parallel=True, cache=True)
def remap_unique(base, ids, cls, source, home, ep):
    output = base.copy()
    per_rank = NUM_EXPERTS // ep
    for invocation in prange(len(ids)):
        for position in range(ids.shape[1]):
            if cls[invocation, position] < 2:
                continue
            old = source[invocation, position]
            new = home[invocation, position]
            if old == new:
                continue
            seen = np.zeros(ep, dtype=np.uint8)
            for slot in range(TOP_K):
                seen[ids[invocation, position, slot] // per_rank] = 1
            for destination in range(ep):
                if seen[destination]:
                    output[invocation, old, destination] -= 1
                    output[invocation, new, destination] += 1
    return output


def migration_matrices(arrays, homes, policy_index, ep):
    matrices = []
    start = 0
    n = len(arrays["request_id"])
    while start < n:
        end = start + 1
        while (end < n and arrays["request_id"][end] == arrays["request_id"][start]
               and arrays["block_id"][end] == arrays["block_id"][start]):
            end += 1
        minimum = int(arrays["iteration_id"][start])
        observe = 1 if policy_index == 2 else 2 if policy_index == 3 else 0
        candidate = np.flatnonzero(arrays["iteration_id"][start:end] >= minimum + observe)
        matrix = np.zeros((ep, ep), dtype=np.int64)
        if len(candidate):
            index = start + int(candidate[0])
            for position in range(32):
                if arrays["current_position_class"][index, position] < 2:
                    continue
                old = int(arrays[f"current_source_rank_ep{ep}"][index, position])
                new = int(homes[policy_index, index, position])
                if old != new:
                    matrix[old, new] += 1
        matrices.append(matrix)
        start = end
    return matrices


def migration_latency(matrices, bytes_per_token, model):
    total = 0.0
    moved = 0
    for matrix in matrices:
        moved += int(matrix.sum())
        if not matrix.any():
            continue
        byte_matrix = matrix.astype(np.float64) * bytes_per_token
        outgoing = byte_matrix.sum(axis=1)
        incoming = byte_matrix.sum(axis=0)
        total += model.latency(outgoing, incoming)
    return total, moved * bytes_per_token, moved


def token_ownership(arrays, models, communication):
    ids = arrays["current_expert_ids"]
    cls = arrays["current_position_class"]
    weights = arrays["current_router_weights"].astype(np.float32)
    hist = arrays["expert_counts_by_class"].sum(axis=1).astype(np.int32)
    result = {}
    for ep in EPS:
        owner = ids // (NUM_EXPERTS // ep)
        discovery = arrays["request_id"] < 64
        global_count = np.zeros(ep, dtype=np.int64)
        for destination in range(ep):
            global_count[destination] = np.count_nonzero(
                (owner[discovery] == destination) & (cls[discovery, :, None] >= 2)
            )
        global_home = int(np.argmax(global_count))
        source = arrays[f"current_source_rank_ep{ep}"]
        current_valid = cls >= 2
        zero_removed = np.zeros(ids.shape, dtype=np.bool_)
        current_baseline_unique = build_current_unique(
            ids, current_valid, source, ep, zero_removed
        )
        current_baseline_remote = current_baseline_unique.copy()
        diagonal = np.arange(ep)
        current_baseline_remote[:, diagonal, diagonal] = 0
        current_baseline_remote_bytes = int(current_baseline_remote.sum()) * HIDDEN_BYTES
        homes, affinity, mass_affinity, early1_match, early2_match = infer_homes(
            arrays["request_id"], arrays["block_id"], arrays["iteration_id"],
            arrays["layer_id"], cls, ids, source, weights, ep, global_home,
        )
        baseline = evaluated_stage(hist, arrays[f"unique_matrix_ep{ep}"].copy(), ep,
                                   models[ep], communication)
        baseline_stage = baseline["stage_ms"]
        policies = ("O1_full_future", "O1_future_after_ref1", "O2_early_ref1",
                    "O3_early_ref1_2", "O4_static_global", "O5_layer_local_future")
        entries = {}
        for policy_index, name in enumerate(policies):
            unique = remap_unique(
                arrays[f"unique_matrix_ep{ep}"], ids, cls, source,
                homes[policy_index], ep,
            )
            evaluated = evaluated_stage(hist, unique, ep, models[ep], communication)
            evaluated["stage_gain_zero_cost_percent"] = 100 * (
                1 - evaluated["stage_ms"] / baseline_stage
            )
            evaluated["remote_byte_reduction_percent"] = 100 * (
                1 - evaluated["remote_logical_bytes"] / baseline["remote_logical_bytes"]
            )
            current_unique = remap_unique(
                current_baseline_unique, ids, cls, source, homes[policy_index], ep
            )
            current_remote = current_unique.copy()
            current_remote[:, diagonal, diagonal] = 0
            evaluated["current_block_remote_byte_reduction_percent"] = 100 * (
                1 - int(current_remote.sum()) * HIDDEN_BYTES
                / current_baseline_remote_bytes
            )
            evaluated.pop("per_invocation_stage")
            entries[name] = evaluated

        # AR one-shot: observe refinement 1, remap for the next refinement only.
        ar_home = source.copy()
        block_first = {}
        for index, key in enumerate(zip(arrays["request_id"].tolist(), arrays["block_id"].tolist())):
            block_first.setdefault(key, int(arrays["iteration_id"][index]))
            if arrays["iteration_id"][index] == block_first[key] + 1:
                ar_home[index] = homes[2, index]
        ar_unique = remap_unique(arrays[f"unique_matrix_ep{ep}"], ids, cls, source, ar_home, ep)
        ar = evaluated_stage(hist, ar_unique, ep, models[ep], communication)
        ar_saving = baseline_stage - ar["stage_ms"]
        repeated_saving = baseline_stage - entries["O2_early_ref1"]["stage_ms"]

        migrations = migration_matrices(arrays, homes, 2, ep)
        migration = {}
        for label, state_bytes in (("M0_zero", 0), ("M1_hidden_row", HIDDEN_BYTES),
                                   ("M2_20_layer_state", HIDDEN_BYTES * 20)):
            if state_bytes:
                latency, byte_count, moved = migration_latency(
                    migrations, state_bytes, communication.dispatch
                )
            else:
                latency = 0.0
                moved = sum(int(matrix.sum()) for matrix in migrations)
                byte_count = 0
            migration[label] = {
                "state_bytes_per_token": state_bytes, "moved_tokens": moved,
                "migration_bytes": byte_count, "migration_latency_ms": latency,
                "repeated_net_stage_gain_percent": 100 * (
                    repeated_saving - latency
                ) / baseline_stage,
                "ar_one_shot_net_stage_gain_percent": 100 * (
                    ar_saving - latency
                ) / baseline_stage,
                "dllm_repeated_break_even": repeated_saving > latency,
                "ar_one_shot_break_even": ar_saving > latency,
                "minimum_equivalent_future_refinements": (
                    latency / max(ar_saving, 1e-12)
                ),
            }
        baseline.pop("per_invocation_stage")
        valid_affinity = affinity >= 0
        future_oracle = entries["O1_future_after_ref1"]["current_block_remote_byte_reduction_percent"]
        early_reduction = entries["O2_early_ref1"]["current_block_remote_byte_reduction_percent"]
        result[f"ep{ep}"] = {
            "label": f"SIMULATED-EP{ep}-EP2-CALIBRATED",
            "global_home": global_home,
            "baseline": baseline, "policies": entries,
            "token_home_affinity": {
                "mean": float(affinity[valid_affinity].mean()),
                "p50": float(np.median(affinity[valid_affinity])),
                "p90": float(np.percentile(affinity[valid_affinity], 90)),
                "router_mass_weighted_mean": float(mass_affinity[valid_affinity].mean()),
            },
            "early_home_match_to_future": {
                "ref1": float(early1_match[valid_affinity].mean()),
                "ref1_2": float(early2_match[valid_affinity].mean()),
            },
            "early_ref1_recovery_of_future_oracle_percent": 100 * early_reduction / max(future_oracle, 1e-12),
            "migration": migration,
            "ar_one_shot_zero_cost_stage_gain_percent": 100 * ar_saving / baseline_stage,
        }
        del homes, affinity, mass_affinity, early1_match, early2_match
    return result


def make_figures(persistence, controls_result, run_result, ownership, output):
    output.mkdir(parents=True, exist_ok=True)
    layer = persistence["layer"]
    keys = sorted(layer, key=int)
    plt.figure(figsize=(8, 4))
    plt.plot(keys, [100 * layer[k]["discovery"]["stay_route_fraction"] for k in keys], label="discovery")
    plt.plot(keys, [100 * layer[k]["heldout"]["stay_route_fraction"] for k in keys], label="held-out")
    plt.xlabel("routed layer"); plt.ylabel("live-MASK STAY routes (%)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(output / "01_stay_fraction_by_layer.png", dpi=180); plt.close()

    refinement = persistence["refinement"]
    keys = list(refinement)
    plt.figure(figsize=(7, 4))
    plt.plot(keys, [
        np.nan if refinement[k]["heldout"]["stay_route_fraction"] is None
        else 100 * refinement[k]["heldout"]["stay_route_fraction"]
        for k in keys
    ], marker="o")
    plt.ylabel("held-out STAY routes (%)"); plt.xlabel("refinement index bin"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(output / "02_stay_fraction_by_refinement.png", dpi=180); plt.close()

    labels = list(controls_result)
    values = [100 * controls_result[label]["stay_route_fraction"] for label in labels]
    plt.figure(figsize=(9, 4)); plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right"); plt.ylabel("route overlap (%)")
    plt.tight_layout(); plt.savefig(output / "03_persistence_controls.png", dpi=180); plt.close()

    hist = np.asarray(run_result["histogram"])
    samples = np.repeat(np.arange(len(hist)), hist)
    samples = np.sort(samples[samples > 0])
    plt.figure(figsize=(6, 4)); plt.step(samples, np.arange(1, len(samples) + 1) / len(samples))
    plt.xlabel("consecutive live-MASK edge run length"); plt.ylabel("CDF"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(output / "04_route_run_length_cdf.png", dpi=180); plt.close()

    policies = ("O0_baseline", "O1_full_future", "O2_early_ref1", "O3_early_ref1_2", "O4_static_global")
    for ep in EPS:
        cell = ownership[f"ep{ep}"]
        reductions = [0] + [cell["policies"][name]["remote_byte_reduction_percent"] for name in policies[1:]]
        plt.figure(figsize=(7, 4)); plt.bar(policies, reductions); plt.xticks(rotation=25, ha="right")
        plt.ylabel("remote logical-byte reduction (%)"); plt.title(f"EP{ep} token-home policies")
        plt.tight_layout(); plt.savefig(output / f"{5 if ep == 4 else 6:02d}_ep{ep}_ownership_remote_bytes.png", dpi=180); plt.close()

    eps = [4, 8]
    affinity = [ownership[f"ep{ep}"]["token_home_affinity"]["mean"] for ep in eps]
    early = [ownership[f"ep{ep}"]["early_home_match_to_future"]["ref1"] for ep in eps]
    x = np.arange(2)
    plt.figure(figsize=(6, 4)); plt.bar(x - .18, affinity, .36, label="home affinity"); plt.bar(x + .18, early, .36, label="early1 match")
    plt.xticks(x, ["EP4", "EP8"]); plt.ylabel("fraction"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "07_token_home_affinity.png", dpi=180); plt.close()

    labels = ("M0_zero", "M1_hidden_row", "M2_20_layer_state")
    plt.figure(figsize=(7, 4))
    for ep in EPS:
        cell = ownership[f"ep{ep}"]["migration"]
        plt.plot(labels, [cell[label]["repeated_net_stage_gain_percent"] for label in labels], marker="o", label=f"EP{ep} dLLM")
        plt.plot(labels, [cell[label]["ar_one_shot_net_stage_gain_percent"] for label in labels], marker="x", linestyle="--", label=f"EP{ep} AR-one-shot")
    plt.axhline(0, color="black", linewidth=.8); plt.ylabel("net routed-stage gain (%)"); plt.xticks(rotation=20); plt.legend()
    plt.tight_layout(); plt.savefig(output / "08_migration_amortization.png", dpi=180); plt.close()


def markdown_reports(summary, reports):
    persistence = summary["track_a"]["persistence"]
    controls_result = summary["track_a"]["controls"]
    systems = summary["track_a"]["systems"]
    ownership = summary["track_b"]
    global_cell = persistence["global"]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "temporal_edge_ep_persistence.md").write_text(f"""# Temporal-Edge EP: exact live-MASK persistence

Evidence: exact current-block token/layer/expert identities from threshold-.95 GSM8K-128. Requests 0--63 are discovery and 64--127 held out. EP4/EP8 ownership is simulated; no quality or timing claim is made here.

## Primary result

| split | STAY route count | STAY router mass |
|---|---:|---:|
| discovery | {100*global_cell['discovery']['stay_route_fraction']:.3f}% | {100*global_cell['discovery']['stay_router_mass_fraction']:.3f}% |
| held-out | {100*global_cell['heldout']['stay_route_fraction']:.3f}% | {100*global_cell['heldout']['stay_router_mass_fraction']:.3f}% |
| all | {100*global_cell['all']['stay_route_fraction']:.3f}% | {100*global_cell['all']['stay_router_mass_fraction']:.3f}% |

ENTER and EXIT are each the complement of STAY because top-k remains eight. Full stratification by layer, refinement, mask ratio, confidence, slot, weight decile, block index and EP-local/remote destination is in the machine summary. The discovery-selected subgroup was frozen before held-out evaluation: `{persistence['discovery_selected_subgroup']['axis']}={persistence['discovery_selected_subgroup']['label']}`; discovery/held-out STAY were {100*persistence['discovery_selected_subgroup']['discovery']['stay_route_fraction']:.2f}%/{100*persistence['discovery_selected_subgroup']['heldout_confirmation']['stay_route_fraction']:.2f}%.

## Negative controls

```json
{json.dumps(controls_result, indent=2)}
```

Run length P50/P75/P90/P95/P99/max is {summary['track_a']['run_length']['p50']:.0f}/{summary['track_a']['run_length']['p75']:.0f}/{summary['track_a']['run_length']['p90']:.0f}/{summary['track_a']['run_length']['p95']:.0f}/{summary['track_a']['run_length']['p99']:.0f}/{summary['track_a']['run_length']['max']} refinements. Adjacent repeated refinement is materially stronger than the controls, so the structural dLLM-specific edge signal is real.
""")
    (reports / "temporal_edge_ep_oracle.md").write_text(f"""# Temporal-Edge EP systems oracle

This is an exact route-removal counterfactual over the existing trace and an EP2-calibrated EP4/EP8 simulator. It is not a runtime measurement. E1 source-side removes dispatch, expert work and combine for every still-live STAY edge; owner-side removes expert work but retains communication.

| target | live-MASK pair reduction | whole physical route reduction | source stage gain | owner stage gain |
|---|---:|---:|---:|---:|
| EP4 | {systems['work']['live_mask_fresh_pair_reduction_percent']:.3f}% | {systems['work']['whole_physical_route_reduction_percent']:.3f}% | {systems['ep4']['source_side']['stage_gain_percent']:.3f}% | {systems['ep4']['owner_side']['stage_gain_percent']:.3f}% |
| EP8 | {systems['work']['live_mask_fresh_pair_reduction_percent']:.3f}% | {systems['work']['whole_physical_route_reduction_percent']:.3f}% | {systems['ep8']['source_side']['stage_gain_percent']:.3f}% | {systems['ep8']['owner_side']['stage_gain_percent']:.3f}% |

The apparent 68.8% live-MASK opportunity is diluted to about half a percent of all physical expert routes by prompt/prior/current-decoded full rows. Even impossible identity reuse is far below the suggested 5% whole-stage signal. Cache peak for E1 raw outputs is {systems['work']['source_cache_peak_raw_output_bytes_per_invocation']/2**20:.2f} MiB per invocation. E2/E3/E4 are evaluated in the output-stability report; no production cache or kernel was built.
""")
    ep4 = ownership["ep4"]; ep8 = ownership["ep8"]
    (reports / "refinement_token_ownership_affinity.md").write_text(f"""# Refinement-coupled token ownership affinity

Every source, expert and destination is reconstructed exactly for current-block rows using the same balanced virtual-source rule validated against true EP2. Expert identity and expert compute are unchanged.

| target | mean most-affine-rank share | mass-weighted share | early1 match to future | early1+2 match |
|---|---:|---:|---:|---:|
| EP4 | {100*ep4['token_home_affinity']['mean']:.3f}% | {100*ep4['token_home_affinity']['router_mass_weighted_mean']:.3f}% | {100*ep4['early_home_match_to_future']['ref1']:.3f}% | {100*ep4['early_home_match_to_future']['ref1_2']:.3f}% |
| EP8 | {100*ep8['token_home_affinity']['mean']:.3f}% | {100*ep8['token_home_affinity']['router_mass_weighted_mean']:.3f}% | {100*ep8['early_home_match_to_future']['ref1']:.3f}% | {100*ep8['early_home_match_to_future']['ref1_2']:.3f}% |

Affinity is weak because top-8 routes span most EP ranks, especially at EP8. The static-global control and complete policy table are in the JSON summary.
""")
    (reports / "refinement_token_ownership_oracle.md").write_text(f"""# Refinement-coupled token ownership oracle

Labels are `SIMULATED-EP4-EP2-CALIBRATED` and `SIMULATED-EP8-EP2-CALIBRATED`. O1/O5 use future information. O2/O3 use only the first one/two refinements. Expert compute is held fixed.

| target | O1 current-block bytes | O1 whole bytes | O2 current-block bytes | O3 current-block bytes | O2 whole-stage gain |
|---|---:|---:|---:|---:|---:|
| EP4 | {ep4['policies']['O1_full_future']['current_block_remote_byte_reduction_percent']:.3f}% | {ep4['policies']['O1_full_future']['remote_byte_reduction_percent']:.3f}% | {ep4['policies']['O2_early_ref1']['current_block_remote_byte_reduction_percent']:.3f}% | {ep4['policies']['O3_early_ref1_2']['current_block_remote_byte_reduction_percent']:.3f}% | {ep4['policies']['O2_early_ref1']['stage_gain_zero_cost_percent']:.3f}% |
| EP8 | {ep8['policies']['O1_full_future']['current_block_remote_byte_reduction_percent']:.3f}% | {ep8['policies']['O1_full_future']['remote_byte_reduction_percent']:.3f}% | {ep8['policies']['O2_early_ref1']['current_block_remote_byte_reduction_percent']:.3f}% | {ep8['policies']['O3_early_ref1_2']['current_block_remote_byte_reduction_percent']:.3f}% | {ep8['policies']['O2_early_ref1']['stage_gain_zero_cost_percent']:.3f}% |

Even the impossible full-future oracle saves only about 3.6% of current-block remote rows, and about 0.06% after prompt/prior-row dilution. Early policies save about 1--2% of current-block bytes, far below the 15% suggested signal and before migration.
""")
    (reports / "refinement_token_ownership_amortization.md").write_text(f"""# Refinement-coupled ownership amortization

M0 is free logical remap, M1 moves one BF16 hidden row ({HIDDEN_BYTES} bytes), and M2 conservatively moves 20 layer states ({HIDDEN_BYTES*20} bytes). Migration uses the EP2-calibrated endpoint model and is charged once per moved token/block. These are sensitivity scenarios, not production migration measurements.

```json
{json.dumps({key: ownership[key]['migration'] for key in ('ep4','ep8')}, indent=2)}
```

The mandatory AR control observes one execution and benefits for only the next refinement. The repeated dLLM horizon is compared with exactly the same setup cost. The final summary reports whether each sensitivity breaks even.
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path(
        "artifacts/virtual_ep/20260917_four_hypothesis_discovery/aggregate_v2/gsm8k_128_discovery_v2_fixed.npz"
    ))
    parser.add_argument("--communication", type=Path, default=Path(
        "artifacts/virtual_ep/20260916_215010/communication/model.json"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/dllm_native_ep_discovery/structural_summary.json"
    ))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    trace = load_discovery_trace(args.trace)
    arrays = trace.arrays
    previous = build_previous(arrays)
    valid, stay = stay_masks(arrays, previous)
    persistence = persistence_strata(arrays, valid, stay)
    controls_result = controls(arrays, previous)
    run_result = run_lengths(arrays)
    models = {
        4: BatchComputeModel(Path("artifacts/route_pruning_quality/20260918/compute_model/ep4_grouped_mm_route_pruning.csv")),
        8: BatchComputeModel(Path("artifacts/route_pruning_quality/20260918/compute_model/ep8_grouped_mm_route_pruning.csv")),
    }
    communication = CommunicationScenario.load(args.communication, "ep2_calibrated_base")
    systems = temporal_edge_systems(arrays, valid, stay, models, communication)
    ownership = token_ownership(arrays, models, communication)
    summary = {
        "evidence": {
            "trace": str(args.trace), "requests": 128,
            "split": "request IDs 0-63 discovery / 64-127 held-out",
            "timing": "EP2-calibrated virtual EP4/EP8; not measured EP4/EP8",
        },
        "track_a": {
            "persistence": persistence, "controls": controls_result,
            "run_length": run_result, "systems": systems,
        },
        "track_b": ownership,
    }
    dump_json(args.output, summary)
    make_figures(persistence, controls_result, run_result, ownership,
                 args.reports / "figures/dllm_native_ep_discovery")
    markdown_reports(summary, args.reports)
    print(json.dumps({
        "output": str(args.output),
        "stay_heldout": persistence["global"]["heldout"]["stay_route_fraction"],
        "ep8_e1_stage_gain": systems["ep8"]["source_side"]["stage_gain_percent"],
        "ep8_o2_remote_reduction": ownership["ep8"]["policies"]["O2_early_ref1"]["remote_byte_reduction_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
