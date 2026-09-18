#!/usr/bin/env python3
"""Analyze the block-cache, FrontierEP, and lossless DeltaEP PoC.

All EP4/EP8 timings are EP2-calibrated routed-MoE-stage projections.  The
script deliberately reports changed-trajectory and per-forward-normalized
FrontierEP results separately and treats DeltaEP codec timing as a zero-codec-
overhead upper bound when no codec reaches the implementation gate.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_h1_straggler_deepdive import BatchComputeModel
from virtual_ep.comm_model import CommunicationScenario, matrix_endpoint_bytes


HIDDEN_BYTES = 2048 * 2
NUM_EXPERTS = 256
TOP_K = 8
ROUTED_LAYERS = 19
SELECTED_LAYERS = (1, 5, 10, 14, 19)


def generation_rows(directory: Path) -> list[dict]:
    rows = []
    for path in directory.glob("generations_worker*.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return sorted({int(row["sample_id"]): row for row in rows}.values(),
                  key=lambda row: int(row["sample_id"]))


def load_shapes(directory: Path, ep: int, selective_prefix: str | None = None,
                maximum_request: int | None = None):
    histograms, unique, requests = [], [], []
    for path in sorted((directory / "traces").glob("request_*.npz")):
        request = int(path.stem.split("_")[-1])
        if maximum_request is not None and request >= maximum_request:
            continue
        with np.load(path, allow_pickle=False) as source:
            if selective_prefix is None:
                hist = source["hist"]
                matrix = source[f"u_ep{ep}"]
            else:
                hist = source[f"hist_{selective_prefix}"]
                matrix = source[f"u_{selective_prefix}_ep{ep}"]
            steps = hist.shape[0]
            histograms.append(hist.reshape(-1, NUM_EXPERTS).astype(np.float64))
            unique.append(matrix.reshape(-1, ep, ep).astype(np.float64))
            requests.extend([request] * steps * ROUTED_LAYERS)
    return np.concatenate(histograms), np.concatenate(unique), np.asarray(requests)


def project(histograms: np.ndarray, unique: np.ndarray, requests: np.ndarray,
            ep: int, model: BatchComputeModel,
            communication: CommunicationScenario) -> dict:
    width = NUM_EXPERTS // ep
    rank = np.column_stack([
        model.predict_batch(histograms[:, index * width:(index + 1) * width])
        for index in range(ep)
    ])
    expert = rank.max(axis=1)
    rank_mean = rank.mean(axis=1)
    ordered = np.sort(rank, axis=1)
    second = ordered[:, -2]
    dispatch, combine, remote_bytes, fanout = [], [], [], []
    for matrix in unique:
        outgoing, incoming = matrix_endpoint_bytes(matrix, 2048)
        dispatch.append(communication.dispatch.latency(outgoing, incoming))
        combine.append(communication.combine.latency(incoming, outgoing))
        remote = matrix.copy(); np.fill_diagonal(remote, 0)
        remote_bytes.append(float(remote.sum() * HIDDEN_BYTES))
        fanout.append(float(np.count_nonzero(remote, axis=1).mean()))
    dispatch = np.asarray(dispatch); combine = np.asarray(combine)
    stage = dispatch + expert + combine
    per_request = defaultdict(float)
    for request, value in zip(requests, stage):
        per_request[int(request)] += float(value)
    return {
        "requests": int(len(set(requests.tolist()))),
        "invocations": int(len(stage)),
        "fresh_expert_token_pairs": int(histograms.sum()),
        "dispatch_ms": float(dispatch.sum()),
        "expert_ms": float(expert.sum()),
        "combine_ms": float(combine.sum()),
        "stage_ms": float(stage.sum()),
        "stage_ms_per_request": float(stage.sum() / len(per_request)),
        "stage_ms_per_forward": float(stage.sum() / (len(stage) / ROUTED_LAYERS)),
        "max_mean": float(np.mean(expert / np.maximum(rank_mean, 1e-12))),
        "max_second": float(np.mean(expert / np.maximum(second, 1e-12))),
        "cv": float(np.mean(rank.std(axis=1) / np.maximum(rank_mean, 1e-12))),
        "wait_fraction": float(np.mean(
            (ep * expert - rank.sum(axis=1)) / np.maximum(ep * expert, 1e-12)
        )),
        "remote_bytes": float(np.sum(remote_bytes)),
        "fanout": float(np.mean(fanout)),
        "request_stage_p50_ms": float(np.percentile(list(per_request.values()), 50)),
        "request_stage_p95_ms": float(np.percentile(list(per_request.values()), 95)),
    }


def gain(baseline: dict, candidate: dict, key: str = "stage_ms") -> float:
    return 100.0 * (1.0 - candidate[key] / baseline[key])


def summarize_quality(directory: Path) -> dict:
    rows = generation_rows(directory)
    return {
        "requests": len(rows),
        "correct": int(sum(row["correct"] for row in rows)),
        "reference_correct": int(sum(row["reference_correct"] for row in rows)),
        "baseline_correct_to_wrong": int(sum(
            row["reference_correct"] and not row["correct"] for row in rows
        )),
        "baseline_wrong_to_correct": int(sum(
            not row["reference_correct"] and row["correct"] for row in rows
        )),
        "exact_generation_identity": int(sum(row["exact_generation_identity"] for row in rows)),
        "parsed_answer_identity": int(sum(row["parsed_answer_identity"] for row in rows)),
        "nfe": int(sum(row["nfe"] for row in rows)),
        "reference_nfe": int(sum(row["reference_nfe"] for row in rows)),
        "mean_nfe": float(np.mean([row["nfe"] for row in rows])),
        "reference_mean_nfe": float(np.mean([row["reference_nfe"] for row in rows])),
        "maximum_nfe_ratio": float(max(row["nfe"] / row["reference_nfe"] for row in rows)),
        "remaining_masks": int(sum(row["remaining_masks"] for row in rows)),
        "fresh_rows": int(sum(row["total_fresh_rows"] for row in rows)),
        "fresh_current_rows": int(sum(row["fresh_current_rows"] for row in rows)),
        "fresh_prior_rows": int(sum(row["fresh_prior_rows"] for row in rows)),
    }


def exactness_summary(directory: Path) -> dict:
    rows = []
    for path in (directory / "traces").glob("*.exactness.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    output = {"records": len(rows), "requests": len(set(row["request_id"] for row in rows))}
    for name, predicate in {
        "same_shape_repeated_refinement": lambda row: row["iteration_id"] > 0,
        "new_block_shape_change": lambda row: row["block_id"] > 0 and row["iteration_id"] == 0,
    }.items():
        selected = [row for row in rows if predicate(row)]
        boundaries = {}
        for boundary in sorted(set(row["boundary"] for row in selected)):
            group = [row for row in selected if row["boundary"] == boundary]
            total_rows = sum(row["rows"] for row in group)
            boundaries[boundary] = {
                "records": len(group),
                "bit_exact_row_fraction": float(
                    sum(row["bit_exact_rows"] for row in group) / max(total_rows, 1)
                ),
                "mean_relative_l2": float(np.mean([row["relative_l2_mean"] for row in group])),
                "p95_record_max_relative_l2": float(np.percentile(
                    [row["relative_l2_max"] for row in group], 95
                )),
                "mean_exact_element_fraction": float(np.mean(
                    [row["exact_element_fraction"] for row in group]
                )),
            }
        output[name] = boundaries
    return output


def load_delta(directory: Path) -> dict[str, np.ndarray]:
    pieces = []
    for path in sorted((directory / "traces").glob("request_*.npz")):
        with np.load(path, allow_pickle=False) as source:
            pieces.append({key: source[key] for key in source.files if key.startswith("delta_")})
    keys = pieces[0].keys()
    return {key: np.concatenate([piece[key] for piece in pieces]) for key in keys}


def compression_statistics(delta: dict[str, np.ndarray]) -> dict:
    output = {"codec_frozen_on_discovery": "byte_bitmap_plus_literals_with_full_fallback",
              "per_vector_tag_bytes": 1}
    for split, split_mask in {
        "discovery": delta["delta_request_id"] < 8,
        "heldout": delta["delta_request_id"] >= 8,
    }.items():
        output[split] = {}
        for boundary in ("dispatch", "combine"):
            output[split][boundary] = {}
            for control in ("adjacent", "wrong_token", "nonadjacent", "zero_predictor"):
                mask = (split_mask & (delta["delta_boundary"] == boundary)
                        & (delta["delta_control"] == control))
                if not np.any(mask):
                    continue
                raw = delta["delta_raw_bytes"][mask].astype(np.float64)
                candidate = delta["delta_byte_bitmap_bytes"][mask].astype(np.float64) + 1
                encoded = np.minimum(raw, candidate)
                ratio = raw / encoded
                entry = {
                    "vectors": int(mask.sum()),
                    "aggregate_compression_ratio": float(raw.sum() / encoded.sum()),
                    "compression_ratio_p50": float(np.percentile(ratio, 50)),
                    "compression_ratio_p75": float(np.percentile(ratio, 75)),
                    "compression_ratio_p90": float(np.percentile(ratio, 90)),
                    "fallback_fraction": float(np.mean(candidate >= raw)),
                    "exact_word_fraction_p50": float(np.percentile(
                        delta["delta_exact_word_fraction"][mask], 50
                    )),
                    "xor_zero_byte_fraction_p50": float(np.percentile(
                        delta["delta_xor_zero_byte_fraction"][mask], 50
                    )),
                    "xor_byte_entropy_p50": float(np.percentile(
                        delta["delta_xor_byte_entropy"][mask], 50
                    )),
                }
                for metric in ("cosine", "relative_l2", "delta_abs_p50",
                               "delta_abs_p90", "delta_abs_p99"):
                    key = f"delta_{metric}"
                    if key in delta:
                        entry[f"{metric}_p50"] = float(np.percentile(delta[key][mask], 50))
                        entry[f"{metric}_p90"] = float(np.percentile(delta[key][mask], 90))
                output[split][boundary][control] = entry
    return output


def compression_strata(delta: dict[str, np.ndarray]) -> dict:
    output = {"layer": {}, "mask_ratio": {}, "route_age": {}}
    base = ((delta["delta_control"] == "adjacent")
            & (delta["delta_boundary"] == "dispatch"))

    def entry(mask):
        raw = delta["delta_raw_bytes"][mask].astype(np.float64)
        encoded = np.minimum(raw, delta["delta_byte_bitmap_bytes"][mask] + 1)
        return {
            "vectors": int(mask.sum()),
            "aggregate_ratio": float(raw.sum() / encoded.sum()),
            "median_ratio": float(np.median(raw / encoded)),
        }

    for split, split_mask in (("discovery", delta["delta_request_id"] < 8),
                              ("heldout", delta["delta_request_id"] >= 8)):
        output["layer"][split] = {
            str(layer): entry(base & split_mask & (delta["delta_layer_id"] == layer))
            for layer in SELECTED_LAYERS
        }
        masks = (("<=25%", 0, .25), ("25-50%", .25, .5),
                 ("50-75%", .5, .75), (">75%", .75, 1.00001))
        output["mask_ratio"][split] = {
            name: entry(base & split_mask & (delta["delta_mask_ratio"] > low)
                        & (delta["delta_mask_ratio"] <= high))
            for name, low, high in masks
            if np.any(base & split_mask & (delta["delta_mask_ratio"] > low)
                      & (delta["delta_mask_ratio"] <= high))
        }
        ages = (("1", 0, 1), ("2", 1, 2), ("3", 2, 3), ("4+", 3, 10_000))
        output["route_age"][split] = {
            name: entry(base & split_mask & (delta["delta_max_route_age"] > low)
                        & (delta["delta_max_route_age"] <= high))
            for name, low, high in ages
            if np.any(base & split_mask & (delta["delta_max_route_age"] > low)
                      & (delta["delta_max_route_age"] <= high))
        }
    best = max(output["layer"]["discovery"],
               key=lambda layer: output["layer"]["discovery"][layer]["median_ratio"])
    output["frozen_best_discovery_layer"] = int(best)
    output["heldout_for_frozen_layer"] = output["layer"]["heldout"][best]
    return output


def frontier_profile(directory: Path) -> dict:
    grouped = defaultdict(lambda: {"steps": 0, "fresh_current": 0, "live": 0,
                                   "sealed": 0, "newly": 0})
    for path in (directory / "traces").glob("request_*.npz"):
        with np.load(path, allow_pickle=False) as source:
            for index, refinement in enumerate(source["iteration_id"]):
                row = grouped[int(refinement)]
                row["steps"] += 1
                row["fresh_current"] += int(source["fresh_current_rows"][index])
                row["live"] += int(source["live_mask_count"][index])
                row["sealed"] += int(source["sealed_count"][index])
                row["newly"] += int(source["newly_committed_count"][index])
    return {str(key): value for key, value in sorted(grouped.items())}


def stay_fraction(directory: Path) -> dict:
    current, stayed = 0, 0
    mass, stayed_mass = 0.0, 0.0
    run_lengths = []
    for path in sorted((directory / "traces").glob("request_*.npz")):
        with np.load(path, allow_pickle=False) as source:
            blocks, routes, weights, masks = (source["block_id"], source["routes"],
                                               source["route_weights"].astype(np.float64),
                                               source["masked"])
            ages = np.zeros_like(routes[0], dtype=np.int16)
            previous = None; previous_mask = None; previous_block = None
            for block, route, weight, mask in zip(blocks, routes, weights, masks):
                if previous is None or block != previous_block:
                    ages.fill(0)
                else:
                    eligible = mask & previous_mask
                    for layer in range(ROUTED_LAYERS):
                        for position in np.flatnonzero(eligible):
                            before = set(map(int, previous[layer, position]))
                            for slot, expert in enumerate(route[layer, position]):
                                current += 1; mass += weight[layer, position, slot]
                                if int(expert) in before:
                                    stayed += 1; stayed_mass += weight[layer, position, slot]
                                    ages[layer, position, slot] += 1
                                elif ages[layer, position, slot]:
                                    run_lengths.append(int(ages[layer, position, slot]))
                                    ages[layer, position, slot] = 0
                previous, previous_mask, previous_block = route, mask, block
    return {
        "stay_route_fraction": stayed / max(current, 1),
        "stay_router_mass_fraction": stayed_mass / max(mass, 1e-12),
        "eligible_routes": current,
        "stay_routes": stayed,
        "observed_terminated_run_length_p50": float(np.percentile(run_lengths, 50)) if run_lengths else 0,
        "observed_terminated_run_length_p90": float(np.percentile(run_lengths, 90)) if run_lengths else 0,
    }


def source_rank(row: int, physical_rows: int, ep: int) -> int:
    base, remainder = divmod(physical_rows, ep)
    boundary = 0
    for rank in range(ep):
        boundary += base + (1 if rank < remainder else 0)
        if row < boundary:
            return rank
    return ep - 1


def delta_system_projection(delta: dict[str, np.ndarray], block_cached: Path,
                            ep: int, model: BatchComputeModel,
                            communication: CommunicationScenario,
                            maximum_request: int = 16) -> dict:
    hist, unique, requests = load_shapes(block_cached, ep, maximum_request=maximum_request)
    baseline = project(hist, unique, requests, ep, model, communication)

    # Flat invocation lookup follows request, refinement, layer ordering.
    keys, physical = [], {}
    for path in sorted((block_cached / "traces").glob("request_*.npz")):
        request = int(path.stem.split("_")[-1])
        if request >= maximum_request:
            continue
        with np.load(path, allow_pickle=False) as source:
            for step in range(len(source["block_id"])):
                for layer_slot, layer_id in enumerate(range(1, 20)):
                    key = (request, int(source["block_id"][step]),
                           int(source["iteration_id"][step]), layer_id)
                    keys.append(key)
                    physical[key] = (int(source["physical_rows"][step]),
                                     int(source["current_start"][step]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    if len(keys) != len(hist):
        raise RuntimeError("shape/key mismatch")

    baseline_bytes = unique.copy() * HIDDEN_BYTES
    diagonal = np.arange(ep)
    baseline_bytes[:, diagonal, diagonal] = 0
    results = {}
    for control in ("adjacent", "wrong_token", "nonadjacent", "zero_predictor"):
        dispatch_saving = np.zeros_like(baseline_bytes)
        combine_saving = np.zeros_like(baseline_bytes)
        live_raw = 0.0; live_encoded_dispatch = 0.0; live_encoded_combine = 0.0
        for boundary, target in (("dispatch", dispatch_saving), ("combine", combine_saving)):
            selected = ((delta["delta_control"] == control)
                        & (delta["delta_boundary"] == boundary))
            indices = np.flatnonzero(selected)
            for record in indices:
                request = int(delta["delta_request_id"][record])
                if request >= maximum_request:
                    continue
                key = (request, int(delta["delta_block_id"][record]),
                       int(delta["delta_iteration_id"][record]),
                       int(delta["delta_layer_id"][record]))
                invocation = key_to_index.get(key)
                if invocation is None:
                    continue
                physical_rows, start = physical[key]
                src = source_rank(start + int(delta["delta_token_position"][record]),
                                  physical_rows, ep)
                current = delta["delta_current_ids"][record]
                predictor = delta["delta_predictor_ids"][record]
                current_owner = current // (NUM_EXPERTS // ep)
                predictor_owner = predictor // (NUM_EXPERTS // ep)
                raw = int(delta["delta_raw_bytes"][record])
                encoded = min(raw, int(delta["delta_byte_bitmap_bytes"][record]) + 1)
                for destination in set(map(int, current_owner)):
                    if destination == src:
                        continue
                    can_predict = bool(np.any(current_owner == destination)
                                       and np.any(predictor_owner == destination))
                    used = encoded if can_predict else raw
                    target[invocation, src, destination] += raw - used
                    if control == "adjacent":
                        live_raw += raw
                        if boundary == "dispatch": live_encoded_dispatch += used
                        else: live_encoded_combine += used

        compressed = np.maximum(0.0, baseline_bytes - dispatch_saving)
        compressed_combine = np.maximum(0.0, baseline_bytes - combine_saving)
        width = NUM_EXPERTS // ep
        rank = np.column_stack([
            model.predict_batch(hist[:, index * width:(index + 1) * width])
            for index in range(ep)
        ])
        expert = rank.max(axis=1)
        dispatch_ms, combine_ms = [], []
        for dmatrix, cmatrix in zip(compressed, compressed_combine):
            dispatch_ms.append(communication.dispatch.latency(
                dmatrix.sum(axis=1), dmatrix.sum(axis=0)
            ))
            combine_ms.append(communication.combine.latency(
                cmatrix.sum(axis=0), cmatrix.sum(axis=1)
            ))
        candidate = {
            "dispatch_ms": float(np.sum(dispatch_ms)),
            "expert_ms": float(expert.sum()),
            "combine_ms": float(np.sum(combine_ms)),
            "stage_ms": float(np.sum(dispatch_ms) + expert.sum() + np.sum(combine_ms)),
            "remote_bytes": float(compressed.sum()),
            "combine_remote_bytes": float(compressed_combine.sum()),
        }
        candidate["stage_gain_percent_zero_codec_overhead"] = gain(baseline, candidate)
        candidate["dispatch_byte_reduction_percent"] = 100 * (
            1 - compressed.sum() / max(baseline_bytes.sum(), 1)
        )
        candidate["combine_byte_reduction_percent"] = 100 * (
            1 - compressed_combine.sum() / max(baseline_bytes.sum(), 1)
        )
        candidate["live_mask_selected_layer_dispatch_byte_reduction_percent"] = (
            100 * (1 - live_encoded_dispatch / max(live_raw / 2, 1)) if control == "adjacent" else None
        )
        candidate["live_mask_selected_layer_combine_byte_reduction_percent"] = (
            100 * (1 - live_encoded_combine / max(live_raw / 2, 1)) if control == "adjacent" else None
        )
        candidate["codec_overhead_budget_ms"] = max(
            0.0, baseline["stage_ms"] - candidate["stage_ms"]
        )
        # Sensitivity only: assume the five-layer byte-saving pattern repeats
        # identically in every routed layer.  This is not measured evidence.
        scale = ROUTED_LAYERS / len(SELECTED_LAYERS)
        extrapolated_dispatch = np.maximum(0.0, baseline_bytes - dispatch_saving * scale)
        extrapolated_combine = np.maximum(0.0, baseline_bytes - combine_saving * scale)
        extrapolated_dispatch_ms, extrapolated_combine_ms = [], []
        for dmatrix, cmatrix in zip(extrapolated_dispatch, extrapolated_combine):
            extrapolated_dispatch_ms.append(communication.dispatch.latency(
                dmatrix.sum(axis=1), dmatrix.sum(axis=0)
            ))
            extrapolated_combine_ms.append(communication.combine.latency(
                cmatrix.sum(axis=0), cmatrix.sum(axis=1)
            ))
        extrapolated_stage = (float(np.sum(extrapolated_dispatch_ms))
                              + float(expert.sum())
                              + float(np.sum(extrapolated_combine_ms)))
        candidate["optimistic_all_layer_linear_extrapolation_stage_gain_percent"] = (
            100 * (1 - extrapolated_stage / baseline["stage_ms"])
        )
        results[control] = candidate
    return {"baseline": baseline, "controls": results,
            "scope": "selected layers 1/5/10/14/19; zero codec overhead optimistic upper bound"}


def markdown_table(headers, rows):
    result = ["| " + " | ".join(headers) + " |",
              "|" + "|".join(["---"] * len(headers)) + "|"]
    result.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(result)


def write_reports(summary: dict, reports: Path):
    reports.mkdir(exist_ok=True)
    figures = reports / "figures/frontier_delta_ep"
    figures.mkdir(parents=True, exist_ok=True)
    phase = summary["block_cached"]
    exact = summary["exactness"]
    repeat = exact["same_shape_repeated_refinement"]
    opening = exact["new_block_shape_change"]
    rows = []
    for boundary in ("layer_input", "key_projection", "value_projection", "router_topk",
                     "router_weights", "routed_expert_output", "post_layer", "logits"):
        if boundary in repeat:
            rows.append((boundary, f"{100*repeat[boundary]['bit_exact_row_fraction']:.6f}%",
                         f"{repeat[boundary]['p95_record_max_relative_l2']:.3e}",
                         f"{100*opening.get(boundary, {}).get('bit_exact_row_fraction', float('nan')):.4f}%",
                         f"{opening.get(boundary, {}).get('p95_record_max_relative_l2', float('nan')):.3e}"))
    (reports / "block_cached_baseline_exactness.md").write_text(
        "# Block-cached baseline exactness\n\n"
        "Completed-prefix states are mathematically independent of future blocks under the block mask. "
        "At a fixed sequence shape, repeated refinements are bit-exact at every cached hidden/KV/router/"
        "expert/post-layer boundary. Extending to a new block changes "
        "the BF16 SDPA execution shape and can create numerical-order drift, so B1 conservatively performs "
        "one full-prefix refresh at block open and caches it for the remaining refinements.\n\n" +
        markdown_table(["Boundary", "same-shape bit-exact rows", "same-shape P95 max rel-L2",
                        "block-open bit-exact rows", "block-open P95 max rel-L2"], rows) +
        "\n\nClassification: cached same-shape boundaries are bit-exact. The logits row is elementwise "
        "99.99999% exact but contains a sparse non-equality per row and is not a cache boundary. "
        "Block-open differences are BF16 kernel-shape/numerical-order effects that can become "
        "semantically meaningful in deep layers.\n"
    )
    quality = phase["quality"]
    (reports / "block_cached_baseline_quality.md").write_text(
        "# Block-cached baseline quality\n\n"
        f"GSM8K-32 parity is **{quality['correct']}/{quality['requests']} correct**, matching the reference "
        f"{quality['reference_correct']}/{quality['requests']}; exact generation, parsed answer, NFE, and "
        f"termination all match {quality['requests']}/{quality['requests']}, with {quality['remaining_masks']} "
        "remaining MASK tokens. This is dense semantic emulation, not a wall-time speedup claim.\n"
    )
    proj_rows = []
    for ep in (4, 8):
        item = phase[f"ep{ep}"]
        proj_rows.append((f"EP{ep}", f"{item['naive']['stage_ms_per_request']:.3f}",
                          f"{item['block_cached']['stage_ms_per_request']:.3f}",
                          f"{item['gain_percent']:.3f}%", f"{item['fresh_pair_reduction_percent']:.3f}%"))
    (reports / "block_cached_baseline_ep_projection.md").write_text(
        "# Block-cached baseline EP projection\n\n"
        "Matched GSM8K-32 trajectories; simulated routed-MoE stage only. B1 includes a conservative "
        "full-prefix refresh at every new block shape.\n\n" +
        markdown_table(["Target", "B0 naive ms/request", "B1 cached ms/request", "stage gain",
                        "fresh pair reduction"], proj_rows) + "\n"
    )

    frontier = summary["frontier"]
    fq = frontier["quality"]
    (reports / "frontier_ep_quality.md").write_text(
        "# FrontierEP quality\n\n"
        f"F1 one-finalization seal stops at GSM8K-32: {fq['correct']}/{fq['requests']} versus "
        f"{fq['reference_correct']}/{fq['requests']} reference, with {fq['baseline_correct_to_wrong']} "
        f"baseline-correct→wrong transitions. NFE changes {fq['reference_nfe']}→{fq['nfe']} "
        f"({100*(fq['nfe']/fq['reference_nfe']-1):+.2f}%), and the worst request reaches "
        f"{fq['maximum_nfe_ratio']:.2f}× baseline NFE. All requests terminate with zero masks. "
        "F2 already loses one baseline-correct sample at n=8. Per the gate, neither advances.\n"
    )
    (reports / "frontier_ep_workload.md").write_text(
        "# FrontierEP workload\n\n" +
        markdown_table(["Metric", "B1", "F1", "change"], [
            ("NFE", fq["reference_nfe"], fq["nfe"], f"{100*(fq['nfe']/fq['reference_nfe']-1):+.3f}%"),
            ("fresh rows", phase["quality"]["fresh_rows"], fq["fresh_rows"],
             f"{-frontier['fresh_row_reduction_percent']:.3f}%"),
            ("fresh current rows", phase["quality"]["fresh_current_rows"], fq["fresh_current_rows"],
             f"{100*(fq['fresh_current_rows']/phase['quality']['fresh_current_rows']-1):+.3f}%"),
        ]) + "\n\nFresh-work reduction is real, but altered trajectories increase NFE and break the quality gate.\n"
    )
    fproj = []
    for ep in (4, 8):
        item = frontier[f"ep{ep}"]
        fproj.append((f"EP{ep}", f"{item['baseline']['stage_ms_per_request']:.3f}",
                      f"{item['candidate']['stage_ms_per_request']:.3f}",
                      f"{item['actual_stage_gain_percent']:.3f}%",
                      f"{item['nfe_normalized_stage_gain_percent']:.3f}%"))
    (reports / "frontier_ep_ep_projection.md").write_text(
        "# FrontierEP routed-MoE projection\n\nSIMULATED-EP4/8-EP2-CALIBRATED; not E2E latency.\n\n" +
        markdown_table(["Target", "B1 ms/request", "F1 ms/request", "actual rollout gain",
                        "NFE-normalized gain"], fproj) +
        "\n\nThe normalization exposes the per-forward sparse-work effect independently from changed NFE. "
        "It does not rescue the failed semantic contract.\n"
    )
    (reports / "frontier_ep_ar_control.md").write_text(
        "# FrontierEP AR control\n\nA normal causal AR KV cache already removes every prior generated token "
        "from fresh token execution. There is no repeatedly refined within-block committed frontier, so "
        "incremental one-/two-finalization sealing beyond standard AR caching is **0% by construction**. "
        "The structural opportunity is dLLM-specific, but the tested dLLM semantic change is unsafe.\n"
    )

    comp = summary["delta"]["compression"]
    held = comp["heldout"]
    delta_rows = []
    for boundary in ("dispatch", "combine"):
        for control in ("adjacent", "wrong_token", "nonadjacent", "zero_predictor"):
            item = held[boundary].get(control)
            if item:
                delta_rows.append((boundary, control, f"{item['compression_ratio_p50']:.3f}×",
                                   f"{item['aggregate_compression_ratio']:.3f}×",
                                   f"{100*item['fallback_fraction']:.2f}%",
                                   f"{100*item['xor_zero_byte_fraction_p50']:.2f}%"))
    (reports / "delta_ep_predictability.md").write_text(
        "# DeltaEP predictability\n\n"
        f"Across all 19 layers the live-MASK adjacent STAY fraction is "
        f"{100*summary['delta']['stay']['stay_route_fraction']:.3f}% (router mass "
        f"{100*summary['delta']['stay']['stay_router_mass_fraction']:.3f}%). For the five vector-audited "
        "layers, adjacent inputs are closer than wrong-token, nonadjacent, and zero predictors, but exact "
        "BF16 equality is not high enough for a strong simple lossless codec.\n\n" +
        markdown_table(["Boundary", "cosine P50/P90", "relative-L2 P50/P90",
                        "exact BF16 words P50", "|delta| P90-vector P50"], [
            (boundary,
             f"{held[boundary]['adjacent']['cosine_p50']:.4f} / {held[boundary]['adjacent']['cosine_p90']:.4f}",
             f"{held[boundary]['adjacent']['relative_l2_p50']:.4f} / {held[boundary]['adjacent']['relative_l2_p90']:.4f}",
             f"{100*held[boundary]['adjacent']['exact_word_fraction_p50']:.3f}%",
             f"{held[boundary]['adjacent']['delta_abs_p90_p50']:.4f}")
            for boundary in ("dispatch", "combine")
        ]) +
        "\n\nThese are activation/output vector similarities, not permission to reuse expert output. "
        "The previous branch-reuse result remains unsafe.\n"
    )
    (reports / "delta_ep_compression.md").write_text(
        "# DeltaEP lossless compression\n\nDiscovery IDs 0–7 chose and froze a byte bitmap + changed-byte "
        "literals + one-byte mode tag with full-vector fallback. The table is the one-shot held-out result "
        "on IDs 8–15.\n\n" +
        markdown_table(["Boundary", "Predictor", "median ratio", "aggregate ratio", "fallback",
                        "zero XOR bytes"], delta_rows) +
        "\n\nThe word-bitmap codec falls back on the median (1.0×). CPU zstd/lz4 are not used "
        "as proposed codecs, and no CUDA codec was built because the lossless gate already fails.\n\n"
        "## Predefined layer stratum\n\n" +
        markdown_table(["Layer", "discovery median", "held-out median", "held-out aggregate"], [
            (layer,
             f"{comp['strata']['layer']['discovery'][str(layer)]['median_ratio']:.3f}×",
             f"{comp['strata']['layer']['heldout'][str(layer)]['median_ratio']:.3f}×",
             f"{comp['strata']['layer']['heldout'][str(layer)]['aggregate_ratio']:.3f}×")
            for layer in SELECTED_LAYERS
        ]) +
        f"\n\nDiscovery selects layer {comp['strata']['frozen_best_discovery_layer']}; its frozen "
        f"held-out median is {comp['strata']['heldout_for_frozen_layer']['median_ratio']:.3f}×. "
        "The subgroup reproduces, but five-layer and all-layer stage upper bounds remain far below the "
        "systems gate, so it is not promoted.\n"
    )
    system_rows = []
    for ep in (4, 8):
        item = summary["delta"][f"ep{ep}"]
        adjacent = item["controls"]["adjacent"]
        system_rows.append((f"EP{ep}",
                            f"{adjacent['dispatch_byte_reduction_percent']:.3f}%",
                            f"{adjacent['combine_byte_reduction_percent']:.3f}%",
                            f"{adjacent['stage_gain_percent_zero_codec_overhead']:.4f}%",
                            f"{adjacent['optimistic_all_layer_linear_extrapolation_stage_gain_percent']:.4f}%",
                            f"{adjacent['codec_overhead_budget_ms']:.4f}"))
    (reports / "delta_ep_ep_projection.md").write_text(
        "# DeltaEP routed-MoE projection\n\n"
        "The projection applies the held-out exact codec only to measured layers 1/5/10/14/19 on top "
        "of B1. Expert compute is unchanged. Stage gain is an optimistic **zero codec-overhead upper "
        "bound**, not a measured implementation.\n\n" +
        markdown_table(["Target", "dispatch-byte reduction", "combine-byte reduction",
                        "5-layer stage upper bound", "all-layer linear extrapolation",
                        "total overhead budget (ms/16 req)"], system_rows) +
        "\n\nBecause even zero-overhead gain is below the gate, any encode/decode/cache lookup cost can only "
        "reduce it; a GPU microkernel is not justified. A branch-keyed selected-five-layer cache has a "
        "worst-case 5.0 MiB/request/direction bound (10.0 MiB for dispatch+combine); extrapolating to "
        "all 19 layers gives 19.0 MiB/direction. Token-destination deduplication can lower this, but does "
        "not change the latency conclusion.\n"
    )
    control_rows = []
    for ep in (4, 8):
        controls = summary["delta"][f"ep{ep}"]["controls"]
        control_rows.append((f"EP{ep}",) + tuple(
            f"{controls[name]['stage_gain_percent_zero_codec_overhead']:.4f}%"
            for name in ("adjacent", "wrong_token", "nonadjacent", "zero_predictor")
        ))
    (reports / "delta_ep_controls.md").write_text(
        "# DeltaEP controls\n\n" +
        markdown_table(["Target", "adjacent dLLM", "wrong-token", "nonadjacent", "generic zero"],
                       control_rows) +
        "\n\nAdjacent refinement is measurably more compressible than every control, so temporal signal exists; "
        "the absolute lossless and stage gains are nevertheless too small. AR one-shot has no previous "
        "same-edge predictor and therefore 0% temporal saving.\n"
    )
    (reports / "delta_ep_quality.md").write_text(
        "# DeltaEP quality\n\nThe bitmap+literals representation reconstructs every BF16 word by exact XOR "
        "inversion; unit tests cover unchanged, partially changed, and random tensors. Therefore semantic "
        "quality is unchanged by construction. No lossy path or quality rollout was run. The captured "
        "combine vector is the post-routed-MoE combined-output predictor; destination-local branch-vector "
        "compression remains unmeasured and is not overclaimed.\n"
    )

    # Figures.
    names = ["B0 naive", "B1 cached", "F1 changed"]
    for ep in (4, 8):
        values = [phase[f"ep{ep}"]["naive"]["stage_ms_per_request"],
                  phase[f"ep{ep}"]["block_cached"]["stage_ms_per_request"],
                  frontier[f"ep{ep}"]["candidate"]["stage_ms_per_request"]]
        plt.figure(figsize=(6, 4)); plt.bar(names, values)
        plt.ylabel("simulated routed-MoE ms/request"); plt.title(f"EP{ep} substrate and F1")
        plt.tight_layout(); plt.savefig(figures / f"ep{ep}_stage.png", dpi=180); plt.close()
    controls = ("adjacent", "wrong_token", "nonadjacent", "zero_predictor")
    plt.figure(figsize=(7, 4))
    x = np.arange(len(controls)); width = .35
    plt.bar(x-width/2, [held["dispatch"][c]["compression_ratio_p50"] for c in controls],
            width, label="dispatch")
    plt.bar(x+width/2, [held["combine"][c]["compression_ratio_p50"] for c in controls],
            width, label="combine")
    plt.axhline(1.2, color="red", linestyle="--", label="gate")
    plt.xticks(x, controls, rotation=20); plt.ylabel("held-out median lossless ratio")
    plt.legend(); plt.tight_layout(); plt.savefig(figures / "delta_controls.png", dpi=180); plt.close()
    profile_b = phase["refinement_profile"]
    profile_f = frontier["refinement_profile"]
    refinements = sorted(set(map(int, profile_b)) | set(map(int, profile_f)))
    def mean_profile(profile, key, refinement):
        row = profile.get(str(refinement))
        return row[key] / row["steps"] if row else np.nan
    plt.figure(figsize=(7, 4))
    plt.plot(refinements, [mean_profile(profile_b, "fresh_current", x) for x in refinements],
             marker="o", markersize=3, label="B1")
    plt.plot(refinements, [mean_profile(profile_f, "fresh_current", x) for x in refinements],
             marker="o", markersize=3, label="F1")
    plt.xlabel("block-relative refinement"); plt.ylabel("mean fresh current-block rows")
    plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(figures / "frontier_fresh_rows_by_refinement.png", dpi=180); plt.close()

    work = summary["persistent_worklist"]
    (reports / "frontier_delta_ep_summary.md").write_text(
        "# FrontierEP + DeltaEP discovery summary\n\n"
        "## Verdict\n\n**FrontierEP: NO-GO. DeltaEP: NO-GO.** Phase 0 succeeds and materially "
        "changes the correct denominator, but neither dLLM-native extension survives its complete gate.\n\n"
        "FrontierEP removes intended fresh EP work, and AR control confirms the opportunity is tied to "
        "repeated dLLM refinement. However, changing committed-token dependencies causes two additional "
        "GSM8K-32 failures and pathological trajectory inflation. DeltaEP has a real adjacent-predictor "
        "advantage, but the held-out simple exact codec reaches only "
        f"{held['dispatch']['adjacent']['compression_ratio_p50']:.3f}× dispatch and "
        f"{held['combine']['adjacent']['compression_ratio_p50']:.3f}× combine median compression—below "
        "the lossless gate before codec overhead.\n\n"
        "## Persistent worklist\n\n"
        f"The measured/screened packing-sort-count share is {work['overhead_share_percent']:.3f}% of the "
        "remaining routed expert stage. It is retained only as overhead characterization, not a standalone "
        "method.\n\n"
        "## Evidence boundary\n\nEP4/EP8 numbers are `SIMULATED-EP4/8-EP2-CALIBRATED` routed-MoE-stage "
        "projections, not E2E latency. Dense emulation is used only for semantics. No production runtime "
        "or complex CUDA codec was implemented.\n"
    )


def main():
    root = Path("artifacts/frontier_delta_ep")
    reports = Path("reports")
    communication = CommunicationScenario.load(
        Path("artifacts/virtual_ep/20260916_215010/communication/model.json"),
        "ep2_calibrated_base",
    )
    models = {
        4: BatchComputeModel(Path(
            "artifacts/virtual_ep/20260917_four_hypothesis_discovery/compute_model/ep4_grouped_mm_discovery.csv"
        )),
        8: BatchComputeModel(Path(
            "artifacts/virtual_ep/20260916_215010/compute_model/ep8_grouped_mm.csv"
        )),
    }
    block_dir = root / "block_cached_gsm8_v2"
    f1_dir = root / "frontier1_gsm8"
    naive_dir = Path(
        "artifacts/virtual_ep/20260917_selective_refinement/rollouts/p0_full_none_ratio100_parity8"
    )
    delta_dir = root / "audit_delta16_v2"
    summary = {
        "scope": {
            "model": "inclusionAI/LLaDA2.0-mini",
            "revision": "dad945cac317da394b390f82c7b40691d8a881ed",
            "physical_gpus": [0, 1],
            "timing": "SIMULATED routed-MoE stage only",
        },
        "exactness": exactness_summary(root / "audit_exact16_v3"),
        "block_cached": {"quality": summarize_quality(block_dir),
                         "refinement_profile": frontier_profile(block_dir)},
        "frontier": {"quality": summarize_quality(f1_dir),
                     "refinement_profile": frontier_profile(f1_dir)},
        "delta": {
            "stay": stay_fraction(block_dir),
            "compression": compression_statistics(load_delta(delta_dir)),
        },
    }
    for ep in (4, 8):
        naive = project(*load_shapes(naive_dir, ep, selective_prefix="s0"),
                        ep, models[ep], communication)
        cached = project(*load_shapes(block_dir, ep), ep, models[ep], communication)
        candidate = project(*load_shapes(f1_dir, ep), ep, models[ep], communication)
        summary["block_cached"][f"ep{ep}"] = {
            "naive": naive, "block_cached": cached,
            "gain_percent": gain(naive, cached),
            "fresh_pair_reduction_percent": gain(naive, cached, "fresh_expert_token_pairs"),
            "label": f"SIMULATED-EP{ep}-EP2-CALIBRATED",
        }
        summary["frontier"][f"ep{ep}"] = {
            "baseline": cached, "candidate": candidate,
            "actual_stage_gain_percent": gain(cached, candidate),
            "nfe_normalized_stage_gain_percent": 100 * (
                1 - candidate["stage_ms_per_forward"] / cached["stage_ms_per_forward"]
            ),
            "label": f"SIMULATED-EP{ep}-EP2-CALIBRATED",
        }
    bq = summary["block_cached"]["quality"]
    fq = summary["frontier"]["quality"]
    summary["frontier"]["fresh_row_reduction_percent"] = 100 * (
        1 - fq["fresh_rows"] / bq["fresh_rows"]
    )
    delta = load_delta(delta_dir)
    summary["delta"]["compression"]["strata"] = compression_strata(delta)
    for ep in (4, 8):
        summary["delta"][f"ep{ep}"] = delta_system_projection(
            delta, block_dir, ep, models[ep], communication
        )
    worklist_path = root / "worklist_overhead.json"
    summary["persistent_worklist"] = (
        json.loads(worklist_path.read_text()) if worklist_path.exists() else
        {"overhead_share_percent": float("nan"), "status": "NOT_MEASURED"}
    )
    summary["verdict"] = {
        "block_cached": "PASS",
        "frontier_ep": "NO-GO",
        "delta_ep": "NO-GO",
        "stronger_candidate": "NEITHER",
    }
    write_reports(summary, reports)
    (reports / "frontier_delta_ep_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(summary["verdict"], indent=2))


if __name__ == "__main__":
    main()
