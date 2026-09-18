#!/usr/bin/env python3
"""Summarize actual route-pruning rollouts and calibrated EP projections."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.analyze_h1_straggler_deepdive import BatchComputeModel, load_discovery_trace
from scripts.analyze_low_utility_straggler import communication_latency
from virtual_ep.comm_model import CommunicationScenario


NUM_EXPERTS = 256


def load_jsonl(paths):
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["sample_id"])] = row
    return rows


def exact_paired_pvalue(baseline_only: int, method_only: int) -> float:
    total = baseline_only + method_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(baseline_only, method_only) + 1)) / 2**total
    return min(1.0, 2 * tail)


def quantile_from_hist(histogram: np.ndarray, q: float) -> int:
    target = q * histogram.sum()
    return int(np.searchsorted(np.cumsum(histogram), target, side="left"))


def quality_summary(
    method: dict,
    baseline: dict,
    sample_ids: list[int],
    baseline_nfe: dict[int, int],
) -> dict:
    rows = [method[index] for index in sample_ids]
    base = [baseline[index] for index in sample_ids]
    correct = np.asarray([bool(row.get("correct")) for row in rows])
    base_correct = np.asarray([bool(row.get("correct")) for row in base])
    baseline_only = int(np.count_nonzero(base_correct & ~correct))
    method_only = int(np.count_nonzero(~base_correct & correct))
    physical = np.asarray([row["original_routes"] / 8 for row in rows], dtype=np.float64)
    removed_mass = np.asarray([row["actual_removed_mass_fraction"] for row in rows])
    fresh = np.asarray([row["fresh_routes"] for row in rows], dtype=np.int64)
    k_hist = np.sum([np.asarray(row["k_hist"], dtype=np.int64) for row in rows], axis=0)
    method_nfe = np.asarray([row["nfe"] for row in rows], dtype=np.float64)
    reference_nfe = np.asarray([baseline_nfe[index] for index in sample_ids], dtype=np.float64)
    return {
        "n": len(rows),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "baseline_accuracy": float(base_correct.mean()),
        "baseline_correct_to_wrong": baseline_only,
        "baseline_wrong_to_correct": method_only,
        "paired_exact_pvalue": exact_paired_pvalue(baseline_only, method_only),
        "parsed_answer_identity": float(np.mean([
            row.get("parsed_answer") == reference.get("parsed_answer")
            for row, reference in zip(rows, base)
        ])),
        "exact_generation_identity": float(np.mean([
            row.get("output_ids") == reference.get("output_ids")
            for row, reference in zip(rows, base)
        ])),
        "mean_nfe": float(np.mean([row["nfe"] for row in rows])),
        "median_nfe": float(np.median([row["nfe"] for row in rows])),
        "baseline_mean_nfe": float(reference_nfe.mean()),
        "baseline_median_nfe": float(np.median(reference_nfe)),
        "mean_nfe_change_percent": float(100 * (method_nfe.mean() / reference_nfe.mean() - 1)),
        "max_per_request_nfe_ratio": float(np.max(method_nfe / reference_nfe)),
        "mean_blocks": float(np.mean([row.get("blocks", 0) for row in rows])),
        "termination_counts": {
            name: sum(row.get("termination", row.get("termination_reason")) == name for row in rows)
            for name in ("EOS", "GEN_LENGTH_CAP", "REMAINING_MASK")
        },
        "remaining_masks": int(sum(row.get("remaining_masks", row.get("remaining_mask_tokens", 0)) for row in rows)),
        "nominal_removed_mass_fraction": float(rows[0].get("mass_budget", 0)),
        "actual_removed_mass_fraction": float(np.sum(removed_mass * physical) / physical.sum()),
        "fresh_route_count": int(fresh.sum()),
        "route_reduction_fraction": float(1 - fresh.sum() / np.sum(physical * 8)),
        "avg_k": float(fresh.sum() / physical.sum()),
        "k_distribution": {
            "min": int(np.flatnonzero(k_hist)[0]),
            "p10": quantile_from_hist(k_hist, .10),
            "median": quantile_from_hist(k_hist, .50),
            "p90": quantile_from_hist(k_hist, .90),
            "histogram": k_hist.tolist(),
        },
    }


def baseline_arrays(aggregate: Path, sample_ids: list[int]) -> dict:
    arrays = load_discovery_trace(aggregate).arrays
    selected = np.isin(arrays["request_id"], sample_ids)
    return {
        "request_id": arrays["request_id"][selected],
        "physical_rows": arrays["physical_rows"][selected],
        "histogram": arrays["expert_counts_by_class"][selected].sum(axis=1),
        "unique_ep4": arrays["unique_matrix_ep4"][selected],
        "unique_ep8": arrays["unique_matrix_ep8"][selected],
        "fanout_sum_ep4": arrays["mean_fanout_ep4"][selected] * arrays["physical_rows"][selected],
        "fanout_sum_ep8": arrays["mean_fanout_ep8"][selected] * arrays["physical_rows"][selected],
        "unique_experts": np.count_nonzero(arrays["expert_counts_by_class"][selected].sum(axis=1), axis=1),
    }


def baseline_nfe_by_request(aggregate: Path, sample_ids: list[int]) -> dict[int, int]:
    arrays = load_discovery_trace(aggregate).arrays
    output = {}
    for request_id in sample_ids:
        selected = arrays["request_id"] == request_id
        if not np.any(selected):
            raise ValueError(f"aggregate trace missing request {request_id}")
        output[request_id] = int(arrays["nfe"][selected].max()) + 1
    return output


def method_arrays(directory: Path, sample_ids: list[int]) -> dict:
    result: dict[str, list[np.ndarray]] = {}
    for request_id in sample_ids:
        with np.load(directory / "traces" / f"request_{request_id:04d}.npz", allow_pickle=False) as trace:
            for name in (
                "request_id", "physical_rows", "histogram", "unique_ep4", "unique_ep8",
                "fanout_sum_ep4", "fanout_sum_ep8", "unique_experts", "pressure_hist",
            ):
                result.setdefault(name, []).append(trace[name])
    return {name: np.concatenate(parts, axis=0) for name, parts in result.items()}


def project(arrays: dict, ep: int, compute: BatchComputeModel, communication: CommunicationScenario) -> dict:
    histogram = arrays["histogram"].astype(np.float64)
    per_rank = NUM_EXPERTS // ep
    local_histograms = np.concatenate([
        histogram[:, rank * per_rank:(rank + 1) * per_rank] for rank in range(ep)
    ], axis=0)
    local_features = compute.features_for(local_histograms)
    distances, _indices = compute.tree.query(local_features / compute.scale)
    assignment_min = float(compute.features[:, 1].min())
    assignment_max = float(compute.features[:, 1].max())
    times = np.column_stack([
        compute.predict_batch(histogram[:, rank * per_rank:(rank + 1) * per_rank])
        for rank in range(ep)
    ])
    unique = arrays[f"unique_ep{ep}"].astype(np.int32)
    dispatch, outgoing, _incoming = communication_latency(communication.dispatch, unique)
    combine, _reverse_out, _reverse_in = communication_latency(
        communication.combine, np.transpose(unique, (0, 2, 1))
    )
    maximum = times.max(axis=1)
    ordered = np.sort(times, axis=1)
    mean = times.mean(axis=1)
    wait = np.divide(
        ep * maximum - times.sum(axis=1), ep * maximum,
        out=np.zeros_like(maximum), where=maximum > 0,
    )
    request_count = len(np.unique(arrays["request_id"]))
    stage = dispatch + maximum + combine
    return {
        "label": f"SIMULATED-EP{ep}-EP2-CALIBRATED",
        "requests": request_count,
        "invocations": len(histogram),
        "dispatch_ms_per_request": float(dispatch.sum() / request_count),
        "expert_ms_per_request": float(maximum.sum() / request_count),
        "combine_ms_per_request": float(combine.sum() / request_count),
        "stage_ms_per_request": float(stage.sum() / request_count),
        "max_mean": float(np.mean(maximum / np.maximum(mean, 1e-12))),
        "max_second": float(np.mean(maximum / np.maximum(ordered[:, -2], 1e-12))),
        "cv": float(np.mean(times.std(axis=1) / np.maximum(mean, 1e-12))),
        "wait_fraction": float(np.mean(wait)),
        "remote_logical_bytes_per_request": float(outgoing.sum() / request_count),
        "fanout": float(np.sum(arrays[f"fanout_sum_ep{ep}"]) / np.sum(arrays["physical_rows"])),
        "unique_experts_per_invocation": float(np.mean(arrays["unique_experts"])),
        "compute_calibration": {
            "measured_assignment_range": [assignment_min, assignment_max],
            "observed_assignment_range": [
                float(local_features[:, 1].min()), float(local_features[:, 1].max())
            ],
            "out_of_assignment_range_fraction": float(np.mean(
                (local_features[:, 1] < assignment_min) | (local_features[:, 1] > assignment_max)
            )),
            "normalized_nearest_distance_p50": float(np.median(distances)),
            "normalized_nearest_distance_p95": float(np.percentile(distances, 95)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--baseline-generations", type=Path, nargs="+", required=True)
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "DIRECTORY"), required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", required=True)
    parser.add_argument("--ep4-compute", type=Path, required=True)
    parser.add_argument("--ep8-compute", type=Path, required=True)
    parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ids = sorted(set(args.sample_ids))
    baseline_generations = load_jsonl(args.baseline_generations)
    missing = set(ids) - set(baseline_generations)
    if missing:
        raise ValueError(f"baseline generations missing IDs: {sorted(missing)}")
    base_arrays = baseline_arrays(args.aggregate, ids)
    base_nfe = baseline_nfe_by_request(args.aggregate, ids)
    communication = CommunicationScenario.load(args.communication, "ep2_calibrated_base")
    compute = {4: BatchComputeModel(args.ep4_compute), 8: BatchComputeModel(args.ep8_compute)}
    projections = {f"ep{ep}": project(base_arrays, ep, compute[ep], communication) for ep in (4, 8)}
    document = {
        "sample_ids": ids,
        "baseline": {
            "projection": projections,
            "mean_nfe": float(np.mean(list(base_nfe.values()))),
            "median_nfe": float(np.median(list(base_nfe.values()))),
            "nfe_by_request": {str(key): value for key, value in base_nfe.items()},
        },
        "methods": {},
        "quality_is_actual_rollout": True,
        "ep_timing_is_simulated": True,
    }
    for name, raw_directory in args.method:
        directory = Path(raw_directory)
        generations = load_jsonl([directory / "generations.jsonl"])
        if set(ids) - set(generations):
            raise ValueError(f"{name} missing generation IDs")
        arrays = method_arrays(directory, ids)
        quality = quality_summary(generations, baseline_generations, ids, base_nfe)
        method_projection = {}
        for ep in (4, 8):
            value = project(arrays, ep, compute[ep], communication)
            value["stage_gain_percent"] = 100 * (
                1 - value["stage_ms_per_request"] / projections[f"ep{ep}"]["stage_ms_per_request"]
            )
            value["nfe_reduction_percent"] = -quality["mean_nfe_change_percent"]
            value["nfe_normalized_stage_gain_percent"] = 100 * (
                1
                - (value["stage_ms_per_request"] / quality["mean_nfe"])
                / (projections[f"ep{ep}"]["stage_ms_per_request"] / document["baseline"]["mean_nfe"])
            )
            method_projection[f"ep{ep}"] = value
        document["methods"][name] = {
            "directory": str(directory), "quality": quality,
            "projection": method_projection,
            "removed_pressure_hist": arrays.get("pressure_hist", np.zeros((1, 6))).sum(axis=0).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: {
            "accuracy": item["quality"]["accuracy"],
            "additional_wrong": item["quality"]["baseline_correct_to_wrong"],
            "actual_mass": item["quality"]["actual_removed_mass_fraction"],
            "ep8_gain": item["projection"]["ep8"]["stage_gain_percent"],
        } for name, item in document["methods"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
