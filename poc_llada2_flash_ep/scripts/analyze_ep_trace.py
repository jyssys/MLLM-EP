#!/usr/bin/env python3
"""Summarize observer-heavy EP traces without double-counting rank rows.

The dInfer bridge writes one row per physical rank and sparse layer. This
script reconstructs a logical layer invocation using the critical-rank
duration for each sequential stage and concatenates source-token routes from
all ranks. Rank rows are never summed as request latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


STAGES = ("router", "dispatch", "expert", "combine", "shared", "gather")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_of(iteration: int, count: int) -> str:
    ratio = iteration / max(count - 1, 1)
    return "early" if ratio < 1 / 3 else "middle" if ratio < 2 / 3 else "late"


def phase_from_remaining(remaining: list[int], rows: int) -> str:
    if not remaining or rows <= 0:
        return "unknown"
    ratio = sum(remaining) / rows
    return "early" if ratio > 2 / 3 else "middle" if ratio > 1 / 3 else "late"


def optional_float(value: object) -> float:
    return float(value) if value is not None else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    log_text = args.log.read_text(errors="replace")
    nfe_by_request = {
        f"measured_{int(offset)}": int(nfe)
        for offset, nfe in re.findall(r"\[iter\s+(\d+)\]nfe=\s*(\d+)", log_text)
    }
    if not nfe_by_request:
        raise RuntimeError("no measured request/wave NFE records found")

    denoise_by_request: dict[str, list[dict]] = defaultdict(list)
    denoise_prefix = "[LLADA_DENOISE]"
    for line in log_text.splitlines():
        position = line.find(denoise_prefix)
        if position < 0:
            continue
        record = json.loads(line[position + len(denoise_prefix) :])
        if record.get("request_id") in nfe_by_request:
            denoise_by_request[record["request_id"]].append(record)
    if set(denoise_by_request) != set(nfe_by_request):
        raise RuntimeError(
            "logical denoising trace is incomplete; use a dynamic-batching trace "
            f"(nfe requests={sorted(nfe_by_request)}, denoise requests={sorted(denoise_by_request)})"
        )

    ep_rows: list[dict] = []
    for rank in range(4):
        path = args.trace / f"ep_rank{rank}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("request_id") in nfe_by_request:
                    ep_rows.append(row)

    # Keep the final denoising calls per request/layer/rank. NFE also includes
    # prompt/KV prefill. Those rows can retain the previous logical identity,
    # so using NFE would incorrectly align a prefill with a refinement step.
    selected: dict[tuple[int, int, str, int], dict] = {}
    grouped: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in ep_rows:
        grouped[(row["rank"], row["layer"], row["request_id"])].append(row)
    for (rank, layer, request_id), rows in grouped.items():
        rows.sort(key=lambda item: item["invocation"])
        count = len(denoise_by_request[request_id])
        if len(rows) < count:
            raise RuntimeError(
                f"incomplete trace rank={rank} layer={layer} request={request_id}: "
                f"rows={len(rows)} nfe={count}"
            )
        for iteration, row in enumerate(rows[-count:]):
            row = dict(row)
            row["iteration"] = iteration
            row["phase"] = phase_of(iteration, count)
            selected[(rank, layer, request_id, iteration)] = row

    logical_rows: list[dict] = []
    logical_routes: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
    token_observations: dict[tuple[int, str, int, int], list[dict]] = defaultdict(list)
    for layer in range(1, 32):
        for request_id, denoise_records in denoise_by_request.items():
            count = len(denoise_records)
            for iteration in range(count):
                ranks = [selected[(rank, layer, request_id, iteration)] for rank in range(4)]
                denoise = denoise_records[iteration]
                route_ids = np.concatenate(
                    [np.asarray(row["topk_ids"], dtype=np.int64) for row in ranks], axis=0
                )
                route_weights = np.concatenate(
                    [np.asarray(row["topk_weights"], dtype=np.float32) for row in ranks], axis=0
                )
                logical_routes[(layer, request_id, iteration)] = (route_ids, route_weights)

                expert_load = np.bincount(route_ids.ravel(), minlength=256)
                rank_load = np.asarray(
                    [expert_load[start : start + 64].sum() for start in range(0, 256, 64)]
                )
                source_ranks = np.concatenate(
                    [np.full(len(ranks[source]["topk_ids"]), source) for source in range(4)]
                )
                destination_ranks = route_ids // 64
                remote = destination_ranks != source_ranks[:, None]
                fanout = np.asarray([len(np.unique(row)) for row in destination_ranks])
                positive_expert = expert_load[expert_load > 0]
                sequence_ids = [int(value) for value in denoise["sequence_ids"]]
                block_starts = [int(value) for value in denoise["block_starts"]]
                remaining_before = [int(value) for value in denoise["remaining_before"]]
                token_keys = [
                    (sequence_id, block_start + offset)
                    for sequence_id, block_start in zip(sequence_ids, block_starts)
                    for offset in range(32)
                ]
                if token_keys and len(token_keys) != route_ids.shape[0]:
                    raise RuntimeError(
                        f"logical identity mismatch: keys={len(token_keys)}, routes={route_ids.shape[0]}"
                    )
                for row in ranks:
                    if (
                        row.get("sequence_ids") != sequence_ids
                        or row.get("block_starts") != block_starts
                        or int(row["physical_m_global"]) != route_ids.shape[0]
                    ):
                        raise RuntimeError(
                            "rank trace and denoising-log identity disagree: "
                            f"request={request_id}, layer={layer}, iteration={iteration}"
                        )
                phase = phase_from_remaining(remaining_before, route_ids.shape[0])
                logical = {
                    "request_id": request_id,
                    "layer": layer,
                    "iteration": iteration,
                    "phase": phase,
                    "sequence_ids": json.dumps(sequence_ids, separators=(",", ":")),
                    "block_starts": json.dumps(block_starts, separators=(",", ":")),
                    "remaining_masked_positions": int(sum(remaining_before)),
                    "fresh_position_ratio": float(sum(remaining_before) / route_ids.shape[0]) if remaining_before else float("nan"),
                    "physical_m": int(route_ids.shape[0]),
                    "assignments": int(route_ids.size),
                    "remote_assignments": int(remote.sum()),
                    "remote_fraction": float(remote.mean()),
                    "dispatch_payload_bytes_bf16": int(remote.sum() * 4096 * 2),
                    "combine_payload_bytes_bf16": int(remote.sum() * 4096 * 2),
                    "active_experts": int((expert_load > 0).sum()),
                    "tiny_expert_fraction_le4": float(np.mean(positive_expert <= 4)),
                    "expert_group_p10": float(np.percentile(positive_expert, 10)),
                    "expert_group_p50": float(np.percentile(positive_expert, 50)),
                    "expert_group_p90": float(np.percentile(positive_expert, 90)),
                    "rank_load_max": int(rank_load.max()),
                    "rank_load_mean": float(rank_load.mean()),
                    "rank_load_cv": float(rank_load.std() / rank_load.mean()),
                    "rank_fanout_mean": float(fanout.mean()),
                    "rank_fanout_p95": float(np.percentile(fanout, 95)),
                    "input_rel_l2_lag1_median": float(
                        np.nanmedian([optional_float(row["input_rel_l2_lag1"]) for row in ranks])
                    ),
                    "input_cosine_lag1_median": float(
                        np.nanmedian([optional_float(row["input_cosine_lag1"]) for row in ranks])
                    ),
                    "moe_output_rel_l2_lag1_median": float(
                        np.nanmedian([optional_float(row["moe_output_rel_l2_lag1"]) for row in ranks])
                    ),
                    "moe_output_cosine_lag1_median": float(
                        np.nanmedian([optional_float(row["moe_output_cosine_lag1"]) for row in ranks])
                    ),
                }
                for rank_index, load in enumerate(rank_load):
                    logical[f"rank_load_{rank_index}"] = int(load)
                for stage in STAGES:
                    values = [float(row[f"{stage}_ms"]) for row in ranks]
                    logical[f"{stage}_critical_ms"] = max(values)
                    logical[f"{stage}_median_rank_ms"] = float(np.median(values))
                logical["moe_critical_sum_ms"] = sum(
                    logical[f"{stage}_critical_ms"] for stage in STAGES
                )
                logical_rows.append(logical)
                for row_index, (sequence_id, position) in enumerate(token_keys):
                    token_observations[(layer, request_id, sequence_id, position)].append(
                        {
                            "iteration": iteration,
                            "phase": phase,
                            "ids": route_ids[row_index],
                            "weights": route_weights[row_index],
                        }
                    )

    stage_summary: list[dict] = []
    for phase in ("all", "early", "middle", "late"):
        subset = logical_rows if phase == "all" else [r for r in logical_rows if r["phase"] == phase]
        if not subset:
            continue
        for stage in STAGES:
            values = np.asarray([r[f"{stage}_critical_ms"] for r in subset])
            stage_summary.append(
                {
                    "phase": phase,
                    "stage": stage,
                    "logical_invocations": len(values),
                    "median_critical_ms": float(np.median(values)),
                    "p95_critical_ms": float(np.percentile(values, 95)),
                    "mean_critical_ms": float(np.mean(values)),
                    "sum_critical_ms": float(np.sum(values)),
                }
            )

    phase_summary: list[dict] = []
    for phase in ("all", "early", "middle", "late"):
        subset = logical_rows if phase == "all" else [r for r in logical_rows if r["phase"] == phase]
        if not subset:
            continue
        summary = {"phase": phase, "logical_invocations": len(subset)}
        for key in (
            "physical_m", "remote_fraction", "active_experts",
            "tiny_expert_fraction_le4", "rank_load_cv", "rank_fanout_mean",
            "input_rel_l2_lag1_median", "input_cosine_lag1_median",
            "moe_output_rel_l2_lag1_median", "moe_output_cosine_lag1_median",
            "moe_critical_sum_ms",
        ):
            values = np.asarray([r[key] for r in subset], dtype=float)
            summary[f"{key}_median"] = float(np.nanmedian(values))
            summary[f"{key}_mean"] = float(np.nanmean(values))
        phase_summary.append(summary)

    token_items_by_layer: dict[int, list[tuple[tuple, list[dict]]]] = defaultdict(list)
    for key, observations in token_observations.items():
        observations.sort(key=lambda item: item["iteration"])
        token_items_by_layer[key[0]].append((key, observations))

    temporal: list[dict] = []
    for lag in (1, 2, 4, 8):
        for phase in ("all", "early", "middle", "late"):
            for layer_group, layer_ids in (
                ("all", range(1, 32)),
                ("early_layers", range(1, 11)),
                ("middle_layers", range(11, 22)),
                ("late_layers", range(22, 32)),
            ):
                exact, sets, overlaps, destinations, destination_overlaps, weights = [], [], [], [], [], []
                for layer in layer_ids:
                    for key, observations in token_items_by_layer[layer]:
                        for offset in range(len(observations) - lag):
                            first, second = observations[offset], observations[offset + lag]
                            if phase != "all" and first["phase"] != phase:
                                continue
                            ids_a, weights_a = first["ids"], first["weights"]
                            ids_b, weights_b = second["ids"], second["weights"]
                            exact.append(float(np.mean(ids_a == ids_b)))
                            sets.append(float(set(ids_a) == set(ids_b)))
                            overlaps.append(len(set(ids_a) & set(ids_b)) / len(ids_a))
                            dst_a, dst_b = ids_a // 64, ids_b // 64
                            destinations.append(float(set(dst_a) == set(dst_b)))
                            destination_overlaps.append(
                                len(set(dst_a) & set(dst_b)) / max(len(set(dst_a)), 1)
                            )
                            weights.append(cosine(weights_a, weights_b))

                # Rank-load persistence is a wave property, not a token
                # property. Compare only within identical selected-sequence /
                # diffusion-block contexts so dynamic batching cannot create
                # a false temporal mismatch.
                loads, critical = [], []
                wave_groups: dict[tuple, list[dict]] = defaultdict(list)
                for row in logical_rows:
                    if row["layer"] not in layer_ids:
                        continue
                    context = (row["layer"], row["request_id"], row["sequence_ids"], row["block_starts"])
                    wave_groups[context].append(row)
                for rows in wave_groups.values():
                    rows.sort(key=lambda item: item["iteration"])
                    for offset in range(len(rows) - lag):
                        first, second = rows[offset], rows[offset + lag]
                        if phase != "all" and first["phase"] != phase:
                            continue
                        load_a = np.asarray([first[f"rank_load_{rank}"] for rank in range(4)])
                        load_b = np.asarray([second[f"rank_load_{rank}"] for rank in range(4)])
                        loads.append(cosine(load_a, load_b))
                        critical.append(float(load_a.argmax() == load_b.argmax()))
                temporal.append(
                    {
                        "lag": lag, "phase": phase, "layer_group": layer_group,
                        "token_pairs": len(exact),
                        "wave_pairs": len(loads),
                        "ordered_expert_agreement": float(np.mean(exact)),
                        "topk_set_agreement": float(np.mean(sets)),
                        "topk_expert_overlap_fraction": float(np.mean(overlaps)),
                        "destination_set_agreement": float(np.mean(destinations)),
                        "destination_overlap_fraction": float(np.mean(destination_overlaps)),
                        "rank_load_cosine": float(np.mean(loads)),
                        "critical_rank_persistence": float(np.mean(critical)),
                        "topk_weight_cosine": float(np.mean(weights)),
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "logical_invocations.csv", logical_rows)
    write_csv(args.output / "stage_summary.csv", stage_summary)
    write_csv(args.output / "phase_summary.csv", phase_summary)
    write_csv(args.output / "temporal_summary.csv", temporal)
    (args.output / "trace_manifest.json").write_text(
        json.dumps(
            {
                "nfe_by_request": nfe_by_request,
                "denoise_forwards_by_request": {
                    request_id: len(rows) for request_id, rows in denoise_by_request.items()
                },
                "raw_rank_rows": len(ep_rows),
                "logical_invocations": len(logical_rows),
                "rank_rows_are_not_summed_as_request_latency": True,
                "timing_is_observer_heavy": True,
                "payload_is_a_semantic_bf16_lower_bound_excluding_protocol_metadata": True,
            }, indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
