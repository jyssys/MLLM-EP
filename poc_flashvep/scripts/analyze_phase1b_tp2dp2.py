"""Analyze selected-layer TP2/DP2 Phase 1b CUDA-event records."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STAGES = {
    "decoder_layer",
    "input_residual_rmsnorm",
    "attention_block",
    "post_attention_residual_rmsnorm",
    "router_projection",
    "router_topk",
    "dispatch_dpep_agrs",
    "local_expert_execution",
    "combine_dpep_agrs",
    "combine_tp_allreduce_after_dpep",
    "combine_tp_allgather_after_dpep",
    "moe_layer",
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "p90": float(_percentile(values, 0.9)),
        "mean": float(statistics.fmean(values)),
        "stddev": float(statistics.stdev(values) if len(values) > 1 else 0.0),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _request_stats(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    ranks: dict[str, Any] = {}
    for rank_result in result["rank_results"]:
        values = [float(row["wall_ms"]) for row in rank_result["iterations"]]
        ranks[str(rank_result["dp_rank"])] = _stats(values)
    return {"status": result["status"], "dp_ranks": ranks}


def _event_span(event: dict[str, Any]) -> tuple[float, float]:
    return (
        float(event["gpu_start_ms_from_rank_origin"]),
        float(event["gpu_end_ms_from_rank_origin"]),
    )


def _rank_metrics(events: dict[str, dict[str, Any]], tiles: int) -> dict[str, float]:
    layer_start, layer_end = _event_span(events["decoder_layer"])
    producer_start = _event_span(events["input_residual_rmsnorm"])[0]
    dispatch_end = _event_span(events["dispatch_dpep_agrs"])[1]
    expert_start, expert_end = _event_span(events["local_expert_execution"])
    combine_start = _event_span(events["combine_dpep_agrs"])[0]
    gather_end = _event_span(events["combine_tp_allgather_after_dpep"])[1]

    prelude = max(0.0, producer_start - layer_start)
    producer_span = max(0.0, dispatch_end - producer_start)
    handoff_gap = max(0.0, expert_start - dispatch_end)
    expert = float(events["local_expert_execution"]["duration_ms"])
    expert_to_combine_gap = max(0.0, combine_start - expert_end)
    combine_drain = max(0.0, gather_end - combine_start)
    postlude = max(0.0, layer_end - gather_end)
    first_tile_work = producer_span / tiles

    oracle = (
        prelude
        + first_tile_work
        + handoff_gap
        + max(producer_span - first_tile_work, expert)
        + expert_to_combine_gap
        + combine_drain
        + postlude
    )
    current = layer_end - layer_start
    return {
        "T_layer": current,
        "T_attention": float(events["attention_block"]["duration_ms"]),
        "T_norm_router": sum(
            float(events[name]["duration_ms"])
            for name in (
                "input_residual_rmsnorm",
                "post_attention_residual_rmsnorm",
                "router_projection",
                "router_topk",
            )
        ),
        "T_dispatch": float(events["dispatch_dpep_agrs"]["duration_ms"]),
        "T_expert": expert,
        "T_combine_dpep": float(events["combine_dpep_agrs"]["duration_ms"]),
        "T_combine_tp_allgather": float(
            events["combine_tp_allgather_after_dpep"]["duration_ms"]
        ),
        "T_combine_drain": combine_drain,
        "T_full_moe": float(events["moe_layer"]["duration_ms"]),
        "producer_span": producer_span,
        "first_tile_fill": prelude + first_tile_work + handoff_gap,
        "combine_drain": combine_drain,
        "current_stage_overlap": max(
            0.0,
            producer_span
            + handoff_gap
            + expert
            + expert_to_combine_gap
            + combine_drain
            - (gather_end - producer_start),
        ),
        "oracle": oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", type=int, default=400)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.events.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(row.get("duration_ms") is None for row in rows):
        raise RuntimeError("stage event contains a missing duration")
    observed_stages = {row["stage"] for row in rows}
    if observed_stages != STAGES:
        raise RuntimeError(f"stage mismatch: {observed_stages ^ STAGES}")

    grouped: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["iteration_id"]), int(row["layer"]), int(row["rank"]))
        grouped[key][row["stage"]] = row
    for key, stage_map in grouped.items():
        if set(stage_map) != STAGES:
            raise RuntimeError(f"incomplete stage coverage for {key}")

    by_iteration_layer: dict[tuple[int, int], list[tuple[int, dict[str, float]]]] = (
        defaultdict(list)
    )
    for (iteration, layer, rank), stage_map in grouped.items():
        by_iteration_layer[(iteration, layer)].append(
            (rank, _rank_metrics(stage_map, args.tiles))
        )

    samples: list[dict[str, Any]] = []
    for (iteration, layer), rank_rows in sorted(by_iteration_layer.items()):
        rank_rows.sort()
        current_rank, current_metrics = max(
            rank_rows, key=lambda item: item[1]["T_layer"]
        )
        expert_rank, expert_metrics = max(
            rank_rows, key=lambda item: item[1]["T_expert"]
        )
        oracle_rank, oracle_metrics = max(rank_rows, key=lambda item: item[1]["oracle"])

        def maximum(name: str) -> float:
            return max(metrics[name] for _, metrics in rank_rows)

        current = current_metrics["T_layer"]
        oracle = oracle_metrics["oracle"]
        samples.append(
            {
                "iteration_id": iteration,
                "layer": layer,
                "T_layer": current,
                "T_attention": maximum("T_attention"),
                "T_norm_router": maximum("T_norm_router"),
                "T_dispatch": maximum("T_dispatch"),
                "T_expert_max": expert_metrics["T_expert"],
                "T_combine_dpep": maximum("T_combine_dpep"),
                "T_combine_tp_allgather": maximum("T_combine_tp_allgather"),
                "T_combine": maximum("T_combine_drain"),
                "T_full_moe": maximum("T_full_moe"),
                "critical_layer_rank": current_rank,
                "critical_expert_rank": expert_rank,
                "critical_oracle_rank": oracle_rank,
                "expert_fraction": expert_metrics["T_expert"] / current,
                "exposed_nonexpert_fraction": 1.0
                - expert_metrics["T_expert"] / current,
                "dispatch_fraction": maximum("T_dispatch") / current,
                "combine_fraction": maximum("T_combine_drain") / current,
                "first_tile_fill": maximum("first_tile_fill"),
                "existing_stage_overlap": maximum("current_stage_overlap"),
                "oracle": oracle,
                "oracle_speedup": current / oracle,
            }
        )

    metrics = [
        "T_layer",
        "T_attention",
        "T_norm_router",
        "T_dispatch",
        "T_expert_max",
        "T_combine_dpep",
        "T_combine_tp_allgather",
        "T_combine",
        "T_full_moe",
        "expert_fraction",
        "exposed_nonexpert_fraction",
        "dispatch_fraction",
        "combine_fraction",
        "first_tile_fill",
        "existing_stage_overlap",
        "oracle",
        "oracle_speedup",
    ]
    layer_summary: dict[str, Any] = {}
    for layer in sorted({sample["layer"] for sample in samples}):
        selected = [sample for sample in samples if sample["layer"] == layer]
        layer_summary[str(layer)] = {
            name: _stats([float(sample[name]) for sample in selected])
            for name in metrics
        }
        layer_summary[str(layer)]["critical_layer_rank_counts"] = dict(
            sorted(Counter(sample["critical_layer_rank"] for sample in selected).items())
        )
        layer_summary[str(layer)]["critical_expert_rank_counts"] = dict(
            sorted(Counter(sample["critical_expert_rank"] for sample in selected).items())
        )

    request_samples: list[dict[str, float]] = []
    for iteration in sorted({sample["iteration_id"] for sample in samples}):
        selected = [sample for sample in samples if sample["iteration_id"] == iteration]
        current = sum(float(sample["T_layer"]) for sample in selected)
        oracle = sum(float(sample["oracle"]) for sample in selected)
        request_samples.append(
            {
                "iteration_id": iteration,
                "selected_layer_sum": current,
                "oracle_selected_layer_sum": oracle,
                "oracle_speedup": current / oracle,
            }
        )

    workload_summary: dict[str, Any] = {}
    workload_rows = [
        row
        for row in rows
        if row["stage"] == "local_expert_execution"
        and int(row["iteration_id"]) == 0
    ]
    for layer in sorted({int(row["layer"]) for row in workload_rows}):
        selected = sorted(
            (row for row in workload_rows if int(row["layer"]) == layer),
            key=lambda row: int(row["rank"]),
        )
        workload_summary[str(layer)] = {
            "kernel_assignments_including_padding_and_idle": sum(
                int(row["actual_local_assignments"]) for row in selected
            ),
            "real_request_visual_assignments": sum(
                int(row["real_request_visual_local_assignments"])
                for row in selected
            ),
            "real_request_text_special_assignments": sum(
                int(row["real_request_text_special_local_assignments"])
                for row in selected
            ),
            "tp_padding_assignments": sum(
                int(row["tp_padding_local_assignments"]) for row in selected
            ),
            "idle_dp_dummy_assignments": sum(
                int(row["idle_dp_dummy_local_assignments"]) for row in selected
            ),
            "per_rank": [
                {
                    "rank": int(row["rank"]),
                    "physical_gpu": int(row["physical_gpu"]),
                    "local_expert_range": [
                        int(row["local_expert_start"]),
                        int(row["local_expert_end"]),
                    ],
                    "actual_local_assignments": int(
                        row["actual_local_assignments"]
                    ),
                    "max_local_expert_batch": int(row["max_local_expert_batch"]),
                    "local_expert_token_counts": row["local_expert_token_counts"],
                }
                for row in selected
            ],
        }

    baseline = _request_stats(args.baseline)
    profile = _request_stats(args.profile)
    baseline_median = baseline["dp_ranks"]["0"]["median"]
    profile_median = profile["dp_ranks"]["0"]["median"]
    output = {
        "schema": "flashvep.phase1b.tp2dp2.analysis.v1",
        "event_count": len(rows),
        "coverage": {
            "iterations": sorted({sample["iteration_id"] for sample in samples}),
            "layers": sorted({sample["layer"] for sample in samples}),
            "ranks": sorted({key[2] for key in grouped}),
            "stages": sorted(STAGES),
        },
        "oracle_semantics": {
            "timeline": "same iteration/rank/layer CUDA-event timestamps",
            "tile_count": args.tiles,
            "first_tile_work": "measured producer span / 400 one-token tiles",
            "retained": [
                "prelude",
                "first-tile producer work and handoff",
                "expert-to-combine gap",
                "DPEP combine through TP all-gather drain",
                "decoder postlude",
            ],
            "overlap": "remaining producer span with the complete local expert window",
        },
        "baseline_request": baseline,
        "profile_request": profile,
        "profiler_overhead_fraction": profile_median / baseline_median - 1.0,
        "layer_summary": layer_summary,
        "workload_summary_iteration_0": workload_summary,
        "selected_layer_request_summary": {
            name: _stats([float(sample[name]) for sample in request_samples])
            for name in (
                "selected_layer_sum",
                "oracle_selected_layer_sum",
                "oracle_speedup",
            )
        },
        "samples": samples,
        "request_samples": request_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
