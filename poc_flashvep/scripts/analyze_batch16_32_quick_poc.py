"""Analyze one Batch-16/32 run using the Phase 1b timestamp model."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_phase1b_tp2dp2 import STAGES, _event_span, _rank_metrics, _stats


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _request_summary(
    path: Path, batch_size: int, expected_iterations: int
) -> dict[str, Any]:
    result = _read_json(path)
    if result.get("status") != "ok":
        raise RuntimeError(f"request run failed: {path}")
    expected_local = batch_size // 2
    by_iteration: dict[int, list[float]] = defaultdict(list)
    distributions: dict[str, int] = {}
    output_tokens: set[tuple[int, ...]] = set()
    prompt_token_counts: set[int] = set()
    for rank_result in result["rank_results"]:
        dp_rank = str(rank_result["dp_rank"])
        settings = rank_result["settings"]
        local_count = int(settings["real_requests_on_this_dp_rank"])
        if int(settings["global_batch_size"]) != batch_size:
            raise RuntimeError(f"global batch mismatch in {path}")
        if local_count != expected_local:
            raise RuntimeError(
                f"DP rank {dp_rank} received {local_count}, expected {expected_local}"
            )
        distributions[dp_rank] = local_count
        rows = rank_result["iterations"]
        if len(rows) != expected_iterations:
            raise RuntimeError(f"iteration count mismatch in {path}")
        for row in rows:
            if int(row["real_request_count"]) != expected_local:
                raise RuntimeError(f"request distribution changed in {path}")
            if int(row["actual_output_count"]) != expected_local:
                raise RuntimeError(f"output count mismatch in {path}")
            if int(row["total_prompt_token_count"]) != expected_local * 799:
                raise RuntimeError(f"prompt token total mismatch in {path}")
            by_iteration[int(row["iteration_id"])].append(float(row["wall_ms"]))
            prompt_token_counts.add(int(row["prompt_token_count"]))
            output_tokens.update(
                tuple(int(value) for value in token_ids)
                for token_ids in row["output_token_ids_per_request"]
            )
    if sorted(distributions) != ["0", "1"] or sum(distributions.values()) != batch_size:
        raise RuntimeError(f"incomplete DP distribution in {path}")
    if prompt_token_counts != {799}:
        raise RuntimeError(f"unexpected prompt length in {path}: {prompt_token_counts}")
    if output_tokens != {(1986,)}:
        raise RuntimeError(f"output token mismatch in {path}: {output_tokens}")
    if any(len(values) != 2 for values in by_iteration.values()):
        raise RuntimeError(f"missing DP wall time in {path}")
    critical_wall = [max(by_iteration[index]) for index in sorted(by_iteration)]
    return {
        "status": "ok",
        "requests_per_dp_rank": distributions,
        "actual_global_batch_size": sum(distributions.values()),
        "prompt_tokens_per_request": 799,
        "total_prompt_tokens": batch_size * 799,
        "output_token_ids": [1986],
        "critical_request_wall_ms": _stats(critical_wall),
    }


def _extended_rank_metrics(
    events: dict[str, dict[str, Any]], tiles: int
) -> dict[str, float]:
    metrics = _rank_metrics(events, tiles)
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
    best_case_extra_hidden = min(combine_drain, expert)
    unavoidable_lower_bound = (
        prelude
        + first_tile_work
        + handoff_gap
        + max(producer_span - first_tile_work, expert, combine_drain)
        + expert_to_combine_gap
        + postlude
    )
    metrics.update(
        {
            "best_case_extra_hidden": best_case_extra_hidden,
            "unavoidable_prelude_and_drain_lower_bound": unavoidable_lower_bound,
            "extended_oracle": max(
                metrics["oracle"] - best_case_extra_hidden,
                unavoidable_lower_bound,
            ),
        }
    )
    return metrics


def _batch1_reference(path: Path, layers: set[int]) -> dict[str, Any]:
    analysis = _read_json(path)
    samples = [
        sample for sample in analysis["samples"] if int(sample["layer"]) in layers
    ]
    metrics = (
        "T_layer",
        "T_attention",
        "T_norm_router",
        "T_dispatch",
        "T_expert_max",
        "T_combine",
        "T_full_moe",
        "expert_fraction",
        "dispatch_fraction",
        "combine_fraction",
    )
    aggregate = {
        name: _stats([float(sample[name]) for sample in samples]) for name in metrics
    }
    aggregate["communication_to_expert"] = _stats(
        [
            (float(sample["T_dispatch"]) + float(sample["T_combine"]))
            / float(sample["T_expert_max"])
            for sample in samples
        ]
    )
    by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_iteration[int(sample["iteration_id"])].append(sample)
    oracle_speedups = []
    for selected in by_iteration.values():
        current = sum(float(sample["T_layer"]) for sample in selected)
        oracle = sum(float(sample["oracle"]) for sample in selected)
        oracle_speedups.append(current / oracle)
    workload = analysis["workload_summary_iteration_0"]
    selected_workloads = [workload[str(layer)] for layer in sorted(layers)]
    aggregate.update(
        {
            "global_batch_size": 1,
            "requests_per_dp_rank": {"0": 1, "1": 0},
            "total_prompt_tokens": 799,
            "total_routed_assignments": 6392,
            "max_rank_assignments": float(
                statistics.median(
                    max(
                        int(rank["actual_local_assignments"])
                        for rank in layer["per_rank"]
                    )
                    for layer in selected_workloads
                )
            ),
            "max_local_expert_batch": float(
                statistics.median(
                    max(
                        int(rank["max_local_expert_batch"])
                        for rank in layer["per_rank"]
                    )
                    for layer in selected_workloads
                )
            ),
            "oracle_speedup": _stats(oracle_speedups),
            "profiler_overhead_fraction": float(
                analysis["profiler_overhead_fraction"]
            ),
        }
    )
    return aggregate


def _runtime_path(audit_path: Path, layers: set[int]) -> dict[str, Any]:
    rows = _read_jsonl(audit_path)
    runtime = [
        row
        for row in rows
        if row.get("kind") == "runtime_path" and int(row.get("layer", -1)) in layers
    ]
    if len({(int(row["rank"]), int(row["layer"])) for row in runtime}) != 12:
        raise RuntimeError("runtime-path coverage is incomplete")
    prepare = {row.get("prepare_finalize_class") for row in runtime}
    kernels = {row.get("fused_experts_class") for row in runtime}
    if prepare != {"MoEPrepareAndFinalizeNaiveDPEPModular"}:
        raise RuntimeError(f"DPEP prepare/finalize changed: {prepare}")
    collectives = Counter(
        row.get("collective") for row in rows if row.get("kind") == "collective_call"
    )
    if not collectives["dispatch_all_gatherv"] or not collectives["combine_reduce_scatterv"]:
        raise RuntimeError("required DPEP collectives were not observed")
    return {
        "prepare_finalize_classes": sorted(str(value) for value in prepare),
        "local_expert_backend_classes": sorted(str(value) for value in kernels),
        "all2all_manager": "AgRsAll2AllManager",
        "dispatch": "dispatch_all_gatherv",
        "combine": "combine_reduce_scatterv",
        "sequence_parallel_tp_combine": "tensor_model_parallel_all_gather",
        "runtime_records": len(runtime),
        "collective_call_counts": dict(sorted(collectives.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=(16, 32), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--batch1-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="12,24,36")
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    layers = {int(value) for value in args.layers.split(",")}
    if layers != {12, 24, 36}:
        raise ValueError("Quick PoC requires selected layers 12,24,36")
    expected_tokens = args.batch_size * 799
    expected_assignments = expected_tokens * 8

    rows = _read_jsonl(args.events)
    if any(row.get("duration_ms") is None or row.get("error") for row in rows):
        raise RuntimeError("stage event contains a missing/error duration")
    if {row["stage"] for row in rows} != STAGES:
        raise RuntimeError("stage coverage mismatch")
    call_offsets = sorted({int(row["wave_call_offset"]) for row in rows})
    expected_event_count = (
        args.iterations * len(layers) * 4 * len(STAGES) * len(call_offsets)
    )
    if len(rows) != expected_event_count:
        raise RuntimeError(f"unexpected event count: {len(rows)}")
    grouped: dict[
        tuple[int, int, int, int], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["iteration_id"]),
            int(row["layer"]),
            int(row["rank"]),
            int(row["wave_call_offset"]),
        )
        grouped[key][str(row["stage"])] = row
    if any(set(stage_map) != STAGES for stage_map in grouped.values()):
        raise RuntimeError("incomplete per-rank stage coverage")

    expert_events = [
        row for row in rows if row["stage"] == "local_expert_execution"
    ]
    schedule_by_iteration: list[dict[str, Any]] = []
    active_offsets_by_iteration: dict[int, list[int]] = {}
    schedule_patterns: Counter[str] = Counter()
    for iteration in range(args.iterations):
        model_microbatches: list[dict[str, Any]] = []
        for call_offset in call_offsets:
            selected = [
                row
                for row in expert_events
                if int(row["iteration_id"]) == iteration
                and int(row["wave_call_offset"]) == call_offset
            ]
            chunk_sets = {
                tuple(int(value) for value in row["dp_chunk_sizes"])
                for row in selected
            }
            row_counts = {int(row["topk_ids_shape"][0]) for row in selected}
            if len(chunk_sets) != 1 or len(row_counts) != 1:
                raise RuntimeError(
                    f"microbatch metadata disagrees at iteration {iteration}, "
                    f"offset {call_offset}"
                )
            chunks = list(next(iter(chunk_sets)))
            if len(chunks) != 4:
                raise RuntimeError(f"unexpected DP/TP chunks: {chunks}")
            requests = [
                sum(chunks[offset : offset + 2]) // 799
                for offset in range(0, len(chunks), 2)
            ]
            detailed = [
                tuple(int(value) for value in row["observed_real_requests_by_dp_rank"])
                for row in selected
                if "observed_real_requests_by_dp_rank" in row
            ]
            if detailed and set(detailed) != {tuple(requests)}:
                raise RuntimeError(
                    f"derived request count disagrees at iteration {iteration}, "
                    f"offset {call_offset}: {requests} versus {set(detailed)}"
                )
            model_microbatches.append(
                {
                    "wave_call_offset": call_offset,
                    "dp_chunk_sizes": chunks,
                    "kernel_token_rows": next(iter(row_counts)),
                    "real_requests_per_dp_rank": requests,
                    "global_real_requests": sum(requests),
                    "real_prompt_tokens": sum(requests) * 799,
                    "first_token_tiles_per_rank": max(chunks),
                }
            )
        active = [
            item for item in model_microbatches if item["global_real_requests"] > 0
        ]
        if sum(item["global_real_requests"] for item in active) != args.batch_size:
            raise RuntimeError(
                f"iteration {iteration} did not form global batch {args.batch_size}: "
                f"{model_microbatches}"
            )
        if [
            sum(item["real_requests_per_dp_rank"][rank] for item in active)
            for rank in range(2)
        ] != [args.batch_size // 2, args.batch_size // 2]:
            raise RuntimeError(f"DP request distribution changed: {active}")
        active_offsets_by_iteration[iteration] = [
            int(item["wave_call_offset"]) for item in active
        ]
        pattern = json.dumps(
            [item["real_requests_per_dp_rank"] for item in active],
            separators=(",", ":"),
        )
        schedule_patterns[pattern] += 1
        schedule_by_iteration.append(
            {
                "iteration_id": iteration,
                "active_model_microbatch_count": len(active),
                "microbatches": model_microbatches,
            }
        )

    samples: list[dict[str, Any]] = []
    workload_by_layer: dict[str, Any] = {}
    summed_metric_names = (
        "T_layer",
        "T_attention",
        "T_norm_router",
        "T_dispatch",
        "T_expert",
        "T_combine_dpep",
        "T_combine_tp_allgather",
        "T_combine_drain",
        "T_full_moe",
        "first_tile_fill",
        "oracle",
        "best_case_extra_hidden",
        "extended_oracle",
    )
    for iteration, layer in sorted({key[:2] for key in grouped}):
        rank_rows = []
        for rank in range(4):
            stage_maps = [
                grouped[(iteration, layer, rank, call_offset)]
                for call_offset in active_offsets_by_iteration[iteration]
            ]
            submetrics = [
                _extended_rank_metrics(
                    stage_map,
                    max(
                        int(value)
                        for value in stage_map["local_expert_execution"][
                            "dp_chunk_sizes"
                        ]
                    ),
                )
                for stage_map in stage_maps
            ]
            summed = {
                name: sum(float(metrics[name]) for metrics in submetrics)
                for name in summed_metric_names
            }
            rank_rows.append((rank, stage_maps, summed))
        current_rank, _, current_metrics = max(
            rank_rows, key=lambda item: item[2]["T_layer"]
        )
        expert_rank, _, expert_metrics = max(
            rank_rows, key=lambda item: item[2]["T_expert"]
        )
        oracle_rank, _, oracle_metrics = max(
            rank_rows, key=lambda item: item[2]["oracle"]
        )
        extended_rank, _, extended_metrics = max(
            rank_rows, key=lambda item: item[2]["extended_oracle"]
        )

        def maximum(name: str) -> float:
            return max(float(metrics[name]) for _, _, metrics in rank_rows)

        current = float(current_metrics["T_layer"])
        expert = float(expert_metrics["T_expert"])
        oracle = float(oracle_metrics["oracle"])
        extended = float(extended_metrics["extended_oracle"])
        samples.append(
            {
                "iteration_id": iteration,
                "layer": layer,
                "T_layer": current,
                "T_attention": maximum("T_attention"),
                "T_norm_router": maximum("T_norm_router"),
                "T_dispatch": maximum("T_dispatch"),
                "T_expert_max": expert,
                "T_combine_dpep": maximum("T_combine_dpep"),
                "T_combine_tp_allgather": maximum("T_combine_tp_allgather"),
                "T_combine": maximum("T_combine_drain"),
                "T_full_moe": maximum("T_full_moe"),
                "expert_fraction": expert / current,
                "dispatch_fraction": maximum("T_dispatch") / current,
                "combine_fraction": maximum("T_combine_drain") / current,
                "communication_to_expert": (
                    maximum("T_dispatch") + maximum("T_combine_drain")
                )
                / expert,
                "critical_layer_rank": current_rank,
                "critical_expert_rank": expert_rank,
                "critical_oracle_rank": oracle_rank,
                "critical_extended_oracle_rank": extended_rank,
                "first_tile_fill": maximum("first_tile_fill"),
                "oracle": oracle,
                "oracle_speedup": current / oracle,
                "best_case_extra_hidden": maximum("best_case_extra_hidden"),
                "extended_oracle": extended,
                "extended_oracle_speedup": current / extended,
            }
        )
        if iteration == 0:
            per_rank = []
            for rank, stage_maps, _ in rank_rows:
                events = [
                    stage_map["local_expert_execution"] for stage_map in stage_maps
                ]
                per_rank.append(
                    {
                        "rank": rank,
                        "physical_gpu": int(events[0]["physical_gpu"]),
                        "kernel_assignments_including_padding": sum(
                            int(event["actual_local_assignments"])
                            for event in events
                        ),
                        "real_request_local_assignments": sum(
                            int(event["real_request_local_assignments"])
                            for event in events
                        ),
                        "tp_padding_local_assignments": sum(
                            int(event["tp_padding_local_assignments"])
                            for event in events
                        ),
                        "idle_dp_dummy_local_assignments": sum(
                            int(event["idle_dp_dummy_local_assignments"])
                            for event in events
                        ),
                        "max_local_expert_batch": max(
                            int(event["max_local_expert_batch"])
                            for event in events
                        ),
                        "active_local_experts_min": min(
                            int(event["active_local_experts"])
                            for event in events
                        ),
                        "active_local_experts_max": max(
                            int(event["active_local_experts"])
                            for event in events
                        ),
                        "local_assignments_by_dp_rank": [
                            sum(
                                int(event["local_assignments_by_dp_rank"][dp_rank])
                                for event in events
                            )
                            for dp_rank in range(2)
                        ],
                    }
                )
            total_real = sum(
                rank["real_request_local_assignments"] for rank in per_rank
            )
            total_padding = sum(
                rank["tp_padding_local_assignments"] for rank in per_rank
            )
            total_idle_dummy = sum(
                rank["idle_dp_dummy_local_assignments"] for rank in per_rank
            )
            if total_real != expected_assignments:
                raise RuntimeError(
                    f"layer {layer} has {total_real} real assignments, "
                    f"expected {expected_assignments}"
                )
            workload_by_layer[str(layer)] = {
                "total_real_routed_assignments": total_real,
                "total_padding_assignments": total_padding,
                "total_idle_dp_dummy_assignments": total_idle_dummy,
                "kernel_assignments_including_padding": (
                    total_real + total_padding + total_idle_dummy
                ),
                "max_rank_assignments": max(
                    rank["real_request_local_assignments"] for rank in per_rank
                ),
                "max_local_expert_batch": max(
                    rank["max_local_expert_batch"] for rank in per_rank
                ),
                "per_rank": per_rank,
            }

    metric_names = (
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
        "dispatch_fraction",
        "combine_fraction",
        "communication_to_expert",
        "first_tile_fill",
        "oracle_speedup",
        "best_case_extra_hidden",
        "extended_oracle_speedup",
    )
    aggregate = {
        name: _stats([float(sample[name]) for sample in samples])
        for name in metric_names
    }
    layer_summary: dict[str, Any] = {}
    for layer in sorted(layers):
        selected = [sample for sample in samples if sample["layer"] == layer]
        layer_summary[str(layer)] = {
            name: _stats([float(sample[name]) for sample in selected])
            for name in metric_names
        }
        layer_summary[str(layer)]["critical_layer_rank_counts"] = dict(
            sorted(Counter(sample["critical_layer_rank"] for sample in selected).items())
        )
        layer_summary[str(layer)]["critical_expert_rank_counts"] = dict(
            sorted(Counter(sample["critical_expert_rank"] for sample in selected).items())
        )

    request_oracles = []
    for iteration in range(args.iterations):
        selected = [sample for sample in samples if sample["iteration_id"] == iteration]
        current = sum(float(sample["T_layer"]) for sample in selected)
        oracle = sum(float(sample["oracle"]) for sample in selected)
        extended = sum(float(sample["extended_oracle"]) for sample in selected)
        request_oracles.append(
            {
                "iteration_id": iteration,
                "selected_layer_sum_ms": current,
                "oracle_selected_layer_sum_ms": oracle,
                "oracle_speedup": current / oracle,
                "extended_oracle_selected_layer_sum_ms": extended,
                "extended_oracle_speedup": current / extended,
            }
        )
    aggregate["oracle_selected_layer_speedup"] = _stats(
        [row["oracle_speedup"] for row in request_oracles]
    )
    aggregate["extended_oracle_selected_layer_speedup"] = _stats(
        [row["extended_oracle_speedup"] for row in request_oracles]
    )
    aggregate.update(
        {
            "global_batch_size": args.batch_size,
            "total_prompt_tokens": expected_tokens,
            "total_routed_assignments": expected_assignments,
            "max_rank_assignments": float(
                statistics.median(
                    value["max_rank_assignments"]
                    for value in workload_by_layer.values()
                )
            ),
            "max_local_expert_batch": float(
                statistics.median(
                    value["max_local_expert_batch"]
                    for value in workload_by_layer.values()
                )
            ),
        }
    )

    baseline_request = _request_summary(
        args.baseline, args.batch_size, args.iterations
    )
    profile_request = _request_summary(args.profile, args.batch_size, args.iterations)
    baseline_wall = float(baseline_request["critical_request_wall_ms"]["median"])
    profile_wall = float(profile_request["critical_request_wall_ms"]["median"])
    overhead = profile_wall / baseline_wall - 1.0
    aggregate["profiler_overhead_fraction"] = overhead
    batch1 = _batch1_reference(args.batch1_analysis, layers)
    scaling = {
        "expert_scaling": aggregate["T_expert_max"]["median"]
        / batch1["T_expert_max"]["median"],
        "dispatch_scaling": aggregate["T_dispatch"]["median"]
        / batch1["T_dispatch"]["median"],
        "combine_scaling": aggregate["T_combine"]["median"]
        / batch1["T_combine"]["median"],
        "layer_scaling": aggregate["T_layer"]["median"]
        / batch1["T_layer"]["median"],
    }

    output = {
        "schema": "flashvep.batch16_32.quick_poc.analysis.v1",
        "batch_size": args.batch_size,
        "layers": sorted(layers),
        "iterations": args.iterations,
        "event_count": len(rows),
        "model_microbatch_schedule_by_iteration": schedule_by_iteration,
        "model_microbatch_schedule_pattern_counts": dict(
            sorted(schedule_patterns.items())
        ),
        "runtime_path": _runtime_path(args.audit, layers),
        "baseline_request": baseline_request,
        "profile_request": profile_request,
        "profiler_overhead_fraction": overhead,
        "workload_by_layer": workload_by_layer,
        "aggregate": aggregate,
        "layer_summary": layer_summary,
        "batch1_reference_layers_12_24_36": batch1,
        "scaling_vs_batch1": scaling,
        "request_oracles": request_oracles,
        "samples": samples,
        "preconditions": {
            "actual_global_batch_across_model_microbatches": True,
            "model_microbatch_count_range": [
                min(
                    row["active_model_microbatch_count"]
                    for row in schedule_by_iteration
                ),
                max(
                    row["active_model_microbatch_count"]
                    for row in schedule_by_iteration
                ),
            ],
            "balanced_real_requests_across_dp_ranks": True,
            "actual_dpep_dispatch_combine": True,
            "output_token_consistency": True,
            "profiler_overhead_below_20_percent": overhead < 0.20,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "profiler_overhead_percent": overhead * 100.0,
                "expert_ms": aggregate["T_expert_max"]["median"],
                "oracle_speedup": aggregate["oracle_selected_layer_speedup"]["median"],
                "extended_oracle_speedup": aggregate[
                    "extended_oracle_selected_layer_speedup"
                ]["median"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
