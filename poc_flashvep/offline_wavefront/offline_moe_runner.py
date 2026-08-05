"""Opt-in in-worker capture and scheduler-free TritonExperts D/E/C replay.

The vLLM request is used only to obtain one real layer-24 workload and the
already-loaded expert weights.  Once the hook is entered, every timed loop is
an explicit operator replay: no request is submitted to or advanced by the
serving scheduler while the replay is running.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from .capture_schema import CaptureMetadata, SCHEMA_VERSION, validate_capture
from .expert_centered_pipeline import overlap_summary
from .workload_builder import (
    RepeatedWorkload,
    build_repeated_workload,
    rank_slice,
    workload_metrics,
)


_INSTALLED = False
_RAN_RANKS: set[int] = set()
_CONTEXT = threading.local()


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "median_ms": float(statistics.median(values)),
        "p90_ms": float(percentile(0.9)),
        "mean_ms": float(statistics.fmean(values)),
        "stddev_ms": float(statistics.stdev(values) if len(values) > 1 else 0.0),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _layer_from_prefix(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _current_layer() -> int:
    return int(getattr(_CONTEXT, "layer", -1))


def _dp_chunk_sizes() -> list[int] | None:
    try:
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().dp_metadata
        if metadata is None:
            return None
        sizes = metadata.get_chunk_sizes_across_dp_rank()
        return [int(value) for value in sizes] if sizes is not None else None
    except Exception:
        return None


@dataclass
class ExpertSpec:
    fused: Any
    in_dtype: torch.dtype
    w1: torch.Tensor
    w2: torch.Tensor
    activation: Any
    global_num_experts: int
    local_num_experts: int
    expert_map: torch.Tensor | None
    apply_router_weight_on_input: bool


@dataclass
class ReplayBuffers:
    local_hidden: list[torch.Tensor]
    local_weights: list[torch.Tensor]
    local_ids: list[torch.Tensor]
    gathered_hidden: list[torch.Tensor]
    gathered_weights: list[torch.Tensor]
    gathered_ids: list[torch.Tensor]
    expert_outputs: list[torch.Tensor]
    local_outputs: list[torch.Tensor]
    workspace13: torch.Tensor
    workspace2: torch.Tensor


def _allocate_replay_buffers(
    workload: RepeatedWorkload,
    *,
    rank: int,
    ep_size: int,
    microbatches: int,
    spec: ExpertSpec,
) -> ReplayBuffers:
    local = rank_slice(workload.token_count, ep_size, rank)
    local_hidden_full = workload.hidden[local].contiguous()
    local_weights_full = workload.topk_weights[local].contiguous()
    local_ids_full = workload.topk_ids[local].contiguous()
    if local_hidden_full.shape[0] % microbatches:
        raise ValueError("local tokens must divide evenly into microbatches")
    local_hidden = list(local_hidden_full.chunk(microbatches, dim=0))
    local_weights = list(local_weights_full.chunk(microbatches, dim=0))
    local_ids = list(local_ids_full.chunk(microbatches, dim=0))

    gathered_hidden: list[torch.Tensor] = []
    gathered_weights: list[torch.Tensor] = []
    gathered_ids: list[torch.Tensor] = []
    expert_outputs: list[torch.Tensor] = []
    local_outputs: list[torch.Tensor] = []
    for hidden, weights, ids in zip(local_hidden, local_weights, local_ids):
        global_tokens = int(hidden.shape[0]) * ep_size
        gathered_hidden.append(
            torch.empty(
                (global_tokens, hidden.shape[1]), dtype=hidden.dtype, device=hidden.device
            )
        )
        gathered_weights.append(
            torch.empty(
                (global_tokens, weights.shape[1]),
                dtype=weights.dtype,
                device=weights.device,
            )
        )
        gathered_ids.append(
            torch.empty(
                (global_tokens, ids.shape[1]), dtype=ids.dtype, device=ids.device
            )
        )
        expert_outputs.append(
            torch.empty(
                (global_tokens, hidden.shape[1]),
                dtype=spec.in_dtype,
                device=hidden.device,
            )
        )
        local_outputs.append(torch.empty_like(hidden, dtype=spec.in_dtype))

    sample_hidden = gathered_hidden[0]
    sample_ids = gathered_ids[0]
    _, tokens, n_dim, k_dim, top_k = spec.fused.moe_problem_size(
        sample_hidden, spec.w1, spec.w2, sample_ids
    )
    workspace13_shape, workspace2_shape, _ = spec.fused.workspace_shapes(
        tokens,
        n_dim,
        k_dim,
        top_k,
        spec.global_num_experts,
        spec.local_num_experts,
        None,
        spec.activation,
    )
    workspace_dtype = spec.fused.workspace_dtype(spec.in_dtype)
    workspace13 = torch.empty(
        workspace13_shape, dtype=workspace_dtype, device=sample_hidden.device
    )
    workspace2 = torch.empty(
        workspace2_shape, dtype=workspace_dtype, device=sample_hidden.device
    )
    return ReplayBuffers(
        local_hidden=local_hidden,
        local_weights=local_weights,
        local_ids=local_ids,
        gathered_hidden=gathered_hidden,
        gathered_weights=gathered_weights,
        gathered_ids=gathered_ids,
        expert_outputs=expert_outputs,
        local_outputs=local_outputs,
        workspace13=workspace13,
        workspace2=workspace2,
    )


def _dispatch(index: int, buffers: ReplayBuffers, group: Any) -> None:
    dist.all_gather_into_tensor(
        buffers.gathered_hidden[index], buffers.local_hidden[index], group=group
    )
    dist.all_gather_into_tensor(
        buffers.gathered_weights[index], buffers.local_weights[index], group=group
    )
    dist.all_gather_into_tensor(
        buffers.gathered_ids[index], buffers.local_ids[index], group=group
    )


def _expert(index: int, buffers: ReplayBuffers, spec: ExpertSpec) -> None:
    spec.fused.apply(
        output=buffers.expert_outputs[index],
        hidden_states=buffers.gathered_hidden[index],
        w1=spec.w1,
        w2=spec.w2,
        topk_weights=buffers.gathered_weights[index],
        topk_ids=buffers.gathered_ids[index],
        activation=spec.activation,
        global_num_experts=spec.global_num_experts,
        expert_map=spec.expert_map,
        a1q_scale=None,
        a2_scale=spec.fused.a2_scale,
        workspace13=buffers.workspace13,
        workspace2=buffers.workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=spec.apply_router_weight_on_input,
    )


def _combine(index: int, buffers: ReplayBuffers, group: Any) -> None:
    dist.reduce_scatter_tensor(
        buffers.local_outputs[index], buffers.expert_outputs[index], group=group
    )


def _serial_iteration(
    buffers: ReplayBuffers,
    spec: ExpertSpec,
    group: Any,
    stream: torch.cuda.Stream,
    *,
    measure: bool,
) -> dict[str, Any] | None:
    wall_start, wall_end = _event(), _event()
    dispatch_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    expert_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    combine_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with torch.cuda.stream(stream):
        wall_start.record(stream)
        for index in range(len(buffers.local_hidden)):
            start, end = _event(), _event()
            start.record(stream)
            _dispatch(index, buffers, group)
            end.record(stream)
            dispatch_events.append((start, end))

            start, end = _event(), _event()
            start.record(stream)
            _expert(index, buffers, spec)
            end.record(stream)
            expert_events.append((start, end))

            start, end = _event(), _event()
            start.record(stream)
            _combine(index, buffers, group)
            end.record(stream)
            combine_events.append((start, end))
        wall_end.record(stream)
    wall_end.synchronize()
    if not measure:
        return None
    return {
        "wall_ms": _elapsed(wall_start, wall_end),
        "dispatch_ms": sum(_elapsed(start, end) for start, end in dispatch_events),
        "expert_ms": sum(_elapsed(start, end) for start, end in expert_events),
        "combine_ms": sum(_elapsed(start, end) for start, end in combine_events),
    }


def _wavefront_iteration(
    buffers: ReplayBuffers,
    spec: ExpertSpec,
    group: Any,
    dispatch_stream: torch.cuda.Stream,
    expert_stream: torch.cuda.Stream,
    combine_stream: torch.cuda.Stream,
    *,
    measure: bool,
) -> dict[str, Any] | None:
    default_stream = torch.cuda.current_stream()
    origin, wall_end = _event(), _event()
    origin.record(default_stream)
    for stream in (dispatch_stream, expert_stream, combine_stream):
        stream.wait_event(origin)

    dispatch_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    expert_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    combine_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def enqueue_dispatch(index: int, wait_for: torch.cuda.Event | None) -> torch.cuda.Event:
        start, end = _event(), _event()
        with torch.cuda.stream(dispatch_stream):
            if wait_for is not None:
                dispatch_stream.wait_event(wait_for)
            start.record(dispatch_stream)
            _dispatch(index, buffers, group)
            end.record(dispatch_stream)
        dispatch_events.append((start, end))
        return end

    def enqueue_expert(index: int, dispatch_done: torch.cuda.Event) -> torch.cuda.Event:
        start, end = _event(), _event()
        with torch.cuda.stream(expert_stream):
            expert_stream.wait_event(dispatch_done)
            start.record(expert_stream)
            _expert(index, buffers, spec)
            end.record(expert_stream)
        expert_events.append((start, end))
        return end

    def enqueue_combine(
        index: int,
        expert_done: torch.cuda.Event,
        communication_done: torch.cuda.Event,
    ) -> torch.cuda.Event:
        start, end = _event(), _event()
        with torch.cuda.stream(combine_stream):
            combine_stream.wait_event(expert_done)
            combine_stream.wait_event(communication_done)
            start.record(combine_stream)
            _combine(index, buffers, group)
            end.record(combine_stream)
        combine_events.append((start, end))
        return end

    dispatch_done = enqueue_dispatch(0, None)
    expert_done = enqueue_expert(0, dispatch_done)
    communication_done = dispatch_done
    for index in range(len(buffers.local_hidden) - 1):
        next_dispatch_done = enqueue_dispatch(index + 1, communication_done)
        combine_done = enqueue_combine(index, expert_done, next_dispatch_done)
        expert_done = enqueue_expert(index + 1, next_dispatch_done)
        communication_done = combine_done
    final_combine = enqueue_combine(
        len(buffers.local_hidden) - 1, expert_done, communication_done
    )
    default_stream.wait_event(final_combine)
    wall_end.record(default_stream)
    wall_end.synchronize()
    if not measure:
        return None

    def intervals(
        values: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> list[tuple[float, float]]:
        return [(_elapsed(origin, start), _elapsed(origin, end)) for start, end in values]

    dispatch_intervals = intervals(dispatch_events)
    expert_intervals = intervals(expert_events)
    combine_intervals = intervals(combine_events)
    return {
        "wall_ms": _elapsed(origin, wall_end),
        "dispatch_ms": sum(end - start for start, end in dispatch_intervals),
        "expert_ms": sum(end - start for start, end in expert_intervals),
        "combine_ms": sum(end - start for start, end in combine_intervals),
        **overlap_summary(dispatch_intervals, expert_intervals, combine_intervals),
    }


def _run_variant(
    variant: str,
    buffers: ReplayBuffers,
    spec: ExpertSpec,
    group: Any,
    warmups: int,
    iterations: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    serial_stream = torch.cuda.Stream()
    dispatch_stream = torch.cuda.Stream()
    expert_stream = torch.cuda.Stream()
    combine_stream = torch.cuda.Stream()
    run: Callable[..., dict[str, Any] | None]
    if variant in {"full_batch_serial", "microbatch_serial"}:
        run = lambda measure: _serial_iteration(  # noqa: E731
            buffers, spec, group, serial_stream, measure=measure
        )
    elif variant == "expert_centered_wavefront":
        run = lambda measure: _wavefront_iteration(  # noqa: E731
            buffers,
            spec,
            group,
            dispatch_stream,
            expert_stream,
            combine_stream,
            measure=measure,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    for _ in range(warmups):
        run(False)
    samples = [run(True) for _ in range(iterations)]
    measured = [sample for sample in samples if sample is not None]
    result: dict[str, Any] = {
        "variant": variant,
        "warmups": warmups,
        "iterations": iterations,
        "wall_ms": [float(sample["wall_ms"]) for sample in measured],
        "dispatch_ms": [float(sample["dispatch_ms"]) for sample in measured],
        "expert_ms": [float(sample["expert_ms"]) for sample in measured],
        "combine_ms": [float(sample["combine_ms"]) for sample in measured],
    }
    for key in ("wall_ms", "dispatch_ms", "expert_ms", "combine_ms"):
        result[f"{key}_stats"] = _stats(result[key])
    if variant == "expert_centered_wavefront":
        for key in (
            "dispatch_expert_overlap_ms",
            "expert_combine_overlap_ms",
            "actual_overlap_fraction",
        ):
            result[key] = [float(sample[key]) for sample in measured]
            result[f"{key}_stats"] = _stats(result[key])
    output = torch.cat(buffers.local_outputs, dim=0).detach().clone()
    return result, output


def _correctness(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    difference = (reference.float() - candidate.float()).abs()
    ref_flat = reference.float().reshape(-1)
    candidate_flat = candidate.float().reshape(-1)
    denominator = float(ref_flat.norm().item() * candidate_flat.norm().item())
    cosine = (
        float(torch.dot(ref_flat, candidate_flat).item()) / denominator
        if denominator
        else 1.0
    )
    passed = True
    error = None
    try:
        torch.testing.assert_close(candidate, reference, rtol=1e-2, atol=1e-2)
    except AssertionError as exc:
        passed = False
        error = str(exc).splitlines()[0]
    return {
        "passed": passed,
        "rtol": 1e-2,
        "atol": 1e-2,
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "cosine_similarity": cosine,
        "assert_close_error": error,
    }


def _capture(
    hidden: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    spec: ExpertSpec,
    ep_size: int,
) -> dict[str, Any]:
    token_count = _int_env("FLASHVEP_OFFLINE_ORIGINAL_TOKENS", 799)
    hidden = hidden[:token_count].detach().to(torch.bfloat16).clone()
    ids = ids[:token_count].detach().clone()
    weights = weights[:token_count].detach().float().clone()
    local_experts = spec.global_num_experts // ep_size
    metadata = CaptureMetadata(
        schema_version=SCHEMA_VERSION,
        model_path=os.environ.get("FLASHVEP_OFFLINE_MODEL_PATH", "unknown"),
        layer=_int_env("FLASHVEP_OFFLINE_LAYER", 24),
        dtype="torch.bfloat16",
        original_token_count=token_count,
        vision_token_count=_int_env("FLASHVEP_OFFLINE_VISION_TOKENS", 784),
        hidden_size=int(hidden.shape[1]),
        expert_intermediate_size=int(spec.w2.shape[-1]),
        top_k=int(ids.shape[1]),
        global_num_experts=spec.global_num_experts,
        ep_size=ep_size,
        local_experts_per_rank=local_experts,
        source=(
            "Qwen3 layer-24 post-attention RMSNorm output reconstructed from the "
            "value-preserving BF16 AgRs dispatch input"
        ),
    )
    capture = {
        "metadata": metadata.to_dict(),
        "post_attention_hidden": hidden.cpu(),
        "topk_expert_ids": ids.cpu(),
        "topk_weights": weights.cpu(),
        "destination_rank": torch.div(
            ids, local_experts, rounding_mode="floor"
        ).cpu(),
        "local_expert_id": torch.remainder(ids, local_experts).cpu(),
    }
    validate_capture(capture)
    return capture


def _route_identity(buffers: ReplayBuffers, workload: RepeatedWorkload) -> bool:
    gathered = torch.cat(buffers.gathered_ids, dim=0)
    return bool(torch.equal(gathered, workload.topk_ids))


def _run_benchmark(capture: dict[str, Any], spec: ExpertSpec) -> dict[str, Any]:
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    rank = int(ep.rank_in_group)
    ep_size = int(ep.world_size)
    group = ep.device_group
    warmups = _int_env("FLASHVEP_OFFLINE_WARMUPS", 10)
    iterations = _int_env("FLASHVEP_OFFLINE_ITERATIONS", 30)
    batch_values = [16, 32, 64, 128]
    base_hidden = capture["post_attention_hidden"].to(spec.w1.device)
    base_ids = capture["topk_expert_ids"].to(spec.w1.device)
    base_weights = capture["topk_weights"].to(spec.w1.device)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "rank": rank,
        "physical_gpu": [4, 5, 6, 7][rank],
        "settings": {
            "physical_gpus": [4, 5, 6, 7],
            "ep_size": ep_size,
            "layer": int(capture["metadata"]["layer"]),
            "dtype": str(spec.in_dtype),
            "expert_backend": type(spec.fused).__name__,
            "communication_backend": "torch.distributed NCCL all_gather_into_tensor/reduce_scatter_tensor",
            "collective_semantics": "AgRsAll2AllManager equivalent",
            "scheduler_free_replay": True,
            "warmups": warmups,
            "iterations": iterations,
        },
        "o1": [],
        "o2": [],
    }
    o1_critical_inputs: list[tuple[int, float, float, float]] = []

    for batch_equivalent in batch_values:
        workload = build_repeated_workload(
            base_hidden, base_ids, base_weights, batch_equivalent
        )
        torch.cuda.synchronize()
        memory_before = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        buffers = _allocate_replay_buffers(
            workload, rank=rank, ep_size=ep_size, microbatches=1, spec=spec
        )
        variant, output = _run_variant(
            "full_batch_serial", buffers, spec, group, warmups, iterations
        )
        route_identity = _route_identity(buffers, workload)
        wall = float(variant["wall_ms_stats"]["median_ms"])
        dispatch_ms = float(variant["dispatch_ms_stats"]["median_ms"])
        expert_ms = float(variant["expert_ms_stats"]["median_ms"])
        combine_ms = float(variant["combine_ms_stats"]["median_ms"])
        entry = {
            "batch_equivalent": batch_equivalent,
            "workload": workload_metrics(
                workload,
                vision_tokens_per_request=int(capture["metadata"]["vision_token_count"]),
                ep_size=ep_size,
                local_experts_per_rank=int(
                    capture["metadata"]["local_experts_per_rank"]
                ),
            ),
            **variant,
            "expert_fraction": expert_ms / wall,
            "communication_to_expert": (dispatch_ms + combine_ms) / expert_ms,
            "tokens_per_second": workload.token_count * 1000.0 / wall,
            "assignments_per_second": workload.assignment_count * 1000.0 / wall,
            "route_identity": route_identity,
            "token_order_restoration": bool(
                output.shape[0] == workload.token_count // ep_size
            ),
            "memory_allocated_before_bytes": memory_before,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "incremental_peak_memory_bytes": int(
                torch.cuda.max_memory_allocated() - memory_before
            ),
        }
        result["o1"].append(entry)
        o1_critical_inputs.append(
            (batch_equivalent, dispatch_ms, expert_ms, combine_ms)
        )
        del output, buffers, workload
        torch.cuda.empty_cache()

    critical_rows: list[tuple[int, float, float, float]] = []
    for batch_equivalent, dispatch_ms, expert_ms, combine_ms in o1_critical_inputs:
        values = torch.tensor(
            [dispatch_ms, expert_ms, combine_ms],
            dtype=torch.float64,
            device=spec.w1.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
        critical_rows.append(
            (batch_equivalent, *(float(value) for value in values.tolist()))
        )
    preferred = [
        batch
        for batch, dispatch_ms, expert_ms, combine_ms in critical_rows
        if batch in (32, 64)
        and expert_ms / (dispatch_ms + expert_ms + combine_ms) >= 0.25
        and expert_ms >= 1.0
    ]
    eligible = [
        batch
        for batch, dispatch_ms, expert_ms, combine_ms in critical_rows
        if expert_ms / (dispatch_ms + expert_ms + combine_ms) >= 0.25
        and expert_ms >= 1.0
    ]
    candidates = preferred + [batch for batch in eligible if batch not in preferred]
    if len(candidates) < 2:
        candidates = [row[0] for row in sorted(critical_rows, key=lambda row: row[2])[-2:]]
    candidates = candidates[:2]
    result["o2_candidate_selection"] = {
        "critical_rank_stage_medians": [
            {
                "batch_equivalent": batch,
                "dispatch_ms": dispatch_ms,
                "expert_ms": expert_ms,
                "combine_ms": combine_ms,
                "expert_fraction": expert_ms / (dispatch_ms + expert_ms + combine_ms),
            }
            for batch, dispatch_ms, expert_ms, combine_ms in critical_rows
        ],
        "selected": candidates,
        "rule": "prefer B_eq 32/64 when expert>=1ms and expert_fraction>=25%; otherwise next eligible",
    }

    correctness_failed = False
    for batch_equivalent in candidates:
        workload = build_repeated_workload(
            base_hidden, base_ids, base_weights, batch_equivalent
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        full_buffers = _allocate_replay_buffers(
            workload, rank=rank, ep_size=ep_size, microbatches=1, spec=spec
        )
        full, reference = _run_variant(
            "full_batch_serial", full_buffers, spec, group, warmups, iterations
        )
        full_route_identity = _route_identity(full_buffers, workload)
        full_memory = int(torch.cuda.max_memory_allocated())
        del full_buffers
        torch.cuda.empty_cache()

        for microbatches in (2, 4):
            torch.cuda.synchronize()
            memory_before = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            serial_buffers = _allocate_replay_buffers(
                workload,
                rank=rank,
                ep_size=ep_size,
                microbatches=microbatches,
                spec=spec,
            )
            micro_serial, micro_output = _run_variant(
                "microbatch_serial",
                serial_buffers,
                spec,
                group,
                warmups,
                iterations,
            )
            serial_route_identity = _route_identity(serial_buffers, workload)
            serial_correctness = _correctness(reference, micro_output)
            del micro_output, serial_buffers
            torch.cuda.empty_cache()

            wave_buffers = _allocate_replay_buffers(
                workload,
                rank=rank,
                ep_size=ep_size,
                microbatches=microbatches,
                spec=spec,
            )
            wavefront, wave_output = _run_variant(
                "expert_centered_wavefront",
                wave_buffers,
                spec,
                group,
                warmups,
                iterations,
            )
            wave_route_identity = _route_identity(wave_buffers, workload)
            wave_correctness = _correctness(reference, wave_output)
            peak = int(torch.cuda.max_memory_allocated())
            full_wall = float(full["wall_ms_stats"]["median_ms"])
            micro_wall = float(micro_serial["wall_ms_stats"]["median_ms"])
            wave_wall = float(wavefront["wall_ms_stats"]["median_ms"])
            full_expert = float(full["expert_ms_stats"]["median_ms"])
            micro_expert = float(micro_serial["expert_ms_stats"]["median_ms"])
            full_communication = float(full["dispatch_ms_stats"]["median_ms"]) + float(
                full["combine_ms_stats"]["median_ms"]
            )
            micro_communication = float(
                micro_serial["dispatch_ms_stats"]["median_ms"]
            ) + float(micro_serial["combine_ms_stats"]["median_ms"])
            result["o2"].append(
                {
                    "batch_equivalent": batch_equivalent,
                    "microbatches": microbatches,
                    "microbatch_tokens_global": workload.token_count // microbatches,
                    "workload": workload_metrics(
                        workload,
                        vision_tokens_per_request=int(
                            capture["metadata"]["vision_token_count"]
                        ),
                        ep_size=ep_size,
                        local_experts_per_rank=int(
                            capture["metadata"]["local_experts_per_rank"]
                        ),
                    ),
                    "full_batch_serial": full,
                    "microbatch_serial": micro_serial,
                    "expert_centered_wavefront": wavefront,
                    "speedup_vs_full_batch": full_wall / wave_wall,
                    "speedup_vs_microbatch_serial": micro_wall / wave_wall,
                    "throughput_tokens_per_second": workload.token_count * 1000.0 / wave_wall,
                    "throughput_assignments_per_second": workload.assignment_count
                    * 1000.0
                    / wave_wall,
                    "expert_fragmentation_penalty": micro_expert / full_expert - 1.0,
                    "collective_repetition_penalty": micro_communication
                    / full_communication
                    - 1.0,
                    "route_identity": bool(
                        full_route_identity
                        and serial_route_identity
                        and wave_route_identity
                    ),
                    "token_order_restoration": bool(
                        wave_output.shape == reference.shape
                        and microbatches
                        * (workload.token_count // ep_size // microbatches)
                        == reference.shape[0]
                    ),
                    "microbatch_serial_correctness": serial_correctness,
                    "wavefront_correctness": wave_correctness,
                    "full_batch_peak_memory_allocated_bytes": full_memory,
                    "memory_allocated_before_bytes": memory_before,
                    "peak_memory_allocated_bytes": peak,
                    "incremental_peak_memory_bytes": peak - memory_before,
                }
            )
            correctness_failed = correctness_failed or not (
                serial_correctness["passed"]
                and wave_correctness["passed"]
                and full_route_identity
                and serial_route_identity
                and wave_route_identity
            )
            del wave_output, wave_buffers
            torch.cuda.empty_cache()
            if correctness_failed:
                break
        del reference, workload
        torch.cuda.empty_cache()
        if correctness_failed:
            result["status"] = "correctness_failed"
            break

    result["attention_router_extension_executed"] = False
    result["attention_router_extension_reason"] = (
        "Core D/E/C is analyzed first; the enclosing analyzer applies the 1.10x stop rule"
    )
    return result


def _write_rank_result(rank: int, result: dict[str, Any]) -> None:
    directory = Path(os.environ["FLASHVEP_OFFLINE_RESULT_DIR"])
    path = directory / f"rank{rank}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _execute_once(
    kernel: Any,
    in_dtype: torch.dtype,
    a1q: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: Any,
    global_num_experts: int,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> None:
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    rank = int(ep.rank_in_group)
    if rank in _RAN_RANKS:
        return
    expected_tokens = _int_env("FLASHVEP_OFFLINE_ORIGINAL_TOKENS", 799)
    chunks = _dp_chunk_sizes()
    if (
        _current_layer() != _int_env("FLASHVEP_OFFLINE_LAYER", 24)
        or a1q.shape[0] < expected_tokens
        or chunks is None
        or sum(chunks) != a1q.shape[0]
        or chunks[:2] != [400, 400]
    ):
        return
    _RAN_RANKS.add(rank)
    spec = ExpertSpec(
        fused=kernel.fused_experts,
        in_dtype=in_dtype,
        w1=w1,
        w2=w2,
        activation=activation,
        global_num_experts=int(global_num_experts),
        local_num_experts=int(local_num_experts),
        expert_map=expert_map,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
    )
    try:
        capture = _capture(
            a1q, topk_ids, topk_weights, spec, int(ep.world_size)
        )
        capture_path = Path(os.environ["FLASHVEP_OFFLINE_CAPTURE_PATH"])
        if rank == 0:
            if capture_path.exists():
                raise FileExistsError(f"refusing to overwrite {capture_path}")
            torch.save(capture, capture_path)
        result = _run_benchmark(capture, spec)
        result["capture_metadata"] = capture["metadata"]
        result["capture_runtime"] = {
            "dp_chunk_sizes": chunks,
            "captured_a1q_shape": list(a1q.shape),
            "captured_topk_ids_shape": list(topk_ids.shape),
            "fused_experts_backend": type(kernel.fused_experts).__name__,
            "prepare_finalize_backend": type(kernel.prepare_finalize).__name__,
        }
        _write_rank_result(rank, result)
    except BaseException as exc:
        failure = {
            "status": "error",
            "rank": rank,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_rank_result(rank, failure)
        finally:
            raise


def install_offline_wavefront() -> None:
    """Install the opt-in layer context and expert-boundary capture hook."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_offline_layer = _layer_from_prefix(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(getattr(self, "_flashvep_offline_layer", -1))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def patched_experts(
        self: Any,
        in_dtype: torch.dtype,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: Any,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        expert_tokens_meta: Any,
    ) -> torch.Tensor:
        _execute_once(
            self,
            in_dtype,
            a1q,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            local_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )
        return original_experts(
            self,
            in_dtype,
            a1q,
            a1q_scale,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            local_num_experts,
            expert_map,
            apply_router_weight_on_input,
            expert_tokens_meta,
        )

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
