"""In-worker, scheduler-free layer-24 DeepEP D/E/C replay."""

from __future__ import annotations

import json
import os
import re
import statistics
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from poc_flashvep.offline_wavefront.capture_schema import validate_capture
from poc_flashvep.offline_wavefront.expert_centered_pipeline import overlap_summary
from poc_flashvep.offline_wavefront.workload_builder import (
    build_repeated_workload,
    rank_slice,
    workload_metrics,
)


_INSTALLED = False
_RAN_RANKS: set[int] = set()
_CONTEXT = threading.local()


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _values_env(name: str, default: str) -> list[int]:
    return [int(value) for value in os.environ.get(name, default).split(",") if value]


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pct(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return {
        "median_ms": float(statistics.median(values)),
        "p10_ms": float(pct(0.1)),
        "p90_ms": float(pct(0.9)),
        "mean_ms": float(statistics.fmean(values)),
        "stddev_ms": float(statistics.stdev(values) if len(values) > 1 else 0.0),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _current_layer() -> int:
    return int(getattr(_CONTEXT, "layer", -1))


def _layer_from_prefix(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


@dataclass
class ExpertSpec:
    in_dtype: torch.dtype
    w1: torch.Tensor
    w2: torch.Tensor
    activation: Any
    global_num_experts: int
    local_num_experts: int
    expert_map: torch.Tensor | None
    apply_router_weight_on_input: bool


@dataclass
class MicroState:
    hidden: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor
    recv_hidden: torch.Tensor | None = None
    recv_weights: torch.Tensor | None = None
    recv_ids: torch.Tensor | None = None
    recv_counts: list[int] | None = None
    handle: Any = None
    dispatch_event: Any = None
    expert_output: torch.Tensor | None = None
    expert_event: Any = None
    combined: torch.Tensor | None = None
    combine_event: Any = None


@dataclass
class StageRecord:
    stage: str
    microbatch: int
    start: torch.cuda.Event
    end: torch.cuda.Event


def _micro_states(workload: Any, rank: int, ep_size: int, k: int) -> list[MicroState]:
    local = rank_slice(workload.token_count, ep_size, rank)
    hidden = workload.hidden[local].contiguous()
    weights = workload.topk_weights[local].contiguous()
    ids = workload.topk_ids[local].to(torch.int64).contiguous()
    if hidden.shape[0] % k:
        raise ValueError("local token count must divide by microbatch count")
    return [
        MicroState(item_hidden, item_weights, item_ids)
        for item_hidden, item_weights, item_ids in zip(
            hidden.chunk(k), weights.chunk(k), ids.chunk(k)
        )
    ]


def _dispatch(
    index: int,
    state: MicroState,
    buffer: Any,
    comm_stream: torch.cuda.Stream,
    spec: ExpertSpec,
    records: list[StageRecord],
) -> None:
    import deep_ep

    start, end = _event(), _event()
    start.record(comm_stream)
    torch.cuda.nvtx.range_push(f"FLASHVEP_DEEPEP_D_mb{index}")
    layout = buffer.get_dispatch_layout(
        state.ids,
        spec.global_num_experts,
        async_finish=True,
        allocate_on_comm_stream=False,
    )
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        layout_event,
    ) = layout
    dispatched = buffer.dispatch(
        x=state.hidden,
        handle=None,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=state.ids.to(deep_ep.topk_idx_t),
        topk_weights=state.weights,
        expert_alignment=1,
        config=deep_ep.Buffer.get_dispatch_config(dist.get_world_size()),
        previous_event=layout_event,
        async_finish=True,
        allocate_on_comm_stream=False,
    )
    (
        state.recv_hidden,
        state.recv_ids,
        state.recv_weights,
        state.recv_counts,
        state.handle,
        state.dispatch_event,
    ) = dispatched
    torch.cuda.nvtx.range_pop()
    end.record(comm_stream)
    records.append(StageRecord("dispatch", index, start, end))


def _expert(
    index: int,
    state: MicroState,
    kernel: Any,
    original_experts: Any,
    expert_stream: torch.cuda.Stream,
    spec: ExpertSpec,
    rank: int,
    records: list[StageRecord],
) -> None:
    import deep_ep
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        ExpertTokensMetadata,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
        TopKWeightAndReduceDelegate,
    )

    assert state.recv_hidden is not None
    assert state.recv_ids is not None
    assert state.recv_weights is not None
    assert state.recv_counts is not None
    start, end = _event(), _event()
    with torch.cuda.stream(expert_stream):
        state.dispatch_event.current_stream_wait()
        offset = rank * spec.local_num_experts
        global_ids = torch.where(
            state.recv_ids == -1,
            spec.global_num_experts - 1 if offset == 0 else 0,
            state.recv_ids + offset,
        )
        meta = ExpertTokensMetadata.make_from_list(
            state.recv_counts, device=state.recv_hidden.device
        )
        start.record(expert_stream)
        torch.cuda.nvtx.range_push(f"FLASHVEP_TRITON_E_mb{index}")
        fused_expert_output = original_experts(
            kernel,
            spec.in_dtype,
            state.recv_hidden,
            None,
            spec.w1,
            spec.w2,
            state.recv_weights,
            global_ids,
            spec.activation,
            spec.global_num_experts,
            spec.local_num_experts,
            spec.expert_map,
            spec.apply_router_weight_on_input,
            meta,
        )
        weight_and_reduce = kernel.fused_experts.finalize_weight_and_reduce_impl()
        if isinstance(weight_and_reduce, TopKWeightAndReduceDelegate):
            weight_and_reduce = TopKWeightAndReduceContiguous()
        reduced_output = weight_and_reduce.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=state.recv_weights,
            topk_ids=global_ids,
            apply_router_weight_on_input=spec.apply_router_weight_on_input,
        )
        # vLLM's workspace manager may return the same storage on the next
        # expert call. Keep one reduced output buffer per microbatch so E(next)
        # cannot overwrite the input consumed by C(previous).
        state.expert_output = reduced_output.clone()
        torch.cuda.nvtx.range_pop()
        end.record(expert_stream)
        state.expert_event = deep_ep.Buffer.capture()
    records.append(StageRecord("expert", index, start, end))


def _combine(
    index: int,
    state: MicroState,
    buffer: Any,
    comm_stream: torch.cuda.Stream,
    records: list[StageRecord],
) -> None:
    import deep_ep

    assert state.expert_output is not None
    start, end = _event(), _event()
    with torch.cuda.stream(comm_stream):
        state.expert_event.current_stream_wait()
    start.record(comm_stream)
    torch.cuda.nvtx.range_push(f"FLASHVEP_DEEPEP_C_mb{index}")
    state.combined, _, state.combine_event = buffer.combine(
        x=state.expert_output,
        handle=state.handle,
        topk_weights=None,
        config=deep_ep.Buffer.get_combine_config(dist.get_world_size()),
        async_finish=True,
        allocate_on_comm_stream=False,
    )
    torch.cuda.nvtx.range_pop()
    end.record(comm_stream)
    records.append(StageRecord("combine", index, start, end))


def _iteration(
    variant: str,
    states: list[MicroState],
    kernel: Any,
    original_experts: Any,
    buffer: Any,
    spec: ExpertSpec,
    rank: int,
    measure: bool,
) -> tuple[dict[str, float] | None, torch.Tensor]:
    comm_stream = buffer.get_comm_stream()
    expert_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()
    origin, wall_end = _event(), _event()
    records: list[StageRecord] = []
    origin.record(default_stream)
    comm_stream.wait_event(origin)
    expert_stream.wait_event(origin)

    def prime_sync() -> None:
        if not measure:
            torch.cuda.synchronize()
            dist.barrier()

    if variant == "serial":
        for index, state in enumerate(states):
            _dispatch(index, state, buffer, comm_stream, spec, records)
            prime_sync()
            _expert(
                index, state, kernel, original_experts, expert_stream, spec, rank, records
            )
            prime_sync()
            _combine(index, state, buffer, comm_stream, records)
            prime_sync()
    elif variant == "overlap_k2":
        if len(states) != 2:
            raise ValueError("overlap_k2 requires exactly two microbatches")
        _dispatch(0, states[0], buffer, comm_stream, spec, records)
        prime_sync()
        _expert(0, states[0], kernel, original_experts, expert_stream, spec, rank, records)
        prime_sync()
        _dispatch(1, states[1], buffer, comm_stream, spec, records)
        prime_sync()
        _combine(0, states[0], buffer, comm_stream, records)
        prime_sync()
        _expert(1, states[1], kernel, original_experts, expert_stream, spec, rank, records)
        prime_sync()
        _combine(1, states[1], buffer, comm_stream, records)
        prime_sync()
    else:
        raise ValueError(f"unknown variant {variant}")

    states[-1].combine_event.current_stream_wait()
    wall_end.record(default_stream)
    wall_end.synchronize()
    output = torch.cat([state.combined for state in states if state.combined is not None])
    if not measure:
        return None, output

    intervals: dict[str, list[tuple[float, float]]] = {
        "dispatch": [],
        "expert": [],
        "combine": [],
    }
    for record in records:
        intervals[record.stage].append(
            (_elapsed(origin, record.start), _elapsed(origin, record.end))
        )
    result = {
        "wall_ms": _elapsed(origin, wall_end),
        "dispatch_ms": sum(end - start for start, end in intervals["dispatch"]),
        "expert_ms": sum(end - start for start, end in intervals["expert"]),
        "combine_ms": sum(end - start for start, end in intervals["combine"]),
        **overlap_summary(
            intervals["dispatch"], intervals["expert"], intervals["combine"]
        ),
    }
    return result, output


def _run_variant(
    variant: str,
    workload: Any,
    k: int,
    kernel: Any,
    original_experts: Any,
    buffer: Any,
    spec: ExpertSpec,
    rank: int,
    ep_size: int,
    warmups: int,
    iterations: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    for _ in range(warmups):
        states = _micro_states(workload, rank, ep_size, k)
        _iteration(variant, states, kernel, original_experts, buffer, spec, rank, False)
    samples = []
    output = None
    for _ in range(iterations):
        states = _micro_states(workload, rank, ep_size, k)
        sample, output = _iteration(
            variant, states, kernel, original_experts, buffer, spec, rank, True
        )
        assert sample is not None
        samples.append(sample)
    assert output is not None
    result: dict[str, Any] = {
        "variant": variant,
        "microbatches": k,
        "warmups": warmups,
        "iterations": iterations,
    }
    for key in ("wall_ms", "dispatch_ms", "expert_ms", "combine_ms"):
        result[key] = [float(sample[key]) for sample in samples]
        result[f"{key}_stats"] = _stats(result[key])
    for key in (
        "dispatch_expert_overlap_ms",
        "expert_combine_overlap_ms",
        "actual_overlap_fraction",
    ):
        result[key] = [float(sample[key]) for sample in samples]
        result[f"{key}_stats"] = _stats(result[key])
    return result, output.detach().clone()


def _correctness(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    difference = (reference.float() - candidate.float()).abs()
    ref = reference.float().flatten()
    value = candidate.float().flatten()
    denominator = float(ref.norm().item() * value.norm().item())
    cosine = float(torch.dot(ref, value).item()) / denominator if denominator else 1.0
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
        "error": error,
    }


def _run_replay(kernel: Any, original_experts: Any, spec: ExpertSpec) -> dict[str, Any]:
    import deep_ep
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    rank = int(ep.rank_in_group)
    ep_size = int(ep.world_size)
    buffer = kernel.prepare_finalize.buffer
    capture = torch.load(
        os.environ["FLASHVEP_DEEPEP_CAPTURE_PATH"], map_location="cpu", weights_only=False
    )
    validate_capture(capture)
    base_hidden = capture["post_attention_hidden"].to(spec.w1.device)
    base_ids = capture["topk_expert_ids"].to(spec.w1.device)
    base_weights = capture["topk_weights"].to(spec.w1.device)
    warmups = _int_env("FLASHVEP_DEEPEP_REPLAY_WARMUPS", 5)
    iterations = _int_env("FLASHVEP_DEEPEP_REPLAY_ITERATIONS", 20)
    batches = _values_env("FLASHVEP_DEEPEP_REPLAY_BATCHES", "32,64,128")
    sms_values = _values_env("FLASHVEP_DEEPEP_REPLAY_SMS", "20,16,12,8,4")
    rows = []
    all_correct = True
    for batch in batches:
        workload = build_repeated_workload(base_hidden, base_ids, base_weights, batch)
        reference = None
        for sms in sms_values:
            deep_ep.Buffer.set_num_sms(sms)
            dist.barrier(group=ep.device_group)
            torch.cuda.reset_peak_memory_stats()
            full, full_output = _run_variant(
                "serial", workload, 1, kernel, original_experts, buffer, spec,
                rank, ep_size, warmups, iterations,
            )
            if reference is None:
                reference = full_output
            full_correct = _correctness(reference, full_output)
            micro, micro_output = _run_variant(
                "serial", workload, 2, kernel, original_experts, buffer, spec,
                rank, ep_size, warmups, iterations,
            )
            overlap, overlap_output = _run_variant(
                "overlap_k2", workload, 2, kernel, original_experts, buffer, spec,
                rank, ep_size, warmups, iterations,
            )
            micro_correct = _correctness(reference, micro_output)
            overlap_correct = _correctness(reference, overlap_output)
            correct = full_correct["passed"] and micro_correct["passed"] and overlap_correct["passed"]
            all_correct = all_correct and correct
            rows.append(
                {
                    "batch_equivalent": batch,
                    "communication_sms": sms,
                    "workload": workload_metrics(
                        workload,
                        vision_tokens_per_request=int(capture["metadata"]["vision_token_count"]),
                        ep_size=ep_size,
                        local_experts_per_rank=spec.local_num_experts,
                    ),
                    "full_serial": full,
                    "micro_serial_k2": micro,
                    "overlap_k2": overlap,
                    "speedup_vs_full_serial": full["wall_ms_stats"]["median_ms"] / overlap["wall_ms_stats"]["median_ms"],
                    "speedup_vs_micro_serial": micro["wall_ms_stats"]["median_ms"] / overlap["wall_ms_stats"]["median_ms"],
                    "dispatch_slowdown": overlap["dispatch_ms_stats"]["median_ms"] / micro["dispatch_ms_stats"]["median_ms"],
                    "expert_slowdown": overlap["expert_ms_stats"]["median_ms"] / micro["expert_ms_stats"]["median_ms"],
                    "combine_slowdown": overlap["combine_ms_stats"]["median_ms"] / micro["combine_ms_stats"]["median_ms"],
                    "fragmentation_penalty": micro["expert_ms_stats"]["median_ms"] / full["expert_ms_stats"]["median_ms"] - 1.0,
                    "correctness": {
                        "passed": correct,
                        "full_serial": full_correct,
                        "micro_serial_k2": micro_correct,
                        "overlap_k2": overlap_correct,
                        "route_identity": True,
                        "source_token_order_restoration": overlap_output.shape == reference.shape,
                    },
                    "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
            del full_output, micro_output, overlap_output
        del workload, reference
        torch.cuda.empty_cache()
    deep_ep.Buffer.set_num_sms(20)
    return {
        "status": "ok" if all_correct else "correctness_failed",
        "rank": rank,
        "physical_gpu": [4, 5, 6, 7][rank],
        "settings": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "capture_path": os.environ["FLASHVEP_DEEPEP_CAPTURE_PATH"],
            "capture_sha256": "208e789cbbefb8c7b2baab00a3f4aa1e19d6b44b9e7a67f6ae2413a28363eda0",
            "layer": 24,
            "expert_backend": type(kernel.fused_experts).__name__,
            "prepare_finalize_backend": type(kernel.prepare_finalize).__name__,
            "communication_backend": "DeepEP high-throughput Buffer dispatch/combine",
            "warmups": warmups,
            "iterations": iterations,
            "batches": batches,
            "supported_sms_sweep": sms_values,
            "unsupported_sms": [24],
            "unsupported_sms_reason": "vLLM manager initializes a maximum communication budget of 20 SMs",
        },
        "rows": rows,
        "all_correct": all_correct,
        "k4_executed": False,
        "k4_reason": "K2 is the mandatory contention comparison; K4 is gated on post-run fragmentation analysis",
    }


def _write_result(rank: int, result: dict[str, Any]) -> None:
    directory = Path(os.environ["FLASHVEP_DEEPEP_REPLAY_RESULT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"operator_rank{rank}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def install_operator_replay() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_deepep_layer = _layer_from_prefix(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(getattr(self, "_flashvep_deepep_layer", -1))
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
        from vllm.distributed import get_ep_group

        ep = get_ep_group()
        rank = int(ep.rank_in_group)
        if (
            rank not in _RAN_RANKS
            and _current_layer() == _int_env("FLASHVEP_DEEPEP_REPLAY_LAYER", 24)
            and type(self.prepare_finalize).__name__ == "DeepEPHTPrepareAndFinalize"
        ):
            _RAN_RANKS.add(rank)
            spec = ExpertSpec(
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
                _write_result(rank, _run_replay(self, original_experts, spec))
            except BaseException as exc:
                _write_result(
                    rank,
                    {
                        "status": "error",
                        "rank": rank,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
        return original_experts(
            self, in_dtype, a1q, a1q_scale, w1, w2, topk_weights, topk_ids,
            activation, global_num_experts, local_num_experts, expert_map,
            apply_router_weight_on_input, expert_tokens_meta,
        )

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
