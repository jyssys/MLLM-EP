"""Opt-in TP2/DP2 DPEP audit and selected-layer CUDA-event profiling.

This module observes the installed vLLM 0.20.0 execution path. It is inert
unless a Phase 1b output environment variable is set and never changes model
weights, routing, precision, token order, or collective selection.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import re
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


_PATCHED = False
_PENDING: list[dict[str, Any]] = []
_LAYER_CALLS: dict[int, int] = defaultdict(int)
_LAYER_META: dict[int, dict[str, Any]] = {}
_RUNTIME_RECORDED: set[tuple[int, int]] = set()
_MICROBENCH_RECORDED: set[int] = set()
_CONTEXT = threading.local()
_LOCK = threading.Lock()
_ORIGIN_EVENT: torch.cuda.Event | None = None


def _audit_path() -> Path | None:
    raw = os.environ.get("FLASHVEP_PHASE1B_AUDIT_JSONL")
    return Path(raw) if raw else None


def _profile_path() -> Path | None:
    raw = os.environ.get("FLASHVEP_PHASE1B_PROFILE_JSONL")
    return Path(raw) if raw else None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _selected_layers() -> set[int]:
    raw = os.environ.get("FLASHVEP_PHASE1B_LAYERS", "0,12,24,36,47")
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def _selected_call_offsets() -> set[int]:
    raw = os.environ.get("FLASHVEP_PHASE1B_LAYER_CALL_OFFSETS", "0")
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def _layer_id(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)", prefix)
    return int(match.group(1)) if match else -1


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "-1"))


def _physical_gpu(rank: int) -> int | None:
    raw = os.environ.get("FLASHVEP_PHYSICAL_GPUS", "4,5,6,7")
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return int(values[rank]) if 0 <= rank < len(values) else None


def _current_context() -> tuple[int, int] | None:
    layer = getattr(_CONTEXT, "layer", None)
    call_index = getattr(_CONTEXT, "call_index", None)
    if layer is None or call_index is None:
        return None
    return int(layer), int(call_index)


def _should_record(layer: int, call_index: int) -> bool:
    skip = _int_env("FLASHVEP_PHASE1B_SKIP_LAYER_CALLS", 8)
    count = _int_env("FLASHVEP_PHASE1B_MEASURE_LAYER_CALLS", 20)
    stride = max(1, _int_env("FLASHVEP_PHASE1B_LAYER_CALL_STRIDE", 1))
    offset = call_index - skip
    return (
        _profile_path() is not None
        and layer in _selected_layers()
        and offset >= 0
        and offset % stride in _selected_call_offsets()
        and offset // stride < count
    )


def _append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(payload)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parallel_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        from vllm.distributed import (
            get_dp_group,
            get_ep_group,
            get_pp_group,
            get_tp_group,
        )

        tp_group = get_tp_group()
        dp_group = get_dp_group()
        ep_group = get_ep_group()
        pp_group = get_pp_group()
        result.update(
            {
                "tp_size": len(tp_group.ranks),
                "dp_size": len(dp_group.ranks),
                "pp_size": len(pp_group.ranks),
                "configured_ep_size": len(ep_group.ranks),
                "tp_rank": int(tp_group.rank_in_group),
                "dp_rank": int(dp_group.rank_in_group),
                "ep_rank": int(ep_group.rank_in_group),
                "tp_group_ranks": [int(value) for value in tp_group.ranks],
                "dp_group_ranks": [int(value) for value in dp_group.ranks],
                "ep_group_ranks": [int(value) for value in ep_group.ranks],
                "all2all_backend": "allgather_reducescatter",
            }
        )
    except Exception as exc:
        result["parallel_metadata_error"] = repr(exc)
    return result


def _base_record(kind: str) -> dict[str, Any]:
    rank = _rank()
    context = _current_context()
    layer, call_index = context if context is not None else (-1, -1)
    return {
        "kind": kind,
        "run_id": os.environ.get("FLASHVEP_PHASE1B_RUN_ID", "unknown"),
        "pid": os.getpid(),
        "rank": rank,
        "physical_gpu": _physical_gpu(rank),
        "layer": layer,
        "call_index": call_index,
        "time_ns": time.time_ns(),
        **_parallel_metadata(),
    }


def _tensor_shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def _tensor_dtype(value: Any) -> str | None:
    return str(value.dtype) if isinstance(value, torch.Tensor) else None


def _dp_chunk_sizes() -> list[int] | None:
    try:
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().dp_metadata
        if metadata is None:
            return None
        sizes = metadata.get_chunk_sizes_across_dp_rank()
        return None if sizes is None else [int(value) for value in sizes]
    except Exception:
        return None


def _first_cuda_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            return value
    return None


def _timed(
    stage: str,
    function: Callable[..., Any],
    *args: Any,
    extra: dict[str, Any] | None = None,
    postprocess: Callable[[Any], dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    context = _current_context()
    if context is None or not _should_record(*context):
        return function(*args, **kwargs)
    tensor = _first_cuda_tensor(args, kwargs)
    if tensor is None:
        return function(*args, **kwargs)

    layer, call_index = context
    skip = _int_env("FLASHVEP_PHASE1B_SKIP_LAYER_CALLS", 8)
    stride = max(1, _int_env("FLASHVEP_PHASE1B_LAYER_CALL_STRIDE", 1))
    iteration_id = (call_index - skip) // stride
    wave_call_offset = (call_index - skip) % stride
    rank = _rank()
    global _ORIGIN_EVENT
    if _ORIGIN_EVENT is None:
        with _LOCK:
            if _ORIGIN_EVENT is None:
                _ORIGIN_EVENT = torch.cuda.Event(enable_timing=True)
                _ORIGIN_EVENT.record()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    range_name = (
        f"flashvep_phase1b/{stage}/layer_{layer}/"
        f"iter_{iteration_id}/rank_{rank}"
    )
    cpu_start_ns = time.perf_counter_ns()
    start.record()
    torch.cuda.nvtx.range_push(range_name)
    try:
        output = function(*args, **kwargs)
    finally:
        torch.cuda.nvtx.range_pop()
        end.record()
        cpu_end_ns = time.perf_counter_ns()

    record = {
        "kind": "stage",
        "run_id": os.environ.get("FLASHVEP_PHASE1B_RUN_ID", "unknown"),
        "iteration_id": iteration_id,
        "wave_call_offset": wave_call_offset,
        "pid": os.getpid(),
        "rank": rank,
        "physical_gpu": _physical_gpu(rank),
        "layer": layer,
        "call_index": call_index,
        "stage": stage,
        "cpu_enqueue_start_ns": cpu_start_ns,
        "cpu_enqueue_end_ns": cpu_end_ns,
        "input_shape": list(tensor.shape),
        "input_dtype": str(tensor.dtype),
        "gpu_name": torch.cuda.get_device_name(tensor.device),
        "_start_event": start,
        "_end_event": end,
        "_origin_event": _ORIGIN_EVENT,
        **_LAYER_META.get(layer, {}),
    }
    if extra:
        record.update(extra)
    if postprocess is not None:
        record.update(postprocess(output))
    with _LOCK:
        _PENDING.append(record)
    return output


def _wrap_module(module: Any, stage: str) -> None:
    if getattr(module, "_flashvep_phase1b_wrapped", False):
        return
    original = module.forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _timed(stage, original, *args, **kwargs)

    module.forward = wrapped
    module._flashvep_phase1b_wrapped = True


def _patch_decoder() -> None:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        layer = _layer_id(prefix)
        self._flashvep_phase1b_layer_id = layer
        if layer not in _selected_layers():
            return
        _wrap_module(self.self_attn, "attention_block")
        _wrap_module(self.input_layernorm, "input_residual_rmsnorm")
        _wrap_module(
            self.post_attention_layernorm,
            "post_attention_residual_rmsnorm",
        )
        _wrap_module(self.mlp.gate, "router_projection")
        _LAYER_META[layer] = {
            "tp_size": int(self.mlp.tp_size),
            "ep_size": int(self.mlp.ep_size),
            "local_expert_start": int(self.mlp.physical_expert_start),
            "local_expert_end": int(self.mlp.physical_expert_end),
            "local_expert_count": int(self.mlp.n_local_physical_experts),
        }

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        layer = int(getattr(self, "_flashvep_phase1b_layer_id", -1))
        call_index = _LAYER_CALLS[layer]
        _LAYER_CALLS[layer] += 1
        previous = _current_context()
        _CONTEXT.layer = layer
        _CONTEXT.call_index = call_index
        try:
            if layer in _selected_layers():
                return _timed("decoder_layer", original_forward, self, *args, **kwargs)
            return original_forward(self, *args, **kwargs)
        finally:
            if previous is None:
                _CONTEXT.layer = None
                _CONTEXT.call_index = None
            else:
                _CONTEXT.layer, _CONTEXT.call_index = previous

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward


def _runtime_classes(moe: Any) -> dict[str, Any]:
    quant_method = getattr(moe, "quant_method", None)
    kernel = getattr(quant_method, "moe_kernel", None)
    prepare_finalize = getattr(kernel, "prepare_finalize", None)
    fused_experts = getattr(kernel, "fused_experts", None)
    return {
        "fused_moe_class": type(moe).__name__,
        "quant_method_class": type(quant_method).__name__,
        "moe_kernel_class": type(kernel).__name__,
        "prepare_finalize_class": type(prepare_finalize).__name__,
        "fused_experts_class": type(fused_experts).__name__,
        "moe_backend": os.environ.get("FLASHVEP_PHASE1B_MOE_BACKEND", "unknown"),
        "is_sequence_parallel": bool(
            getattr(getattr(moe, "moe_parallel_config", None), "is_sequence_parallel", False)
        ),
    }


def _patch_moe_and_router() -> None:
    from vllm.model_executor.layers.fused_moe import layer as fused_layer
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    original_moe = fused_layer.FusedMoE.forward
    original_router = BaseRouter.select_experts

    def moe_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = _current_context()
        if context is not None and context[0] in _selected_layers():
            key = (_rank(), context[0])
            if key not in _RUNTIME_RECORDED:
                _RUNTIME_RECORDED.add(key)
                record = _base_record("runtime_path")
                record.update(_runtime_classes(self))
                record.update(_LAYER_META.get(context[0], {}))
                _append_jsonl(_audit_path(), record)
        return _timed("moe_layer", original_moe, self, *args, **kwargs)

    def router_select(self: Any, *args: Any, **kwargs: Any) -> Any:
        def details(output: Any) -> dict[str, Any]:
            topk_ids = output[1]
            return {
                "topk_ids_shape": list(topk_ids.shape),
                "topk_ids_dtype": str(topk_ids.dtype),
                "routed_assignments": int(topk_ids.numel()),
            }

        return _timed(
            "router_topk", original_router, self, *args, postprocess=details, **kwargs
        )

    fused_layer.FusedMoE.forward = moe_forward
    BaseRouter.select_experts = router_select


def _audit_collective(
    collective: str,
    manager: Any,
    args: tuple[Any, ...],
    output: Any,
) -> None:
    context = _current_context()
    if context is None or context[0] not in _selected_layers():
        return
    record = _base_record("collective_call")
    record.update(
        {
            "collective": collective,
            "manager_class": type(manager).__name__,
            "dp_chunk_sizes": _dp_chunk_sizes(),
            "input_shapes": [
                list(value.shape) for value in args if isinstance(value, torch.Tensor)
            ],
            "input_dtypes": [
                str(value.dtype) for value in args if isinstance(value, torch.Tensor)
            ],
        }
    )
    if isinstance(output, torch.Tensor):
        record["output_shapes"] = [list(output.shape)]
    elif isinstance(output, tuple):
        record["output_shapes"] = [
            list(value.shape) for value in output if isinstance(value, torch.Tensor)
        ]
    _append_jsonl(_audit_path(), record)


def _patch_agrs_collectives() -> None:
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

    original_dispatch = AgRsAll2AllManager.dispatch
    original_dispatch_logits = AgRsAll2AllManager.dispatch_router_logits
    original_combine = AgRsAll2AllManager.combine

    def dispatch(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = _timed(
            "dispatch_dpep_agrs",
            original_dispatch,
            self,
            *args,
            extra={"dp_chunk_sizes": _dp_chunk_sizes()},
            postprocess=lambda value: {
                "output_shapes": [
                    list(item.shape)
                    for item in value
                    if isinstance(item, torch.Tensor)
                ]
            },
            **kwargs,
        )
        _audit_collective("dispatch_all_gatherv", self, args, output)
        return output

    def dispatch_logits(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = _timed(
            "dispatch_router_logits_dpep_agrs",
            original_dispatch_logits,
            self,
            *args,
            extra={"dp_chunk_sizes": _dp_chunk_sizes()},
            **kwargs,
        )
        _audit_collective("dispatch_router_logits_all_gatherv", self, args, output)
        return output

    def combine(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = _timed(
            "combine_dpep_agrs",
            original_combine,
            self,
            *args,
            extra={"dp_chunk_sizes": _dp_chunk_sizes()},
            postprocess=lambda value: {"output_shape": _tensor_shape(value)},
            **kwargs,
        )
        _audit_collective("combine_reduce_scatterv", self, args, output)
        return output

    AgRsAll2AllManager.dispatch = dispatch
    AgRsAll2AllManager.dispatch_router_logits = dispatch_logits
    AgRsAll2AllManager.combine = combine


def _local_workload(
    topk_ids: torch.Tensor,
    local_start: int,
    local_end: int,
    chunk_sizes: list[int] | None,
) -> dict[str, Any]:
    ids = topk_ids.detach().reshape(topk_ids.shape[0], -1).cpu()
    local_mask = (ids >= local_start) & (ids < local_end)
    local_ids = ids[local_mask] - local_start
    counts = torch.bincount(local_ids, minlength=local_end - local_start).tolist()
    result: dict[str, Any] = {
        "actual_local_assignments": int(local_mask.sum()),
        "local_expert_token_counts": [int(value) for value in counts],
        "max_local_expert_batch": int(max(counts, default=0)),
        "active_local_experts": sum(int(value) > 0 for value in counts),
    }
    if chunk_sizes is not None and sum(chunk_sizes) == ids.shape[0]:
        offset = 0
        by_chunk: list[int] = []
        for size in chunk_sizes:
            by_chunk.append(int(local_mask[offset : offset + size].sum()))
            offset += size
        result["local_assignments_by_dp_chunk"] = by_chunk
        global_batch_size = _int_env("FLASHVEP_PHASE1B_GLOBAL_BATCH_SIZE", 1)
        if global_batch_size > 1:
            real_dp_tp_chunks = _int_env("FLASHVEP_PHASE1B_REAL_DP_TP_CHUNKS", 2)
            prompt_tokens = _int_env("FLASHVEP_PHASE1B_REAL_PROMPT_TOKENS", 799)
            local_real_assignments = 0
            local_padding_assignments = 0
            local_idle_dummy_assignments = 0
            real_requests_by_dp_rank: list[int] = []
            token_offset = 0
            for chunk_offset in range(0, len(chunk_sizes), real_dp_tp_chunks):
                dp_tokens = sum(
                    chunk_sizes[chunk_offset : chunk_offset + real_dp_tp_chunks]
                )
                real_requests = dp_tokens // prompt_tokens
                real_tokens = real_requests * prompt_tokens
                dp_mask = local_mask[token_offset : token_offset + dp_tokens]
                local_real_assignments += int(dp_mask[:real_tokens].sum())
                if real_requests:
                    local_padding_assignments += int(dp_mask[real_tokens:].sum())
                else:
                    local_idle_dummy_assignments += int(dp_mask.sum())
                real_requests_by_dp_rank.append(real_requests)
                token_offset += dp_tokens
            result.update(
                {
                    "global_batch_size": global_batch_size,
                    "observed_real_requests_by_dp_rank": real_requests_by_dp_rank,
                    "real_request_local_assignments": local_real_assignments,
                    "tp_padding_local_assignments": local_padding_assignments,
                    "idle_dp_dummy_local_assignments": local_idle_dummy_assignments,
                    "local_assignments_by_dp_rank": [
                        sum(by_chunk[offset : offset + real_dp_tp_chunks])
                        for offset in range(0, len(by_chunk), real_dp_tp_chunks)
                    ],
                }
            )
            return result
        vision_start = _int_env("FLASHVEP_PHASE1B_VISION_START", 4)
        vision_end = _int_env("FLASHVEP_PHASE1B_VISION_END", 788)
        real_prompt_tokens = _int_env("FLASHVEP_PHASE1B_REAL_PROMPT_TOKENS", 799)
        real_dp_tp_chunks = _int_env("FLASHVEP_PHASE1B_REAL_DP_TP_CHUNKS", 2)
        real_padded_tokens = sum(chunk_sizes[:real_dp_tp_chunks])
        if real_padded_tokens >= real_prompt_tokens >= vision_end:
            result["real_request_visual_local_assignments"] = int(
                local_mask[vision_start:vision_end].sum()
            )
            result["real_request_text_special_local_assignments"] = int(
                local_mask[:vision_start].sum()
                + local_mask[vision_end:real_prompt_tokens].sum()
            )
            result["tp_padding_local_assignments"] = int(
                local_mask[real_prompt_tokens:real_padded_tokens].sum()
            )
            result["idle_dp_dummy_local_assignments"] = int(
                local_mask[real_padded_tokens:].sum()
            )
    return result


def _patch_sequence_parallel_combine() -> None:
    from vllm.model_executor.models import qwen3_moe

    original_all_gather = qwen3_moe.tensor_model_parallel_all_gather

    def all_gather(*args: Any, **kwargs: Any) -> Any:
        return _timed(
            "combine_tp_allgather_after_dpep",
            original_all_gather,
            *args,
            **kwargs,
        )

    qwen3_moe.tensor_model_parallel_all_gather = all_gather


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "median_ms": float(statistics.median(values)),
        "p10_ms": float(percentile(0.1)),
        "p90_ms": float(percentile(0.9)),
        "mean_ms": float(statistics.fmean(values)),
        "stddev_ms": float(statistics.stdev(values) if len(values) > 1 else 0.0),
    }


def _patch_expert_and_tp_combine() -> None:
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_reduce = MoERunner._maybe_reduce_final_output

    def experts(
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
        context = _current_context()
        layer = context[0] if context is not None else -1
        call_index = context[1] if context is not None else -1
        chunk_sizes = _dp_chunk_sizes()
        meta = _LAYER_META.get(layer, {})
        local_start = int(meta.get("local_expert_start", 0))
        local_end = int(meta.get("local_expert_end", local_num_experts))
        detailed = (
            context is not None
            and _should_record(layer, call_index)
            and (call_index - _int_env("FLASHVEP_PHASE1B_SKIP_LAYER_CALLS", 8))
            // max(1, _int_env("FLASHVEP_PHASE1B_LAYER_CALL_STRIDE", 1))
            == 0
        )
        extra: dict[str, Any] = {
            "a1q_shape": list(a1q.shape),
            "a1q_dtype": str(a1q.dtype),
            "w1_shape": list(w1.shape),
            "w1_dtype": str(w1.dtype),
            "w2_shape": list(w2.shape),
            "w2_dtype": str(w2.dtype),
            "topk_ids_shape": list(topk_ids.shape),
            "topk_ids_dtype": str(topk_ids.dtype),
            "dp_chunk_sizes": chunk_sizes,
            "fused_experts_backend_class": type(self.fused_experts).__name__,
        }
        if detailed:
            extra["_workload_topk_ids"] = topk_ids.detach().clone()

        call_args = (
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
        output = _timed(
            "local_expert_execution",
            original_experts,
            *call_args,
            extra=extra,
        )

        skip = _int_env("FLASHVEP_PHASE1B_SKIP_LAYER_CALLS", 8)
        measured = _int_env("FLASHVEP_PHASE1B_MEASURE_LAYER_CALLS", 20)
        stride = max(1, _int_env("FLASHVEP_PHASE1B_LAYER_CALL_STRIDE", 1))
        capture_layer = _int_env("FLASHVEP_PHASE1B_CAPTURE_LAYER", 24)
        do_microbench = os.environ.get("FLASHVEP_PHASE1B_MICROBENCH") == "1"
        rank = _rank()
        if (
            do_microbench
            and layer == capture_layer
            and call_index == skip + measured * stride
            and rank not in _MICROBENCH_RECORDED
        ):
            _MICROBENCH_RECORDED.add(rank)
            warmups = _int_env("FLASHVEP_PHASE1B_MICROBENCH_WARMUPS", 20)
            iterations = _int_env("FLASHVEP_PHASE1B_MICROBENCH_ITERATIONS", 100)
            for _ in range(warmups):
                original_experts(*call_args)
            torch.cuda.synchronize(a1q.device)
            starts: list[torch.cuda.Event] = []
            ends: list[torch.cuda.Event] = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                original_experts(*call_args)
                end.record()
                starts.append(start)
                ends.append(end)
            ends[-1].synchronize()
            durations = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
            workload = _local_workload(topk_ids, local_start, local_end, chunk_sizes)
            hidden_size = int(a1q.shape[-1])
            intermediate_size = int(w2.shape[-1])
            flops = (
                int(workload["actual_local_assignments"])
                * 6
                * hidden_size
                * intermediate_size
            )
            result = _base_record("expert_microbenchmark")
            result.update(
                {
                    "warmups": warmups,
                    "iterations": iterations,
                    "hidden_size": hidden_size,
                    "expert_intermediate_size": intermediate_size,
                    "estimated_local_expert_flops": flops,
                    "backend_class": type(self.fused_experts).__name__,
                    "input_shape": list(a1q.shape),
                    "input_dtype": str(a1q.dtype),
                    "output_shape": list(output.shape),
                    "output_dtype": str(output.dtype),
                    **workload,
                    **_stats(durations),
                }
            )
            result["achieved_tflops_at_median"] = (
                flops / (result["median_ms"] / 1000.0) / 1e12
            )
            _append_jsonl(_audit_path(), result)
        return output

    def reduce(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed(
            "combine_tp_allreduce_after_dpep",
            original_reduce,
            self,
            *args,
            **kwargs,
        )

    FusedMoEKernelModularImpl._fused_experts = experts
    MoERunner._maybe_reduce_final_output = reduce


def _json_record(record: dict[str, Any]) -> dict[str, Any]:
    start = record.pop("_start_event")
    end = record.pop("_end_event")
    origin = record.pop("_origin_event")
    end.synchronize()
    record["duration_ms"] = float(start.elapsed_time(end))
    record["gpu_start_ms_from_rank_origin"] = float(origin.elapsed_time(start))
    record["gpu_end_ms_from_rank_origin"] = float(origin.elapsed_time(end))
    topk_ids = record.pop("_workload_topk_ids", None)
    if topk_ids is not None:
        record.update(
            _local_workload(
                topk_ids,
                int(record["local_expert_start"]),
                int(record["local_expert_end"]),
                record.get("dp_chunk_sizes"),
            )
        )
    return record


def flush_phase1b_records() -> None:
    """Synchronize pending events once at shutdown and append JSONL records."""

    path = _profile_path()
    if path is None:
        return
    with _LOCK:
        pending = list(_PENDING)
        _PENDING.clear()
    for pending_record in pending:
        try:
            record = _json_record(pending_record)
        except Exception as exc:
            pending_record.pop("_start_event", None)
            pending_record.pop("_end_event", None)
            pending_record.pop("_origin_event", None)
            pending_record.pop("_workload_topk_ids", None)
            pending_record["duration_ms"] = None
            pending_record["error"] = repr(exc)
            record = pending_record
        _append_jsonl(path, record)


def install_phase1b() -> bool:
    """Install Phase 1b observers only when explicitly configured."""

    global _PATCHED
    if _PATCHED or (_audit_path() is None and _profile_path() is None):
        return _PATCHED
    _patch_decoder()
    _patch_moe_and_router()
    _patch_agrs_collectives()
    _patch_expert_and_tp_combine()
    _patch_sequence_parallel_combine()
    atexit.register(flush_phase1b_records)
    _PATCHED = True
    return True
