"""Opt-in CUDA-event and NVTX instrumentation for Qwen3-MoE layers.

The profiler is installed only when ``FLASHVEP_PROFILE_JSONL`` is set. CUDA
events are synchronized at worker shutdown, outside the measured request
window. The patch records the existing execution path and does not change
routing, precision, token order, expert placement, or collectives.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import re
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
_CONTEXT = threading.local()
_LOCK = threading.Lock()
_ORIGIN_EVENT: torch.cuda.Event | None = None


def _trace_path() -> Path | None:
    raw = os.environ.get("FLASHVEP_PROFILE_JSONL")
    return Path(raw) if raw else None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _layer_id(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)", prefix)
    return int(match.group(1)) if match else -1


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "-1"))


def _physical_gpu(logical_rank: int) -> int | None:
    raw = os.environ.get("FLASHVEP_PHYSICAL_GPUS", "")
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if 0 <= logical_rank < len(values):
        return int(values[logical_rank])
    return None


def _current_context() -> tuple[int, int] | None:
    layer = getattr(_CONTEXT, "layer", None)
    call_index = getattr(_CONTEXT, "call_index", None)
    if layer is None or call_index is None:
        return None
    return int(layer), int(call_index)


def _should_record(call_index: int) -> bool:
    skip = _int_env("FLASHVEP_SKIP_LAYER_CALLS", 8)
    count = _int_env("FLASHVEP_MEASURE_LAYER_CALLS", 20)
    return skip <= call_index < skip + count


def _stage_enabled(stage: str) -> bool:
    raw = os.environ.get("FLASHVEP_PROFILE_STAGES", "")
    if not raw:
        return True
    return stage in {value.strip() for value in raw.split(",") if value.strip()}


def _first_cuda_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            return value
    return None


def _timed(
    stage: str,
    function: Callable[..., Any],
    *args: Any,
    postprocess: Callable[[Any], dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    context = _current_context()
    if context is None or not _should_record(context[1]) or not _stage_enabled(stage):
        return function(*args, **kwargs)

    tensor = _first_cuda_tensor(args, kwargs)
    if tensor is None:
        return function(*args, **kwargs)

    layer, call_index = context
    rank = _rank()
    iteration = call_index - _int_env("FLASHVEP_SKIP_LAYER_CALLS", 8)
    global _ORIGIN_EVENT
    if _ORIGIN_EVENT is None:
        with _LOCK:
            if _ORIGIN_EVENT is None:
                _ORIGIN_EVENT = torch.cuda.Event(enable_timing=True)
                _ORIGIN_EVENT.record()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    range_name = f"flashvep/{stage}/layer_{layer}/iter_{iteration}/rank_{rank}"
    cpu_start_ns = time.perf_counter_ns()
    start.record()
    torch.cuda.nvtx.range_push(range_name)
    try:
        output = function(*args, **kwargs)
    finally:
        torch.cuda.nvtx.range_pop()
        end.record()
        cpu_end_ns = time.perf_counter_ns()
    extra = postprocess(output) if postprocess is not None else {}

    if stage == "decoder_layer":
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 2:
            hidden_states = args[2]
        hidden_tokens = (
            int(hidden_states.shape[0])
            if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim > 0
            else 0
        )
    else:
        hidden_tokens = int(tensor.shape[0]) if tensor.ndim > 0 else 0
    metadata = _LAYER_META.get(layer, {})
    record: dict[str, Any] = {
        "run_id": os.environ.get("FLASHVEP_RUN_ID", "unknown"),
        "request_id": f"measure_{iteration}",
        "iteration_id": iteration,
        "pid": os.getpid(),
        "rank": rank,
        "physical_gpu": _physical_gpu(rank),
        "layer": layer,
        "stage": stage,
        "cpu_enqueue_start_ns": cpu_start_ns,
        "cpu_enqueue_end_ns": cpu_end_ns,
        "hidden_tokens": hidden_tokens,
        "tp_size": int(metadata.get("tp_size", -1)),
        "ep_size": int(metadata.get("ep_size", -1)),
        "local_expert_start": metadata.get("local_expert_start"),
        "local_expert_end": metadata.get("local_expert_end"),
        "gpu_name": torch.cuda.get_device_name(tensor.device),
        "_start_event": start,
        "_end_event": end,
        "_origin_event": _ORIGIN_EVENT,
    }
    record.update(extra)
    with _LOCK:
        _PENDING.append(record)
    return output


def _wrap_module(module: Any, stage: str) -> None:
    if getattr(module, "_flashvep_profile_wrapped", False):
        return
    original = module.forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _timed(stage, original, *args, **kwargs)

    module.forward = wrapped
    module._flashvep_profile_wrapped = True


def _route_postprocess(output: Any) -> dict[str, Any]:
    topk_ids = output[1]
    return {
        "routed_token_assignments": int(topk_ids.numel()),
        "_topk_ids": topk_ids,
    }


def _patch_decoder() -> None:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        layer = _layer_id(prefix)
        self._flashvep_layer_id = layer

        _wrap_module(self.self_attn.qkv_proj, "qkv_projection")
        _wrap_module(self.self_attn.attn, "attention_core")
        _wrap_module(self.self_attn.o_proj, "attention_output_projection")
        _wrap_module(self.self_attn, "attention_block")
        _wrap_module(self.input_layernorm, "input_residual_rmsnorm")
        _wrap_module(
            self.post_attention_layernorm,
            "post_attention_residual_rmsnorm",
        )

        if hasattr(self.mlp, "experts"):
            _wrap_module(self.mlp.gate, "router_projection")
            _LAYER_META[layer] = {
                "tp_size": int(self.mlp.tp_size),
                "ep_size": int(self.mlp.ep_size),
                "local_expert_start": int(self.mlp.physical_expert_start),
                "local_expert_end": int(self.mlp.physical_expert_end),
            }

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        layer = int(getattr(self, "_flashvep_layer_id", -1))
        call_index = _LAYER_CALLS[layer]
        _LAYER_CALLS[layer] += 1
        previous_layer = getattr(_CONTEXT, "layer", None)
        previous_call = getattr(_CONTEXT, "call_index", None)
        _CONTEXT.layer = layer
        _CONTEXT.call_index = call_index
        try:
            return _timed("decoder_layer", original_forward, self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous_layer
            _CONTEXT.call_index = previous_call

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward


def _patch_moe_stages() -> None:
    from vllm.model_executor.layers.fused_moe import layer as fused_layer
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    original_moe = fused_layer.FusedMoE.forward
    original_router = BaseRouter.select_experts
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize
    original_reduce = MoERunner._maybe_reduce_final_output

    def moe_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed("moe_layer", original_moe, self, *args, **kwargs)

    def router_select(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed(
            "router_topk",
            original_router,
            self,
            *args,
            postprocess=_route_postprocess,
            **kwargs,
        )

    def prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed(
            "dispatch_prepare_no_collective", original_prepare, self, *args, **kwargs
        )

    def experts(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed("local_expert_execution", original_experts, self, *args, **kwargs)

    def finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed("local_finalize", original_finalize, self, *args, **kwargs)

    def reduce(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed("combine_tp_allreduce", original_reduce, self, *args, **kwargs)

    fused_layer.FusedMoE.forward = moe_forward
    BaseRouter.select_experts = router_select
    FusedMoEKernelModularImpl._prepare = prepare
    FusedMoEKernelModularImpl._fused_experts = experts
    FusedMoEKernelModularImpl._finalize = finalize
    MoERunner._maybe_reduce_final_output = reduce


def _json_record(record: dict[str, Any]) -> dict[str, Any]:
    start = record.pop("_start_event")
    end = record.pop("_end_event")
    origin = record.pop("_origin_event")
    end.synchronize()
    record["duration_ms"] = float(start.elapsed_time(end))
    record["gpu_start_ms_from_rank_origin"] = float(origin.elapsed_time(start))
    record["gpu_end_ms_from_rank_origin"] = float(origin.elapsed_time(end))
    topk_ids = record.pop("_topk_ids", None)
    if topk_ids is not None:
        global_counts = [0] * 128
        for expert_id in topk_ids.reshape(-1).cpu().tolist():
            global_counts[int(expert_id)] += 1
        start_expert = record.get("local_expert_start")
        end_expert = record.get("local_expert_end")
        if start_expert is not None and end_expert is not None:
            local_counts = global_counts[int(start_expert) : int(end_expert)]
        else:
            local_counts = []
        record["global_expert_token_counts"] = global_counts
        record["local_expert_token_counts"] = local_counts
        record["max_local_expert_batch"] = max(local_counts, default=0)
    return record


def flush_records() -> None:
    """Synchronize pending events once and append records atomically."""

    path = _trace_path()
    if path is None:
        return
    with _LOCK:
        pending = list(_PENDING)
        _PENDING.clear()
    if not pending:
        return

    records: list[dict[str, Any]] = []
    for pending_record in pending:
        try:
            records.append(_json_record(pending_record))
        except Exception as exc:
            pending_record.pop("_start_event", None)
            pending_record.pop("_end_event", None)
            pending_record.pop("_topk_ids", None)
            pending_record.pop("_origin_event", None)
            pending_record["duration_ms"] = None
            pending_record["error"] = repr(exc)
            records.append(pending_record)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(payload)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def install() -> bool:
    """Install profiler patches when an output path is configured."""

    global _PATCHED
    if _PATCHED or _trace_path() is None:
        return _PATCHED
    _patch_decoder()
    _patch_moe_stages()
    atexit.register(flush_records)
    _PATCHED = True
    return True
