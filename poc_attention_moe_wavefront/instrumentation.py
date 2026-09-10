"""Low-perturbation, same-device CUDA timing for stock Qwen3-VL prefills."""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import torch


_LOCK = threading.Lock()
_CONTEXT = threading.local()
_CONTROL: dict[str, Any] = {}
_SIGNATURE: tuple[int, int, int] | None = None
_ROWS: list[dict[str, Any]] = []
_COUNTERS: Counter[str] = Counter()
_PROFILED_WAVES: set[int] = set()
_FLUSHED = False
_EP_RANK_CACHE: int | None = None


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _refresh_control() -> dict[str, Any]:
    global _CONTROL, _SIGNATURE
    path = Path(os.environ["WAVEFRONT_CONTROL"])
    if not path.exists():
        return {}
    stat = path.stat()
    signature = (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))
    if signature != _SIGNATURE:
        _CONTROL = json.loads(path.read_text(encoding="utf-8"))
        _SIGNATURE = signature
        _COUNTERS["control_reads"] += 1
    return _CONTROL


def _ep_rank() -> int:
    global _EP_RANK_CACHE
    if _EP_RANK_CACHE is not None:
        return _EP_RANK_CACHE
    from vllm.distributed import get_ep_group

    _EP_RANK_CACHE = int(get_ep_group().rank_in_group)
    return _EP_RANK_CACHE


def _event() -> torch.cuda.Event:
    _COUNTERS["events"] += 1
    return torch.cuda.Event(enable_timing=True)


def _token_count(args: tuple[Any, ...], kwargs: dict[str, Any], index: int) -> int:
    value = kwargs.get("hidden_states")
    if value is None and len(args) > index:
        value = args[index]
    return int(value.shape[0]) if isinstance(value, torch.Tensor) else -1


def _record(stage: str, start: torch.cuda.Event, end: torch.cuda.Event,
            tokens: int, layer: int, **extra: Any) -> None:
    entry = dict(getattr(_CONTEXT, "entry", {}))
    with _LOCK:
        _ROWS.append(
            {
                "run_id": os.environ["WAVEFRONT_RUN_ID"],
                "wave": int(entry["wave"]),
                "request_id": entry["request_id"],
                "workload_id": entry["workload_id"],
                "iteration": int(entry["iteration"]),
                "instrumented": bool(entry["instrumented"]),
                "ep_rank": _ep_rank(),
                "layer": int(layer),
                "stage": stage,
                "tokens": int(tokens),
                "start": start,
                "end": end,
                **extra,
            }
        )


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    if not _ROWS:
        return
    _FLUSHED = True
    if _ROWS:
        _ROWS[-1]["end"].synchronize()
    output = Path(os.environ["WAVEFRONT_RAW"])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(_ROWS[0]["ep_rank"])
    rows = []
    for item in _ROWS:
        row = {key: value for key, value in item.items() if key not in ("start", "end")}
        row["duration_ms"] = float(item["start"].elapsed_time(item["end"]))
        rows.append(row)
    payload = {
        "status": "ok",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_cuda_device": int(torch.cuda.current_device()),
        "ep_rank": rank,
        "rows": rows,
        "profiled_waves": sorted(_PROFILED_WAVES),
        "counters": dict(_COUNTERS),
        "timing_semantics": "same-device CUDA event duration; no cross-device timestamp subtraction",
    }
    (output / f"rank{rank}.json").write_text(json.dumps(payload, indent=2) + "\n")


def install() -> None:
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeAttention,
        Qwen3MoeDecoderLayer,
        Qwen3MoeSparseMoeBlock,
    )
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    from vllm.v1.worker import gpu_model_runner as gmr

    if getattr(Qwen3VLMoeForConditionalGeneration, "_attention_wavefront_phase0", False):
        return
    Qwen3VLMoeForConditionalGeneration._attention_wavefront_phase0 = True

    original_outer = Qwen3VLMoeForConditionalGeneration.forward
    original_execute = gmr.GPUModelRunner.execute_model
    original_attn_init = Qwen3MoeAttention.__init__
    original_attn = Qwen3MoeAttention.forward
    original_layer_init = Qwen3MoeDecoderLayer.__init__
    original_layer = Qwen3MoeDecoderLayer.forward
    original_moe = Qwen3MoeSparseMoeBlock.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    def patched_attn_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_attn_init(self, *args, **kwargs)
        self._wavefront_layer = _layer(str(kwargs.get("prefix", "")))

    def patched_layer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_layer_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._wavefront_layer = _layer(prefix)

    def active() -> bool:
        return bool(getattr(_CONTEXT, "active", False))

    def timed(stage: str, original: Any, token_index: int, layer_from_self: bool = False):
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not active():
                return original(self, *args, **kwargs)
            start, end = _event(), _event()
            start.record(torch.cuda.current_stream())
            result = original(self, *args, **kwargs)
            end.record(torch.cuda.current_stream())
            layer = int(getattr(self, "_wavefront_layer", -1)) if layer_from_self else int(getattr(_CONTEXT, "layer", -1))
            _record(stage, start, end, _token_count(args, kwargs, token_index), layer)
            return result
        return wrapper

    def patched_layer(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(self._wavefront_layer)
        if not active():
            try:
                return original_layer(self, *args, **kwargs)
            finally:
                _CONTEXT.layer = previous
        start, end = _event(), _event()
        start.record(torch.cuda.current_stream())
        try:
            result = original_layer(self, *args, **kwargs)
            end.record(torch.cuda.current_stream())
            _record("layer_total", start, end, _token_count(args, kwargs, 1), self._wavefront_layer)
            return result
        finally:
            _CONTEXT.layer = previous

    def patched_outer(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _refresh_control()
        wave = int(entry.get("wave", -1))
        profile = bool(entry.get("instrumented")) and wave not in _PROFILED_WAVES
        previous_entry = getattr(_CONTEXT, "entry", {})
        previous_active = getattr(_CONTEXT, "active", False)
        _CONTEXT.entry = entry
        _CONTEXT.active = profile
        if not profile:
            try:
                return original_outer(self, *args, **kwargs)
            finally:
                _CONTEXT.entry = previous_entry
                _CONTEXT.active = previous_active
        _PROFILED_WAVES.add(wave)
        start, end = _event(), _event()
        start.record(torch.cuda.current_stream())
        try:
            result = original_outer(self, *args, **kwargs)
            end.record(torch.cuda.current_stream())
            _record("decoder_total", start, end, _token_count(args, kwargs, 1), -1)
            return result
        finally:
            _CONTEXT.entry = previous_entry
            _CONTEXT.active = previous_active

    def patched_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_execute(self, *args, **kwargs)
        if _refresh_control().get("flush"):
            _flush()
        return result

    gmr.GPUModelRunner.execute_model = patched_execute
    Qwen3VLMoeForConditionalGeneration.forward = patched_outer
    Qwen3MoeAttention.__init__ = patched_attn_init
    Qwen3MoeAttention.forward = timed("attention_total", original_attn, 1, True)
    Qwen3MoeDecoderLayer.__init__ = patched_layer_init
    Qwen3MoeDecoderLayer.forward = patched_layer
    Qwen3MoeSparseMoeBlock.forward = timed("moe_total", original_moe, 0)
    FusedMoEKernelModularImpl._prepare = timed("dispatch", original_prepare, 0)
    FusedMoEKernelModularImpl._fused_experts = timed("expert", original_experts, 1)
    FusedMoEKernelModularImpl._finalize = timed("combine", original_finalize, 0)
    atexit.register(_flush)
