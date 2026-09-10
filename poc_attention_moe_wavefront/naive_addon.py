"""Stage timing layered over the existing validated naive DBO wavefront hook."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import torch


_LOCK = threading.Lock()
_CONTEXT = threading.local()
_ROWS: list[dict[str, Any]] = []
_ORIGINS: dict[int, torch.cuda.Event] = {}
_FLUSHED = False
_INSTALLED = False


def control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_LIVE_WAVEFRONT_CONTROL"])
    return json.loads(path.read_text()) if path.exists() else {}


def tokens(args: tuple[Any, ...], kwargs: dict[str, Any], index: int) -> int:
    value = kwargs.get("hidden_states")
    if value is None and len(args) > index:
        value = args[index]
    return int(value.shape[0]) if isinstance(value, torch.Tensor) else -1


def rank() -> int:
    from vllm.distributed import get_ep_group
    return int(get_ep_group().rank_in_group)


def flush() -> None:
    global _FLUSHED
    if _FLUSHED or not _ROWS:
        return
    _FLUSHED = True
    torch.cuda.synchronize()
    output = Path(os.environ["FLASHVEP_LIVE_WAVEFRONT_RAW"]) / "addon"
    output.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in _ROWS:
        origin = _ORIGINS[int(row["wave"])]
        output_rows.append({key: value for key, value in row.items() if key not in ("start", "end")} | {
            "start_ms": float(origin.elapsed_time(row["start"])),
            "end_ms": float(origin.elapsed_time(row["end"])),
            "duration_ms": float(row["start"].elapsed_time(row["end"])),
        })
    ep_rank = rank()
    (output / f"rank{ep_rank}.json").write_text(json.dumps({
        "status": "ok", "ep_rank": ep_rank,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mode": os.environ["FLASHVEP_LIVE_WAVEFRONT_MODE"],
        "rows": output_rows,
        "timing_semantics": "same-device CUDA event duration and relative event timeline",
    }, indent=2) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention, Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_enabled

    if getattr(Qwen3VLMoeForConditionalGeneration, "_wavefront_naive_addon", False):
        return
    Qwen3VLMoeForConditionalGeneration._wavefront_naive_addon = True
    original_outer = Qwen3VLMoeForConditionalGeneration.forward
    original_layer = Qwen3MoeDecoderLayer.forward
    original_attn = Qwen3MoeAttention.forward
    original_moe = Qwen3MoeSparseMoeBlock.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    def active() -> bool:
        entry = getattr(_CONTEXT, "entry", {})
        return (bool(entry.get("timeline")) and entry.get("phase") == "measured"
                and bool(getattr(_CONTEXT, "target_prefill", False)))

    def is_target(entry: dict[str, Any], count: int) -> bool:
        if entry.get("phase") == "flush":
            return False
        if os.environ["FLASHVEP_LIVE_WAVEFRONT_MODE"] == "wavefront" and dbo_enabled():
            ubatch = int(dbo_current_ubatch_id())
            expected = entry.get("prefix_tokens") if ubatch == 0 else entry.get("tail_tokens")
        else:
            expected = entry.get("prompt_tokens")
        return expected is not None and count == int(expected)

    def append(stage: str, start: torch.cuda.Event, end: torch.cuda.Event, count: int) -> None:
        entry = dict(_CONTEXT.entry)
        with _LOCK:
            _ROWS.append({
                "wave": int(entry["wave"]), "request_id": entry["request_id"],
                "iteration": int(entry["iteration"]), "layer": int(getattr(_CONTEXT, "layer", -1)),
                "ubatch_id": int(dbo_current_ubatch_id()) if dbo_enabled() else -1,
                "stage": stage, "tokens": count, "start": start, "end": end,
            })

    def timed(stage: str, original: Any, index: int):
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not active():
                return original(self, *args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream())
            result = original(self, *args, **kwargs)
            end.record(torch.cuda.current_stream())
            append(stage, start, end, tokens(args, kwargs, index))
            return result
        return wrapper

    def patched_layer(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(getattr(self, "_flashvep_wavefront_layer", -1))
        if not active():
            try: return original_layer(self, *args, **kwargs)
            finally: _CONTEXT.layer = previous
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        try:
            result = original_layer(self, *args, **kwargs)
            end.record(torch.cuda.current_stream())
            append("layer_total", start, end, tokens(args, kwargs, 1))
            return result
        finally:
            _CONTEXT.layer = previous

    def patched_outer(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = control()
        if entry.get("phase") == "flush":
            result = original_outer(self, *args, **kwargs)
            flush()
            return result
        previous = getattr(_CONTEXT, "entry", {})
        previous_target = getattr(_CONTEXT, "target_prefill", False)
        _CONTEXT.entry = entry
        _CONTEXT.target_prefill = is_target(entry, tokens(args, kwargs, 1))
        try:
            if active() and (not dbo_enabled() or int(dbo_current_ubatch_id()) in (-1, 0)):
                origin = torch.cuda.Event(enable_timing=True)
                origin.record(torch.cuda.current_stream())
                with _LOCK: _ORIGINS.setdefault(int(entry["wave"]), origin)
            return original_outer(self, *args, **kwargs)
        finally:
            _CONTEXT.entry = previous
            _CONTEXT.target_prefill = previous_target

    Qwen3VLMoeForConditionalGeneration.forward = patched_outer
    Qwen3MoeDecoderLayer.forward = patched_layer
    Qwen3MoeAttention.forward = timed("attention_total", original_attn, 1)
    Qwen3MoeSparseMoeBlock.forward = timed("moe_total", original_moe, 0)
    FusedMoEKernelModularImpl._prepare = timed("dispatch", original_prepare, 0)
    FusedMoEKernelModularImpl._fused_experts = timed("expert", original_experts, 1)
    FusedMoEKernelModularImpl._finalize = timed("combine", original_finalize, 0)
