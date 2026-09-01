"""Add DP-local pre-MoE spans to the validated read-only MoE hook.

The existing hook owns dispatch/expert/combine timing and flushes CUDA events at
the explicit request boundary. This wrapper records a separate layer-entry to
MoE-entry span and EP-entry to finalize span without changing tensors or routes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from poc_flashvep.ep4_serving_straggler_regime import live_instrumentation as base

_INSTALLED = False
_ARRIVAL: list[dict[str, Any]] = []
_BY_RECORD: dict[int, dict[str, Any]] = {}
_FLUSHED = False


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _write_topology() -> None:
    from .topology_probe import write_once
    write_once()


def _resolve(ev_a: torch.cuda.Event, ev_b: torch.cuda.Event) -> float:
    try:
        return float(ev_a.elapsed_time(ev_b))
    except RuntimeError:
        return float("nan")


def _flush_arrival() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    if not _ARRIVAL:
        return
    torch.cuda.synchronize()
    from vllm.distributed import get_ep_group, get_tp_group
    dp = int(os.environ.get("VLLM_DP_RANK", -1))
    tp = int(get_tp_group().rank_in_group)
    ep = int(get_ep_group().rank_in_group)
    out = Path(os.environ["FLASHVEP_MATRIX_RAW_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"arrival_dp{dp}_tp{tp}_ep{ep}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in _ARRIVAL:
            row = {k: v for k, v in item.items() if k not in ("layer_start", "moe_entry", "moe_done")}
            row["attention_cuda_ms"] = _resolve(item["layer_start"], item["moe_entry"])
            row["pre_moe_cuda_ms"] = row["attention_cuda_ms"]
            row["ep_entry_to_done_ms"] = _resolve(item["moe_entry"], item["moe_done"])
            row["layer_entry_to_ep_done_ms"] = _resolve(item["layer_start"], item["moe_done"])
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
    (out / f"arrival_dp{dp}_tp{tp}_ep{ep}.proof.json").write_text(
        json.dumps({"events": len(_ARRIVAL), "cuda_sync": "single explicit flush", "cross_gpu_absolute_subtraction": False}, indent=2) + "\n",
        encoding="utf-8",
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    base.install()
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_forward = Qwen3MoeDecoderLayer.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_finalize = FusedMoEKernelModularImpl._finalize

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        layer = int(getattr(self, "_flashvep_serving_layer", -1))
        ctx = base._CONTEXT
        if layer == 0 and base._control().get("flush"):
            _flush_arrival()
        # Base's patched forward installs the request control on layer 0.  We
        # read that same immutable record once here so layer-0 timing is not
        # lost; later layers reuse the thread-local cache.
        entry = base._control() if layer == 0 else getattr(ctx, "entry", {})
        enabled = bool(entry.get("instrument")) and layer >= 0
        if enabled:
            start = _event()
            start.record(torch.cuda.current_stream())
            ctx.arrival_layer_start = start
            ctx.arrival_layer = layer
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            if enabled and layer == 47:
                ctx.arrival_layer_start = None

    def patched_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = base._CONTEXT
        entry = getattr(ctx, "entry", {})
        layer = int(getattr(ctx, "layer", -1))
        enabled = bool(entry.get("instrument")) and layer >= 0
        if not enabled:
            return original_prepare(self, *args, **kwargs)
        _write_topology()
        entry_event = getattr(ctx, "arrival_layer_start", None)
        moe_entry = _event()
        moe_entry.record(torch.cuda.current_stream())
        value = original_prepare(self, *args, **kwargs)
        record = getattr(ctx, "record", None)
        if record is not None and entry_event is not None:
            item = {
                "wave": int(entry.get("wave", -1)), "batch_id": entry.get("batch_id"),
                "condition": entry.get("condition"), "modality": entry.get("modality"),
                "concurrency": int(entry.get("concurrency", 1)), "phase": entry.get("phase"),
                "iteration": int(entry.get("iteration", -1)), "measured": bool(entry.get("measured", False)),
                "scheduler_iteration": int(getattr(ctx, "forward_index", 0)),
                "worker_dp_rank": int(os.environ.get("VLLM_DP_RANK", -1)), "layer": layer,
                "layer_start": entry_event, "moe_entry": moe_entry, "moe_done": None,
            }
            _ARRIVAL.append(item)
            _BY_RECORD[id(record)] = item
        return value

    def patched_finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        record = getattr(base._CONTEXT, "record", None)
        value = original_finalize(self, *args, **kwargs)
        if record is not None:
            item = _BY_RECORD.pop(id(record), None)
            if item is not None:
                done = _event()
                done.record(torch.cuda.current_stream())
                item["moe_done"] = done
        return value

    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._prepare = patched_prepare
    FusedMoEKernelModularImpl._finalize = patched_finalize
