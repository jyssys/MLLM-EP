"""Runtime CUDA-event profiler for vLLM ``FusedMoE`` layers.

This patch is deliberately measurement-only: it wraps ``FusedMoE.forward`` with
CUDA events, records per-layer/per-rank elapsed time, and writes JSONL records.
It does not alter routing, expert placement, fused kernels, precision, or
all-to-all behavior.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from vllm_custom_placement import layer_id_from_prefix


_PATCHED_ATTR = "_mllm_moe_cuda_timing_patched"
_PENDING: list[dict[str, Any]] = []
_CALL_INDEX: dict[tuple[int, int], int] = defaultdict(int)
_GLOBAL_CALL_INDEX = 0


def _jsonl_path() -> str | None:
    return os.environ.get("VLLM_MOE_TIMING_JSONL")


def _flush_every() -> int:
    raw = os.environ.get("VLLM_MOE_TIMING_FLUSH_EVERY", "48")
    try:
        return max(1, int(raw))
    except ValueError:
        return 48


def _append_jsonl(records: list[dict[str, Any]]) -> None:
    path_raw = _jsonl_path()
    if not path_raw or not records:
        return

    path = Path(path_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(lines)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            handle.write(lines)


def flush_moe_timing_records() -> None:
    """Synchronize pending CUDA events and append measured records to JSONL."""

    if not _PENDING:
        return

    pending = list(_PENDING)
    _PENDING.clear()
    records: list[dict[str, Any]] = []
    for record in pending:
        start = record.pop("_start_event")
        end = record.pop("_end_event")
        try:
            end.synchronize()
            record["elapsed_ms"] = float(start.elapsed_time(end))
            records.append(record)
        except Exception as exc:
            record["elapsed_ms"] = None
            record["error"] = repr(exc)
            records.append(record)
    _append_jsonl(records)


def apply_vllm_moe_timing_patch() -> bool:
    """Patch vLLM ``FusedMoE.forward`` when ``VLLM_MOE_TIMING_JSONL`` is set."""

    if not _jsonl_path():
        return False

    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    fused_moe_cls = fused_layer.FusedMoE
    if getattr(fused_moe_cls, _PATCHED_ATTR, False):
        return True

    original_forward = fused_moe_cls.forward

    def patched_forward(
        self: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global _GLOBAL_CALL_INDEX

        if not torch.cuda.is_available() or hidden_states.device.type != "cuda":
            return original_forward(self, hidden_states, router_logits, input_ids)

        layer_id = layer_id_from_prefix(getattr(self, "layer_name", ""))
        if layer_id is None:
            layer_id = -1
        ep_rank = int(getattr(self, "ep_rank", -1))
        key = (int(layer_id), ep_rank)
        call_index = _CALL_INDEX[key]
        _CALL_INDEX[key] += 1
        global_call_index = _GLOBAL_CALL_INDEX
        _GLOBAL_CALL_INDEX += 1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = original_forward(self, hidden_states, router_logits, input_ids)
        end.record()

        record = {
            "pid": os.getpid(),
            "time_s": time.time(),
            "layer": int(layer_id),
            "ep_rank": ep_rank,
            "ep_size": int(getattr(self, "ep_size", -1)),
            "local_num_experts": int(getattr(self, "local_num_experts", -1)),
            "global_num_experts": int(getattr(self, "global_num_experts", -1)),
            "call_index": int(call_index),
            "global_call_index": int(global_call_index),
            "hidden_tokens": int(hidden_states.shape[0])
            if hidden_states.ndim >= 1
            else 0,
            "hidden_shape": list(hidden_states.shape),
            "router_logits_shape": list(router_logits.shape),
            "_start_event": start,
            "_end_event": end,
        }
        _PENDING.append(record)
        if len(_PENDING) >= _flush_every():
            flush_moe_timing_records()
        return output

    fused_moe_cls.forward = patched_forward
    setattr(fused_moe_cls, _PATCHED_ATTR, True)
    atexit.register(flush_moe_timing_records)
    return True
