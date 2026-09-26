"""Bounded full-model CUDA-event attribution layered on the proven EP hook.

This module does not alter tensors or routing.  It wraps the vision encoder,
decoder layer, attention and complete sparse-MoE block with CUDA events.  The
existing DeepEP hook supplies dispatch/expert/combine events.  Results are
resolved once, at an explicit final flush, to keep host synchronization out of
the timed requests.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch


_INSTALLED = False
_CONTEXT = threading.local()
_PENDING: list[dict[str, Any]] = []
_FLUSHED = False
_LAST_WAVE = -1


def _layer(prefix: str) -> int:
    """Resolve the decoder layer without relying on the EP profiler.

    The full-model profiler is intentionally usable on its own.  Previously
    ``_flashvep_matrix_layer`` was only populated by the optional EP profiler,
    so ``PROFILE_FULL`` installed in the real workers but silently recorded no
    decoder events (every layer looked like ``-1``).
    """
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_MATRIX_CONTROL"])
    value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    by_dp = value.get("by_dp")
    if by_dp:
        value = {**value, **by_dp.get(str(os.environ.get("VLLM_DP_RANK", "0")), {})}
    return value


def _timed(kind: str, layer: int, function: Any, *args: Any, **kwargs: Any) -> Any:
    entry = dict(getattr(_CONTEXT, "entry", {}))
    if not entry.get("instrument"):
        return function(*args, **kwargs)
    start, end = _event(), _event()
    start.record(torch.cuda.current_stream())
    value = function(*args, **kwargs)
    end.record(torch.cuda.current_stream())
    _PENDING.append({
        "kind": kind,
        "layer": layer,
        "wave": int(entry["wave"]),
        "request_id": entry["request_id"],
        "iteration": int(entry["iteration"]),
        "measured": bool(entry["measured"]),
        "worker_dp_rank": int(os.environ["VLLM_DP_RANK"]),
        "start": start,
        "end": end,
    })
    return value


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    output = Path(os.environ["FLASHVEP_MATRIX_RAW_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"full_flush_pid{os.getpid()}.json").write_text(json.dumps({
        "pid": os.getpid(),
        "pending_events": len(_PENDING),
        "control": _control(),
    }, indent=2) + "\n", encoding="utf-8")
    if not _PENDING:
        return
    torch.cuda.synchronize()
    from vllm.distributed import get_ep_group

    rank = int(get_ep_group().rank_in_group)
    with (output / f"full_rank{rank}.jsonl").open("w", encoding="utf-8") as handle:
        for item in _PENDING:
            row = {key: value for key, value in item.items() if key not in ("start", "end")}
            row["ms"] = float(item["start"].elapsed_time(item["end"]))
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / f"full_rank{rank}.proof.json").write_text(json.dumps({
        "status": "ok",
        "ep_rank": rank,
        "events": len(_PENDING),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolution": "one final bounded cuda synchronize",
        "semantics": "read-only CUDA events; no tensor or routing mutation",
    }, indent=2) + "\n", encoding="utf-8")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    output = Path(os.environ["FLASHVEP_MATRIX_RAW_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"full_install_pid{os.getpid()}.json").write_text(json.dumps({
        "pid": os.getpid(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "profile_mode": {
            "both": os.environ.get("MLLM_EP_FOUNDATION_PROFILE"),
            "full": os.environ.get("MLLM_EP_FOUNDATION_PROFILE_FULL"),
            "ep": os.environ.get("MLLM_EP_FOUNDATION_PROFILE_EP"),
        },
    }, indent=2) + "\n", encoding="utf-8")

    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeAttention,
        Qwen3MoeDecoderLayer,
        Qwen3MoeSparseMoeBlock,
    )
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    decoder_forward = Qwen3MoeDecoderLayer.forward
    decoder_init = Qwen3MoeDecoderLayer.__init__
    attention_forward = Qwen3MoeAttention.forward
    moe_forward = Qwen3MoeSparseMoeBlock.forward
    vision_forward = Qwen3_VisionTransformer.forward

    def patched_decoder_init(self: Any, *args: Any, **kwargs: Any) -> None:
        decoder_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_matrix_layer = _layer(prefix)

    def patched_decoder(self: Any, *args: Any, **kwargs: Any) -> Any:
        layer = int(getattr(self, "_flashvep_matrix_layer", -1))
        global _LAST_WAVE
        if layer == 0:
            entry = _control()
            if entry.get("flush"):
                _flush()
            wave = int(entry.get("wave", -1))
            # DP padding can invoke the model repeatedly while the peer rank
            # is still busy.  Attribute only the first model invocation in a
            # wave, matching the proven DeepEP hook's boundary.
            _CONTEXT.active_entry = (
                entry if wave != _LAST_WAVE else {**entry, "instrument": False}
            )
            _LAST_WAVE = wave
        entry = dict(getattr(_CONTEXT, "active_entry", {}))
        prior_entry = getattr(_CONTEXT, "entry", {})
        prior_layer = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.entry = entry
        _CONTEXT.layer = layer
        try:
            return _timed("decoder_layer", layer, decoder_forward, self, *args, **kwargs)
        finally:
            _CONTEXT.entry = prior_entry
            _CONTEXT.layer = prior_layer
            if layer == 47:
                _CONTEXT.active_entry = {}

    def patched_attention(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed("attention", int(getattr(_CONTEXT, "layer", -1)),
                      attention_forward, self, *args, **kwargs)

    def patched_moe(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _timed("moe_total", int(getattr(_CONTEXT, "layer", -1)),
                      moe_forward, self, *args, **kwargs)

    def patched_vision(self: Any, *args: Any, **kwargs: Any) -> Any:
        prior_entry = getattr(_CONTEXT, "entry", {})
        _CONTEXT.entry = _control()
        try:
            return _timed("vision_encoder", -1, vision_forward, self, *args, **kwargs)
        finally:
            _CONTEXT.entry = prior_entry

    Qwen3MoeDecoderLayer.__init__ = patched_decoder_init
    Qwen3MoeDecoderLayer.forward = patched_decoder
    Qwen3MoeAttention.forward = patched_attention
    Qwen3MoeSparseMoeBlock.forward = patched_moe
    Qwen3_VisionTransformer.forward = patched_vision
    # Multi-wave external DP generation can deadlock in the installed vLLM
    # runtime after the first wave.  Resolving the bounded CUDA-event ledger
    # at normal worker exit preserves the one-sync contract without issuing a
    # second dummy model wave solely to trigger a flush.
    atexit.register(_flush)
