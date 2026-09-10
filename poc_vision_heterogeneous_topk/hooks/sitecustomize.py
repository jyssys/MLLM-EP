"""Nonblocking per-layer attention timer for the bottleneck screen.

The worker writes only completed same-device CUDA event pairs.  It never
synchronizes the device and never reads model tensors, so the output is an
attention attribution trace rather than a route/quality observer.
"""

from __future__ import annotations

import functools
import json
import os
import re
import threading
from pathlib import Path


_ROOT = os.environ.get("VISION_TOPK_ATTN_TRACE_DIR")
if _ROOT:
    import torch

    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeAttention,
        Qwen3MoeDecoderLayer,
    )

    _LOCK = threading.Lock()
    _TLS = threading.local()
    _PENDING: list[dict] = []
    _LAYERS: dict[int, int] = {}

    def _rank() -> dict:
        out = {"pid": os.getpid(), "tp_rank": -1, "dp_rank": -1, "ep_rank": -1}
        try:
            from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
            out.update(tp_rank=int(get_tp_group().rank_in_group),
                       dp_rank=int(get_dp_group().rank_in_group),
                       ep_rank=int(get_ep_group().rank_in_group))
        except Exception:
            pass
        return out

    def _drain() -> None:
        ready, waiting = [], []
        for row in _PENDING:
            try:
                done = bool(row["end"].query())
            except Exception:
                done = False
            (ready if done else waiting).append(row)
        _PENDING[:] = waiting
        if not ready:
            return
        root = Path(_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"attention_pid{os.getpid()}.jsonl"
        with _LOCK, path.open("a", encoding="utf-8") as handle:
            for row in ready:
                out = {k: v for k, v in row.items() if k not in {"start", "end"}}
                out["attention_cuda_ms"] = float(row["start"].elapsed_time(row["end"]))
                handle.write(json.dumps(out, separators=(",", ":")) + "\n")

    original_init = Qwen3MoeDecoderLayer.__init__
    original_layer = Qwen3MoeDecoderLayer.forward
    original_attention = Qwen3MoeAttention.forward

    @functools.wraps(original_init)
    def layer_init(self, vllm_config, prefix=""):
        original_init(self, vllm_config, prefix)
        match = re.search(r"(?:layers|h)\.(\d+)", str(prefix))
        _LAYERS[id(self)] = int(match.group(1)) if match else -1

    @functools.wraps(original_layer)
    def layer_forward(self, *args, **kwargs):
        _drain()
        old = getattr(_TLS, "layer", -1)
        _TLS.layer = _LAYERS.get(id(self), -1)
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            _TLS.layer = old

    @functools.wraps(original_attention)
    def attention_forward(self, *args, **kwargs):
        hidden = args[1] if len(args) > 1 else kwargs["hidden_states"]
        stream = torch.cuda.current_stream()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        try:
            return original_attention(self, *args, **kwargs)
        finally:
            end.record(stream)
            _PENDING.append({"start": start, "end": end, "layer": int(getattr(_TLS, "layer", -1)),
                             "M": int(hidden.shape[0]), "stream_id": int(stream.cuda_stream), **_rank()})

    if not getattr(Qwen3MoeAttention.forward, "_vision_topk_timer", False):
        layer_init._vision_topk_timer = True
        layer_forward._vision_topk_timer = True
        attention_forward._vision_topk_timer = True
        Qwen3MoeDecoderLayer.__init__ = layer_init
        Qwen3MoeDecoderLayer.forward = layer_forward
        Qwen3MoeAttention.forward = attention_forward
