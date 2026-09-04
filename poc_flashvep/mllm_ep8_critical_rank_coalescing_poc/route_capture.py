"""Read-only token-level router capture for real Qwen3-VL requests.

The hook records the router's actual top-k IDs/weights and the input token IDs
from the immutable per-wave control manifest.  Hidden vectors are sampled only
for vision tokens at preregistered high-straggler layers to keep the trace
portable.  No route, expert placement, or scheduler decision is changed.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

_INSTALLED = False
_CTX = threading.local()
_SEEN: set[tuple[int, int, int, int]] = set()
_LAYERS = {16, 24, 40}
_MAX_HIDDEN = 128


def _layer_of(module: Any) -> int:
    text = str(getattr(module, "layer_name", ""))
    m = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", text)
    return int(m.group(1)) if m else -1


def _layer_of_prefix(prefix: str) -> int:
    m = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(m.group(1)) if m else -1


def _control() -> dict[str, Any]:
    path = os.environ.get("FLASHVEP_ROUTE_CONTROL", "")
    try:
        return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
    except Exception:
        return {}


def _save(layer: int, entry: dict[str, Any], hidden: torch.Tensor,
          weights: torch.Tensor, ids: torch.Tensor,
          input_ids: torch.Tensor | None = None) -> None:
    if not entry.get("capture") or not bool(entry.get("measured", False)):
        return
    wave = int(entry.get("wave", -1)); rank = int(os.environ.get("VLLM_DP_RANK", -1))
    # Each DP engine has two TP workers.  They share VLLM_DP_RANK, so a
    # dp-only filename would be concurrently overwritten by the two workers.
    # Keep one self-contained file per EP rank and select EP0 as the canonical
    # route view during analysis.
    try:
        from vllm.distributed import get_ep_group
        ep_rank = int(get_ep_group().rank_in_group)
    except Exception:
        ep_rank = int(os.environ.get("LOCAL_RANK", -1))
    key = (wave, layer, rank, ep_rank)
    if key in _SEEN:
        return
    _SEEN.add(key)
    # In sequence-parallel MoE the router sees only this TP worker's token
    # shard.  Prefer the actual input_ids passed by vLLM when available; the
    # full prompt manifest is a fallback for Qwen3MoeSparseMoeBlock, which
    # currently does not forward input_ids.  In that case recover contiguous
    # TP chunk positions exactly as sequence_parallel_chunk_impl does.
    token_positions: np.ndarray
    if input_ids is not None:
        token_ids = input_ids.detach().to("cpu", dtype=torch.int64).reshape(-1).numpy()
        token_positions = np.arange(len(token_ids), dtype=np.int64)
    else:
        full_ids = np.asarray(entry.get("token_ids", []), dtype=np.int64)
        try:
            from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
            tp_rank = int(get_tensor_model_parallel_rank())
            tp_size = int(get_tensor_model_parallel_world_size())
        except Exception:
            tp_rank, tp_size = 0, 1
        chunk = int((len(full_ids) + max(tp_size, 1) - 1) // max(tp_size, 1))
        start = tp_rank * chunk
        token_positions = start + np.arange(int(ids.shape[0]), dtype=np.int64)
        token_ids = np.full((int(ids.shape[0]),), -1, dtype=np.int64)
        valid = token_positions < len(full_ids)
        token_ids[valid] = full_ids[token_positions[valid]]
    n = int(ids.shape[0]);
    if len(token_ids) != n:
        # vLLM should give one full prefill call under the bounded prompt
        # lengths.  Preserve the mismatch explicitly rather than inventing
        # token labels.
        token_ids = np.full((n,), -1, dtype=np.int64)
        token_positions = np.full((n,), -1, dtype=np.int64)
    modality = np.where(token_ids == 151655, 1, 0).astype(np.int8)
    dest = ids.detach().to("cpu", dtype=torch.int64).numpy() // 16
    out = Path(os.environ.get("FLASHVEP_ROUTE_RAW_DIR", "")); out.mkdir(parents=True, exist_ok=True)
    stem = f"route_wave{wave}_layer{layer}_dp{rank}_ep{ep_rank}"
    payload: dict[str, Any] = {
        "token_ids": token_ids,
        "token_positions": token_positions,
        "modality": modality,
        "topk_ids": ids.detach().to("cpu", dtype=torch.int16).numpy(),
        "topk_weights": weights.detach().to("cpu", dtype=torch.float32).numpy(),
        "dest_ranks": dest.astype(np.int8),
        "layer": np.asarray(layer, dtype=np.int16),
        "wave": np.asarray(wave, dtype=np.int16),
        "source_dp_rank": np.asarray(int(entry.get("source_dp_rank", -1)), dtype=np.int16),
    }
    if layer in _LAYERS and n:
        vision = np.flatnonzero(modality == 1)[:_MAX_HIDDEN]
        if vision.size:
            payload["hidden_positions"] = vision.astype(np.int32)
            payload["hidden_states_fp16"] = hidden.detach().to("cpu", dtype=torch.float16).numpy()[vision]
    np.savez_compressed(out / f"{stem}.npz", **payload)
    meta = {
        "path": str(out / f"{stem}.npz"), "wave": wave, "layer": layer,
        "worker_dp_rank": rank, "ep_rank": ep_rank, "rows": n, "vision_rows": int(modality.sum()),
        "text_rows": int((modality == 0).sum()), "source_dp_rank": int(entry.get("source_dp_rank", -1)),
        "capture_type": "real_router_select_experts",
    }
    with (out / "route_index.jsonl").open("a") as stream:
        stream.write(json.dumps(meta, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    original_init = Qwen3MoeDecoderLayer.__init__
    original_select = BaseRouter.select_experts
    original_forward = Qwen3MoeDecoderLayer.forward

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_route_layer = _layer_of_prefix(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        prior = getattr(_CTX, "layer", -1)
        _CTX.layer = int(getattr(self, "_flashvep_route_layer", _layer_of(self)))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CTX.layer = prior

    def patched_select(self: Any, hidden_states: torch.Tensor,
                       router_logits: torch.Tensor, *,
                       input_ids: torch.Tensor | None = None):
        weights, ids = original_select(self, hidden_states, router_logits, input_ids=input_ids)
        entry = _control(); layer = int(getattr(_CTX, "layer", -1))
        if entry.get("capture") and bool(entry.get("measured", False)) and layer >= 0:
            try:
                _save(layer, entry, hidden_states, weights, ids, input_ids=input_ids)
            except Exception as exc:
                out = os.environ.get("FLASHVEP_ROUTE_RAW_DIR", "")
                if out:
                    Path(out).mkdir(parents=True, exist_ok=True)
                    with (Path(out) / "capture_errors.log").open("a") as stream:
                        stream.write(f"layer={layer} {type(exc).__name__}: {exc}\n")
        return weights, ids

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    BaseRouter.select_experts = patched_select
