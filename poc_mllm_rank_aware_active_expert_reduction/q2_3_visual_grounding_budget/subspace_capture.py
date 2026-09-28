"""Bounded prefill-subspace and decode-site capture for Q2-3.

The hook never changes model outputs. It stores at most 16 visual and 16 text
prefill rows per TP shard at the frozen target layer, plus one decode router
input/route at the frozen layer and step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import SAMPLE_ROWS_PER_TP

_INSTALLED = False
_STATE = threading.local()
_CALLS: dict[tuple[str, str, int, int], int] = defaultdict(int)
_DECODE_STEP: dict[str, int] = defaultdict(int)


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_VGB_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _layer(prefix: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.mlp$", prefix)
    return int(match.group(1)) if match else -1


def _shard(mask: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    size = int(get_tensor_model_parallel_world_size())
    rank = int(get_tensor_model_parallel_rank())
    remainder = int(mask.shape[0]) % size
    if remainder:
        padded = torch.zeros(mask.shape[0] + size - remainder, dtype=mask.dtype, device=mask.device)
        padded[: mask.shape[0]] = mask
        mask = padded
    width = int(mask.shape[0]) // size
    return mask.narrow(0, rank * width, width)


def _sample(rows: torch.Tensor, cap: int = SAMPLE_ROWS_PER_TP) -> np.ndarray:
    if rows.shape[0] <= cap:
        chosen = rows
    else:
        indices = torch.linspace(0, rows.shape[0] - 1, cap, device=rows.device).round().long()
        chosen = rows.index_select(0, indices)
    return chosen.detach().to(torch.float16).cpu().numpy()


def _fingerprint(row: torch.Tensor) -> str:
    bits = row.detach()[:32].contiguous().view(torch.int16).cpu().numpy()
    return hashlib.sha1(bits.tobytes()).hexdigest()


def _write(kind: str, capture_id: str, layer: int, step: int, payload: dict[str, Any]) -> None:
    from vllm.distributed import get_tensor_model_parallel_rank

    dp = int(os.environ.get("VLLM_DP_RANK", -1))
    tp = int(get_tensor_model_parallel_rank())
    key = (kind, capture_id, layer, tp)
    call = _CALLS[key]
    _CALLS[key] += 1
    root = Path(os.environ["MLLM_VGB_RAW"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{kind}.{capture_id}.dp{dp}.tp{tp}.layer{layer}.step{step}.call{call}.npz"
    np.savez_compressed(path, **payload)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel, Qwen3VLMoeForConditionalGeneration

    original_embed = Qwen3VLMoeForConditionalGeneration.embed_input_ids
    original_model = Qwen3MoeLLMModel.forward
    original_init = Qwen3MoeSparseMoeBlock.__init__
    original_block = Qwen3MoeSparseMoeBlock.forward
    original_select = BaseRouter.select_experts

    def embed(self: Any, input_ids: torch.Tensor, multimodal_embeddings: Any = None,
              *, is_multimodal: torch.Tensor | None = None):
        _STATE.pending_visual = None if is_multimodal is None else is_multimodal.reshape(-1)
        return original_embed(self, input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    def model(self: Any, input_ids: torch.Tensor | None, *args: Any, **kwargs: Any):
        entry = _control()
        capture_id = str(entry.get("capture_id", "startup"))
        if capture_id != getattr(_STATE, "capture_id", None):
            _DECODE_STEP[capture_id] = 0
        _STATE.capture_id = capture_id
        _STATE.entry = entry
        inputs = kwargs.get("inputs_embeds")
        pending = getattr(_STATE, "pending_visual", None)
        if input_ids is not None:
            mask = input_ids.reshape(-1).eq(151655)
        else:
            length = 0 if inputs is None else int(inputs.shape[0])
            device = None if inputs is None else inputs.device
            mask = torch.zeros(length, dtype=torch.bool, device=device)
            if pending is not None:
                copied = min(length, int(pending.shape[0]))
                mask[:copied] = pending[:copied]
        prefill = int(mask.numel()) > 2
        if prefill:
            _DECODE_STEP[capture_id] = 0
            _STATE.step = 0
        else:
            _DECODE_STEP[capture_id] += 1
            _STATE.step = _DECODE_STEP[capture_id]
        prior = getattr(_STATE, "visual", None)
        _STATE.visual = mask
        _STATE.prefill = prefill
        try:
            return original_model(self, input_ids, *args, **kwargs)
        finally:
            _STATE.visual = prior
            _STATE.pending_visual = None

    def block_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._vgb_layer = _layer(str(kwargs.get("prefix", "")))

    def block(self: Any, hidden_states: torch.Tensor):
        mask = getattr(_STATE, "visual", None)
        if mask is None:
            mask = torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)
        if mask.shape[0] != hidden_states.shape[0]:
            mask = _shard(mask)
        if mask.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(f"visual mask {mask.shape} != hidden {hidden_states.shape}")
        prior_mask = getattr(_STATE, "router_visual", None)
        prior_layer = getattr(_STATE, "layer", -1)
        _STATE.router_visual = mask
        _STATE.layer = int(getattr(self, "_vgb_layer", -1))
        try:
            return original_block(self, hidden_states)
        finally:
            _STATE.router_visual = prior_mask
            _STATE.layer = prior_layer

    def select(self: Any, hidden_states: torch.Tensor, router_logits: torch.Tensor,
               *, input_ids: torch.Tensor | None = None):
        weights, ids = original_select(self, hidden_states, router_logits, input_ids=input_ids)
        entry = getattr(_STATE, "entry", {})
        if (
            not entry.get("capture", False)
            or int(os.environ.get("VLLM_DP_RANK", -2)) != int(entry.get("source_dp_rank", -1))
        ):
            return weights, ids
        layer = int(getattr(_STATE, "layer", -1))
        if layer != int(entry.get("target_layer", -2)):
            return weights, ids
        capture_id = str(entry["capture_id"])
        if getattr(_STATE, "prefill", False):
            mask = getattr(_STATE, "router_visual", None)
            if mask is None:
                return weights, ids
            # Sequence parallelism may shard between the sparse block entry
            # and router selection, so reconcile once more at the consumer.
            if mask.shape[0] != hidden_states.shape[0]:
                mask = _shard(mask)
            if mask.shape[0] != hidden_states.shape[0]:
                raise RuntimeError(f"router visual mask {mask.shape} != hidden {hidden_states.shape}")
            visual = hidden_states[mask]
            text = hidden_states[~mask]
            _write("prefill", capture_id, layer, 0, {
                "visual_rows": _sample(visual),
                "text_rows": _sample(text),
                "visual_count": np.asarray(int(visual.shape[0]), dtype=np.int64),
                "text_count": np.asarray(int(text.shape[0]), dtype=np.int64),
            })
        elif (
            int(getattr(_STATE, "step", -1)) == int(entry.get("probe_step", -2))
            and int(get_tensor_model_parallel_rank()) == 0
        ):
            if hidden_states.shape[0] != 1:
                raise RuntimeError(f"decode target expected one row, got {hidden_states.shape}")
            _write("decode", capture_id, layer, int(_STATE.step), {
                "hidden": hidden_states.detach().to(torch.float16).cpu().numpy(),
                "hidden_fingerprint": np.asarray(_fingerprint(hidden_states[0])),
                "topk_ids": ids.detach().cpu().numpy().astype(np.int16),
                "topk_weights": weights.detach().float().cpu().numpy().astype(np.float32),
            })
        return weights, ids

    Qwen3VLMoeForConditionalGeneration.embed_input_ids = embed
    Qwen3MoeLLMModel.forward = model
    Qwen3MoeSparseMoeBlock.__init__ = block_init
    Qwen3MoeSparseMoeBlock.forward = block
    BaseRouter.select_experts = select
