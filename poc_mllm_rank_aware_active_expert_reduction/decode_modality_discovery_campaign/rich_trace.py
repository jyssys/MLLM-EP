"""Read-only multi-site router-input capture for the discovery campaign."""
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

_INSTALLED = False
_STATE = threading.local()
_STEPS: dict[str, int] = defaultdict(int)
_WRITES: dict[tuple[str, int, int], int] = defaultdict(int)


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_DMC_TRACE_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _layer(prefix: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.mlp$", prefix)
    return int(match.group(1)) if match else -1


def _fingerprint(row: torch.Tensor) -> str:
    bits = row.detach()[:32].contiguous().view(torch.int16).cpu().numpy()
    return hashlib.sha1(bits.tobytes()).hexdigest()


def _write(capture_id: str, layer: int, step: int, payload: dict[str, Any]) -> None:
    dp = int(os.environ.get("VLLM_DP_RANK", -1))
    key = (capture_id, layer, step)
    call = _WRITES[key]
    _WRITES[key] += 1
    root = Path(os.environ["MLLM_DMC_TRACE_RAW"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"site.{capture_id}.dp{dp}.layer{layer}.step{step}.call{call}.npz"
    np.savez_compressed(path, **payload)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel

    original_model = Qwen3MoeLLMModel.forward
    original_init = Qwen3MoeSparseMoeBlock.__init__
    original_block = Qwen3MoeSparseMoeBlock.forward
    original_select = BaseRouter.select_experts

    def model(self: Any, input_ids: torch.Tensor | None, *args: Any, **kwargs: Any):
        entry = _control()
        capture_id = str(entry.get("capture_id", "startup"))
        if capture_id != getattr(_STATE, "capture_id", None):
            _STEPS[capture_id] = 0
        _STATE.capture_id = capture_id
        _STATE.entry = entry
        inputs = kwargs.get("inputs_embeds")
        rows = int(input_ids.numel()) if input_ids is not None else (
            0 if inputs is None else int(inputs.shape[0])
        )
        if rows > 2:
            _STEPS[capture_id] = 0
            _STATE.step = 0
            _STATE.prefill = True
        else:
            _STEPS[capture_id] += 1
            _STATE.step = _STEPS[capture_id]
            _STATE.prefill = False
        return original_model(self, input_ids, *args, **kwargs)

    def block_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._dmc_layer = _layer(str(kwargs.get("prefix", "")))

    def block(self: Any, hidden_states: torch.Tensor):
        previous = getattr(_STATE, "layer", -1)
        _STATE.layer = int(getattr(self, "_dmc_layer", -1))
        try:
            return original_block(self, hidden_states)
        finally:
            _STATE.layer = previous

    def select(self: Any, hidden_states: torch.Tensor, router_logits: torch.Tensor,
               *, input_ids: torch.Tensor | None = None):
        weights, ids = original_select(self, hidden_states, router_logits, input_ids=input_ids)
        entry = getattr(_STATE, "entry", {})
        layer, step = int(getattr(_STATE, "layer", -1)), int(getattr(_STATE, "step", -1))
        if (
            not entry.get("capture", False)
            or bool(getattr(_STATE, "prefill", True))
            or int(os.environ.get("VLLM_DP_RANK", -2)) != int(entry.get("source_dp_rank", -1))
            or int(get_tensor_model_parallel_rank()) != 0
            or layer not in {int(value) for value in entry.get("layers", [])}
            or step not in {int(value) for value in entry.get("steps", [])}
        ):
            return weights, ids
        if hidden_states.shape[0] != 1:
            raise RuntimeError(f"decode capture expects one logical row, got {hidden_states.shape}")
        _write(str(entry["capture_id"]), layer, step, {
            "hidden": hidden_states.detach().to(torch.float16).cpu().numpy(),
            "hidden_fingerprint": np.asarray(_fingerprint(hidden_states[0])),
            "router_logits": router_logits.detach().to(torch.float32).cpu().numpy(),
            "topk_ids": ids.detach().cpu().numpy().astype(np.int16),
            "topk_weights": weights.detach().to(torch.float32).cpu().numpy(),
        })
        return weights, ids

    Qwen3MoeLLMModel.forward = model
    Qwen3MoeSparseMoeBlock.__init__ = block_init
    Qwen3MoeSparseMoeBlock.forward = block
    BaseRouter.select_experts = select
