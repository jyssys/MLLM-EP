"""Capture routing-weight-free outputs of selected stock Triton experts.

The diagnostic calls never feed their outputs back into the model.  Each call
keeps the original hidden state and Top-8 IDs, sets one selected slot weight to
one and all others to zero, and invokes the same stock expert implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

LAYERS = {4, 8, 12, 20, 24, 28, 36, 40, 44, 47}
_INSTALLED = False
_CONTEXT = threading.local()
_USED_CAPTURES: set[str] = set()


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_FUNCTIONAL_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fingerprints(tensor: torch.Tensor) -> list[str]:
    # The first 32 BF16 values provide a stable 512-bit dispatch identity while
    # avoiding a full hidden-state device-to-host copy.
    bits = tensor.detach()[:, :32].contiguous().view(torch.int16).cpu().numpy()
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in bits]


def _selected(fingerprints: list[str]) -> np.ndarray:
    # Preregistered deterministic 25% sample, identical on every EP rank.
    return np.asarray([int(value[:2], 16) < 64 for value in fingerprints], dtype=bool)


def _output_dir() -> Path:
    path = Path(os.environ["FLASHVEP_FUNCTIONAL_RAW"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.distributed import get_ep_group, get_tensor_model_parallel_rank
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_select = BaseRouter.select_experts
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._functional_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous_layer = getattr(_CONTEXT, "layer", -1)
        previous_active = getattr(_CONTEXT, "active", False)
        previous_capture_id = getattr(_CONTEXT, "capture_id", "")
        layer = int(getattr(self, "_functional_layer", -1))
        if layer == 0:
            entry = _control()
            capture_id = str(entry.get("capture_id", ""))
            active = bool(entry.get("capture")) and capture_id not in _USED_CAPTURES
            _CONTEXT.active = active
            _CONTEXT.capture_id = capture_id
        _CONTEXT.layer = layer
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous_layer
            if layer == 47:
                _CONTEXT.active = previous_active
                _CONTEXT.capture_id = previous_capture_id

    def patched_select(self: Any, hidden_states: torch.Tensor,
                       router_logits: torch.Tensor, *,
                       input_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = original_select(self, hidden_states, router_logits, input_ids=input_ids)
        layer = int(getattr(_CONTEXT, "layer", -1))
        if not getattr(_CONTEXT, "active", False) or layer not in LAYERS:
            return weights, ids
        # Idle DP router rows are small, but that worker can still receive real
        # expert work from the active DP through DeepEP. Router metadata is only
        # needed from the source DP; expert capture remains active.
        if hidden_states.shape[0] < 50:
            return weights, ids
        dp_rank = int(os.environ["VLLM_DP_RANK"])
        tp_rank = int(get_tensor_model_parallel_rank())
        capture_id = str(getattr(_CONTEXT, "capture_id", "unknown"))
        path = _output_dir() / f"router.{capture_id}.dp{dp_rank}.tp{tp_rank}.layer{layer}.npz"
        if path.exists():
            raise FileExistsError(path)
        fingerprints = _fingerprints(hidden_states)
        token_ids = (input_ids.detach().cpu().numpy().astype(np.int64)
                     if input_ids is not None else np.full(len(fingerprints), -1, np.int64))
        np.savez_compressed(
            path, fingerprints=np.asarray(fingerprints), selected=_selected(fingerprints),
            token_ids=token_ids, topk_ids=ids.detach().cpu().numpy().astype(np.int16),
            topk_weights=weights.detach().float().cpu().numpy().astype(np.float32),
        )
        return weights, ids

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_experts(self, *args, **kwargs)
        layer = int(getattr(_CONTEXT, "layer", -1))
        if not getattr(_CONTEXT, "active", False) or layer not in LAYERS:
            return result
        names = ("in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                 "topk_ids", "activation", "global_num_experts", "local_num_experts",
                 "expert_map", "apply_router_weight_on_input", "expert_tokens_meta")
        values = dict(zip(names, args, strict=False)); values.update(kwargs)
        x = values["a1q"]
        # Reject a previous wave's one-row decode/padding call even if it saw
        # the next control file. The real DeepEP prefill has a bounded >50-row
        # expert batch on every EP worker, including the idle source DP.
        if layer == min(LAYERS) and x.shape[0] < 50:
            _CONTEXT.active = False
            return result
        if layer == min(LAYERS):
            _USED_CAPTURES.add(str(getattr(_CONTEXT, "capture_id", "")))
        # TritonExperts may return a view backed by a reusable internal output
        # buffer.  Preserve the real forward result before the diagnostic
        # per-slot calls reuse that buffer, and return this preserved value.
        forward_result = result.clone()
        fingerprints = _fingerprints(x)
        sample_mask = _selected(fingerprints)
        selected_indices = np.flatnonzero(sample_mask)
        ep_rank = int(get_ep_group().rank_in_group)
        capture_id = str(getattr(_CONTEXT, "capture_id", "unknown"))
        path = _output_dir() / f"experts.{capture_id}.ep{ep_rank}.layer{layer}.npz"
        if path.exists():
            raise FileExistsError(path)
        if not len(selected_indices):
            np.savez_compressed(path, fingerprints=np.asarray([], dtype="U40"),
                                expert_ids=np.empty((0, 8), np.int16),
                                router_weights=np.empty((0, 8), np.float32),
                                local_mask=np.empty((0, 8), bool),
                                raw_outputs=np.empty((0, 8, x.shape[-1]), np.float16),
                                stock_output=np.empty((0, x.shape[-1]), np.float16))
            return forward_result
        index = torch.as_tensor(selected_indices, device=x.device, dtype=torch.long)
        sample_x = x.index_select(0, index).contiguous()
        sample_ids = values["topk_ids"].index_select(0, index).contiguous()
        sample_weights = values["topk_weights"].index_select(0, index).contiguous()
        expert_map = values["expert_map"]
        if expert_map is None:
            start = ep_rank * int(values["local_num_experts"])
            local_mask = (sample_ids >= start) & (sample_ids < start + int(values["local_num_experts"]))
        else:
            local_mask = expert_map[sample_ids.to(torch.long)] >= 0
        raw_slots = []
        for slot in range(sample_ids.shape[1]):
            if not bool(local_mask[:, slot].any()):
                raw_slots.append(torch.zeros_like(sample_x, dtype=values["in_dtype"]))
                continue
            onehot = torch.zeros_like(sample_weights)
            onehot[:, slot] = 1
            raw_slots.append(original_experts(
                self, values["in_dtype"], sample_x, None, values["w1"], values["w2"],
                onehot, sample_ids, values["activation"], values["global_num_experts"],
                values["local_num_experts"], expert_map,
                values["apply_router_weight_on_input"], None).clone())
        raw = torch.stack(raw_slots, dim=1)
        np.savez_compressed(
            path,
            fingerprints=np.asarray([fingerprints[i] for i in selected_indices]),
            expert_ids=sample_ids.detach().cpu().numpy().astype(np.int16),
            router_weights=sample_weights.detach().float().cpu().numpy().astype(np.float32),
            local_mask=local_mask.detach().cpu().numpy().astype(bool),
            raw_outputs=raw.detach().to(torch.float16).cpu().numpy(),
            stock_output=forward_result.index_select(0, index).detach().to(torch.float16).cpu().numpy(),
        )
        return forward_result

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    BaseRouter.select_experts = patched_select
    FusedMoEKernelModularImpl._fused_experts = patched_experts
