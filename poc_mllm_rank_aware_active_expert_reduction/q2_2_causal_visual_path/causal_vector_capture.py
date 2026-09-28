"""Exact-path diagnostic capture for the bounded Q2-2 causal comparison.

The hook never changes model output. At one frozen layer/step per condition it
stores the source router row and recomputes each naturally selected branch with
unit weight. Clean timing is explicitly out of scope.
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

_INSTALLED = False
_CONTEXT = threading.local()
_CALLS: dict[str, int] = defaultdict(int)
_SOURCE_STEP: dict[str, int] = defaultdict(int)
_WRITES: dict[tuple[str, str, int], int] = defaultdict(int)


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_Q22_CAPTURE_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _raw_dir() -> Path:
    path = Path(os.environ["MLLM_Q22_CAPTURE_RAW"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _fingerprints(tensor: torch.Tensor) -> list[str]:
    bits = tensor.detach()[:, :32].contiguous().view(torch.int16).cpu().numpy()
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in bits]


def _path(kind: str, capture_id: str, layer: int, step: int, rank: int) -> Path:
    key = (kind, capture_id, rank)
    call = _WRITES[key]
    _WRITES[key] += 1
    return _raw_dir() / (
        f"{kind}.{capture_id}.rank{rank}.layer{layer}.step{step}.call{call}.npz"
    )


def _enabled() -> bool:
    entry = getattr(_CONTEXT, "entry", {})
    return (
        bool(entry.get("capture"))
        and int(getattr(_CONTEXT, "layer", -1)) == int(entry.get("target_layer", -2))
        and int(getattr(_CONTEXT, "step", -1)) == int(entry.get("target_step", -2))
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_ep_group, get_tensor_model_parallel_rank
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel

    model_forward = Qwen3MoeLLMModel.forward
    layer_init = Qwen3MoeDecoderLayer.__init__
    layer_forward = Qwen3MoeDecoderLayer.forward
    select_experts = BaseRouter.select_experts
    fused_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        capture_id = str(entry.get("capture_id", "startup"))
        previous = getattr(_CONTEXT, "capture_id", None)
        if capture_id != previous:
            _CALLS[capture_id] = 0
            _SOURCE_STEP[capture_id] = 0
        call_index = _CALLS[capture_id]
        _CALLS[capture_id] += 1
        _CONTEXT.capture_id = capture_id
        _CONTEXT.entry = entry
        input_ids = args[0] if args else kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        rows = int(input_ids.numel()) if input_ids is not None else (
            0 if inputs_embeds is None else int(inputs_embeds.shape[0])
        )
        if int(os.environ.get("VLLM_DP_RANK", -2)) == int(entry.get("source_dp_rank", -1)):
            if rows > 2:
                _SOURCE_STEP[capture_id] = 0
                _CONTEXT.step = 0
            else:
                _SOURCE_STEP[capture_id] += 1
                _CONTEXT.step = _SOURCE_STEP[capture_id]
        else:
            _CONTEXT.step = call_index if call_index > 0 else 0
        return model_forward(self, *args, **kwargs)

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        layer_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._q22_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(getattr(self, "_q22_layer", -1))
        try:
            return layer_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def patched_select(
        self: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = select_experts(self, hidden_states, router_logits, input_ids=input_ids)
        entry = getattr(_CONTEXT, "entry", {})
        if (
            not _enabled()
            or int(os.environ["VLLM_DP_RANK"]) != int(entry.get("source_dp_rank", -1))
            or int(get_tensor_model_parallel_rank()) != 0
        ):
            return weights, ids
        capture_id = str(entry["capture_id"])
        layer = int(_CONTEXT.layer)
        step = int(_CONTEXT.step)
        rank = int(os.environ["VLLM_DP_RANK"])
        np.savez_compressed(
            _path("router", capture_id, layer, step, rank),
            fingerprints=np.asarray(_fingerprints(hidden_states)),
            hidden_states=hidden_states.detach().to(torch.float16).cpu().numpy(),
            router_logits=router_logits.detach().float().cpu().numpy().astype(np.float32),
            topk_ids=ids.detach().cpu().numpy().astype(np.int16),
            topk_weights=weights.detach().float().cpu().numpy().astype(np.float32),
        )
        return weights, ids

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = fused_experts(self, *args, **kwargs)
        if not _enabled():
            return result
        names = (
            "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
            "topk_ids", "activation", "global_num_experts", "local_num_experts",
            "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
        )
        values = dict(zip(names, args, strict=False))
        values.update(kwargs)
        x = values["a1q"]
        if x.numel() == 0:
            return result
        ids = values["topk_ids"].contiguous()
        weights = values["topk_weights"].contiguous()
        ep_rank = int(get_ep_group().rank_in_group)
        expert_map = values["expert_map"]
        if expert_map is None:
            start = ep_rank * int(values["local_num_experts"])
            local_mask = (ids >= start) & (ids < start + int(values["local_num_experts"]))
        else:
            local_mask = expert_map[ids.to(torch.long)] >= 0
        raw_slots = []
        for slot in range(ids.shape[1]):
            if not bool(local_mask[:, slot].any()):
                raw_slots.append(torch.zeros_like(x, dtype=values["in_dtype"]))
                continue
            onehot = torch.zeros_like(weights)
            onehot[:, slot] = 1
            raw_slots.append(fused_experts(
                self, values["in_dtype"], x, None, values["w1"], values["w2"],
                onehot, ids, values["activation"], values["global_num_experts"],
                values["local_num_experts"], expert_map,
                values["apply_router_weight_on_input"], None,
            ).clone())
        entry = getattr(_CONTEXT, "entry", {})
        np.savez_compressed(
            _path("experts", str(entry["capture_id"]), int(_CONTEXT.layer), int(_CONTEXT.step), ep_rank),
            fingerprints=np.asarray(_fingerprints(x)),
            hidden_states=x.detach().to(torch.float16).cpu().numpy(),
            expert_ids=ids.detach().cpu().numpy().astype(np.int16),
            router_weights=weights.detach().float().cpu().numpy().astype(np.float32),
            local_mask=local_mask.detach().cpu().numpy().astype(bool),
            raw_outputs=torch.stack(raw_slots, dim=1).detach().to(torch.float16).cpu().numpy(),
            stock_output=result.detach().to(torch.float16).cpu().numpy(),
        )
        return result

    Qwen3MoeLLMModel.forward = patched_model_forward
    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    BaseRouter.select_experts = patched_select
    FusedMoEKernelModularImpl._fused_experts = patched_experts
