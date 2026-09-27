"""Bounded, diagnostic expert-vector capture for long-reasoning decode.

The model result is never modified.  For configured decode steps and layers we
re-run each *selected* Top-K branch with unit router weight, store the raw
branch output, and return a clone of the original fused-MoE result.  The files
are deliberately request/layer/step scoped so a stopped run can be audited and
resumed without treating a partial capture as complete.
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
_CALLS_BY_CAPTURE: dict[str, int] = defaultdict(int)
_SOURCE_DECODE_BY_CAPTURE: dict[str, int] = defaultdict(int)
_WRITES_BY_KEY: dict[tuple[str, int, int, int], int] = defaultdict(int)


def _ints(name: str, default: str) -> set[int]:
    return {int(value) for value in os.environ.get(name, default).split(",") if value}


LAYERS = _ints("MLLM_VCDR_CAPTURE_LAYERS", "4,12,24,36,44,47")
STEPS = _ints(
    "MLLM_VCDR_CAPTURE_STEPS",
    ",".join(str(step) for start in (1, 33, 65, 97) for step in range(start, start + 16)),
)


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_VCDR_CAPTURE_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _output_dir() -> Path:
    path = Path(os.environ["MLLM_VCDR_CAPTURE_RAW"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fingerprints(tensor: torch.Tensor) -> list[str]:
    bits = tensor.detach()[:, :32].contiguous().view(torch.int16).cpu().numpy()
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in bits]


def _next_path(kind: str, capture_id: str, layer: int, step: int, rank: int) -> Path:
    key = (capture_id, layer, step, rank)
    invocation = _WRITES_BY_KEY[key]
    _WRITES_BY_KEY[key] += 1
    return _output_dir() / (
        f"{kind}.{capture_id}.rank{rank}.layer{layer}.step{step}.call{invocation}.npz"
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

    original_model_forward = Qwen3MoeLLMModel.forward
    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_select = BaseRouter.select_experts
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        capture_id = str(entry.get("capture_id", ""))
        previous = getattr(_CONTEXT, "capture_id", None)
        if capture_id != previous:
            _CALLS_BY_CAPTURE[capture_id] = 0
            _SOURCE_DECODE_BY_CAPTURE[capture_id] = 0
        call_index = _CALLS_BY_CAPTURE[capture_id]
        _CALLS_BY_CAPTURE[capture_id] += 1
        _CONTEXT.capture_id = capture_id
        _CONTEXT.capture = bool(entry.get("capture"))
        _CONTEXT.source_dp_rank = int(entry.get("source_dp_rank", -1))
        input_ids = args[0] if args else kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        rows = (
            int(input_ids.numel())
            if input_ids is not None
            else (0 if inputs_embeds is None else int(inputs_embeds.shape[0]))
        )
        if int(os.environ.get("VLLM_DP_RANK", -2)) == _CONTEXT.source_dp_rank:
            if rows > 2:
                # A prompt may be split into more than one prefill call.  Each
                # such call resets rather than advances the decode clock.
                _SOURCE_DECODE_BY_CAPTURE[capture_id] = 0
                _CONTEXT.step = 0
            else:
                _SOURCE_DECODE_BY_CAPTURE[capture_id] += 1
                _CONTEXT.step = _SOURCE_DECODE_BY_CAPTURE[capture_id]
        else:
            # Idle/receiver DPs see synchronized dummy calls but cannot infer
            # how many are prefill chunks.  Their filename step is diagnostic;
            # offline merging uses the hidden fingerprint and the source-side
            # router step as the canonical identity.
            _CONTEXT.step = call_index if call_index > 0 else 0
        return original_model_forward(self, *args, **kwargs)

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._vcdr_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(getattr(self, "_vcdr_layer", -1))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def patched_select(
        self: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = original_select(self, hidden_states, router_logits, input_ids=input_ids)
        layer = int(getattr(_CONTEXT, "layer", -1))
        step = int(getattr(_CONTEXT, "step", 0))
        if (
            not getattr(_CONTEXT, "capture", False)
            or layer not in LAYERS
            or step not in STEPS
            or int(os.environ["VLLM_DP_RANK"]) != int(getattr(_CONTEXT, "source_dp_rank", -1))
            or int(get_tensor_model_parallel_rank()) != 0
        ):
            return weights, ids
        capture_id = str(getattr(_CONTEXT, "capture_id", "unknown"))
        rank = int(os.environ["VLLM_DP_RANK"])
        path = _next_path("router", capture_id, layer, step, rank)
        np.savez_compressed(
            path,
            fingerprints=np.asarray(_fingerprints(hidden_states)),
            hidden_states=hidden_states.detach().to(torch.float16).cpu().numpy(),
            token_ids=(
                input_ids.detach().cpu().numpy().astype(np.int64)
                if input_ids is not None
                else np.full(hidden_states.shape[0], -1, np.int64)
            ),
            topk_ids=ids.detach().cpu().numpy().astype(np.int16),
            topk_weights=weights.detach().float().cpu().numpy().astype(np.float32),
        )
        return weights, ids

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_experts(self, *args, **kwargs)
        layer = int(getattr(_CONTEXT, "layer", -1))
        step = int(getattr(_CONTEXT, "step", 0))
        if (
            not getattr(_CONTEXT, "capture", False)
            or layer not in LAYERS
            or step not in STEPS
        ):
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
        forward_result = result.clone()
        sample_ids = values["topk_ids"].contiguous()
        sample_weights = values["topk_weights"].contiguous()
        ep_rank = int(get_ep_group().rank_in_group)
        expert_map = values["expert_map"]
        if expert_map is None:
            start = ep_rank * int(values["local_num_experts"])
            local_mask = (sample_ids >= start) & (
                sample_ids < start + int(values["local_num_experts"])
            )
        else:
            local_mask = expert_map[sample_ids.to(torch.long)] >= 0

        raw_slots = []
        for slot in range(sample_ids.shape[1]):
            if not bool(local_mask[:, slot].any()):
                raw_slots.append(torch.zeros_like(x, dtype=values["in_dtype"]))
                continue
            onehot = torch.zeros_like(sample_weights)
            onehot[:, slot] = 1
            raw_slots.append(
                original_experts(
                    self,
                    values["in_dtype"],
                    x,
                    None,
                    values["w1"],
                    values["w2"],
                    onehot,
                    sample_ids,
                    values["activation"],
                    values["global_num_experts"],
                    values["local_num_experts"],
                    expert_map,
                    values["apply_router_weight_on_input"],
                    None,
                ).clone()
            )
        raw = torch.stack(raw_slots, dim=1)
        capture_id = str(getattr(_CONTEXT, "capture_id", "unknown"))
        path = _next_path("experts", capture_id, layer, step, ep_rank)
        np.savez_compressed(
            path,
            fingerprints=np.asarray(_fingerprints(x)),
            hidden_states=x.detach().to(torch.float16).cpu().numpy(),
            expert_ids=sample_ids.detach().cpu().numpy().astype(np.int16),
            router_weights=sample_weights.detach().float().cpu().numpy().astype(np.float32),
            local_mask=local_mask.detach().cpu().numpy().astype(bool),
            raw_outputs=raw.detach().to(torch.float16).cpu().numpy(),
            stock_output=forward_result.detach().to(torch.float16).cpu().numpy(),
        )
        return forward_result

    Qwen3MoeLLMModel.forward = patched_model_forward
    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    BaseRouter.select_experts = patched_select
    FusedMoEKernelModularImpl._fused_experts = patched_experts
