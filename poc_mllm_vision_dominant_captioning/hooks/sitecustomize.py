"""Opt-in compact router trace hook for the mixed-domain PoC."""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

_STATE = threading.local()
_TRACE: list[dict[str, Any]] = []
_FORWARD = 0
_DECODE: dict[str, int] = {}
_LAST_POSITION: dict[str, tuple[int, ...]] = {}
_SELECTED = {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256}


def _control() -> dict[str, Any]:
    path = os.environ.get("MLLM_VDC_CONTROL")
    if not path or not Path(path).is_file():
        return {}
    try:
        payload = json.loads(Path(path).read_text())
        return payload.get("by_dp", {}).get(os.environ.get("VLLM_DP_RANK", "0"), {})
    except (OSError, json.JSONDecodeError):
        return {}


def _flush() -> None:
    if not _TRACE:
        return
    root = os.environ.get("MLLM_VDC_TRACE_DIR")
    if not root:
        _TRACE.clear()
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"route_summary.pid{os.getpid()}.jsonl"
    with target.open("a") as handle:
        for row in _TRACE:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _TRACE.clear()


def install() -> None:
    if os.environ.get("MLLM_VDC_TRACE") != "1":
        return

    from vllm.distributed import get_ep_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel, Qwen3VLMoeForConditionalGeneration

    original_model = Qwen3MoeLLMModel.forward
    original_embed = Qwen3VLMoeForConditionalGeneration.embed_input_ids
    original_init = Qwen3MoeSparseMoeBlock.__init__
    original_block = Qwen3MoeSparseMoeBlock.forward
    original_select = BaseRouter.select_experts

    def shard(mask: torch.Tensor) -> torch.Tensor:
        size = int(get_tensor_model_parallel_world_size())
        rank = int(get_tensor_model_parallel_rank())
        remainder = int(mask.shape[0]) % size
        if remainder:
            padded = torch.zeros(int(mask.shape[0]) + size - remainder, dtype=mask.dtype, device=mask.device)
            padded[: mask.shape[0]] = mask
            mask = padded
        width = int(mask.shape[0]) // size
        return mask.narrow(0, rank * width, width)

    def embed(self: Any, input_ids: torch.Tensor, multimodal_embeddings: Any = None, *, is_multimodal: torch.Tensor | None = None):
        _STATE.pending_visual = None if is_multimodal is None else is_multimodal.reshape(-1)
        return original_embed(self, input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    def model(self: Any, input_ids: torch.Tensor | None, *args: Any, **kwargs: Any):
        nonlocal original_model
        global _FORWARD
        inputs = kwargs.get("inputs_embeds")
        pending = getattr(_STATE, "pending_visual", None)
        if input_ids is not None:
            image_id = 151655
            mask = input_ids.reshape(-1).eq(image_id)
        else:
            length = 0 if inputs is None else int(inputs.shape[0])
            device = None if inputs is None else inputs.device
            mask = torch.zeros(length, dtype=torch.bool, device=device)
            if pending is not None:
                copied = min(length, int(pending.shape[0]))
                mask[:copied] = pending[:copied]
        control = _control()
        request = str(control.get("request_id", "UNMAPPED"))
        prefill = int(mask.numel()) > 2
        if prefill:
            _DECODE[request] = 0
            _LAST_POSITION.pop(request, None)
            step = None
            position_key = None
            new_position = False
        else:
            positions = kwargs.get("positions")
            if positions is None and args:
                positions = args[0]
            if positions is None:
                position_key = (_FORWARD,)
            else:
                position_key = tuple(
                    int(value)
                    for value in positions.detach().reshape(-1).to(device="cpu").tolist()
                )
            new_position = _LAST_POSITION.get(request) != position_key
            if new_position:
                _DECODE[request] = _DECODE.get(request, 0) + 1
                _LAST_POSITION[request] = position_key
            step = _DECODE[request]
        _FORWARD += 1
        prior = getattr(_STATE, "visual", None)
        _STATE.visual = mask
        _STATE.control = control
        _STATE.forward = _FORWARD
        _STATE.step = step
        _STATE.position_key = position_key
        _STATE.capture = prefill or (new_position and step in _SELECTED)
        try:
            return original_model(self, input_ids, *args, **kwargs)
        finally:
            _STATE.visual = prior
            _STATE.pending_visual = None
            _flush()

    def block_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        match = re.search(r"\.layers\.(\d+)\.mlp$", str(kwargs.get("prefix", "")))
        self._mllm_mdoc_layer = int(match.group(1)) if match else -1

    def block(self: Any, hidden_states: torch.Tensor):
        mask = getattr(_STATE, "visual", None)
        if mask is None:
            mask = torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)
        if mask.shape[0] != hidden_states.shape[0]:
            mask = shard(mask)
        if mask.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(f"visual mask {mask.shape} != hidden {hidden_states.shape}")
        prior = getattr(_STATE, "router_visual", None)
        _STATE.router_visual = mask
        _STATE.layer = int(getattr(self, "_mllm_mdoc_layer", -1))
        try:
            return original_block(self, hidden_states)
        finally:
            _STATE.router_visual = prior

    def select(self: Any, hidden_states: torch.Tensor, router_logits: torch.Tensor, *, input_ids: torch.Tensor | None = None):
        weights, ids = original_select(self, hidden_states, router_logits, input_ids=input_ids)
        control = dict(getattr(_STATE, "control", {}))
        if control.get("request_id") in (None, "UNMAPPED") or not getattr(_STATE, "capture", False):
            return weights, ids
        mask = getattr(_STATE, "router_visual", None)
        if mask is None:
            mask = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        if mask.shape[0] != ids.shape[0]:
            mask = shard(mask)
        valid = ids.ge(0)
        all_ids = ids[valid].to(torch.int64)
        all_weights = weights[valid].to(torch.float32)
        histogram = torch.bincount(all_ids, minlength=128)
        mass = torch.zeros(128, dtype=torch.float32, device=weights.device)
        mass.scatter_add_(0, all_ids, all_weights)
        vision_ids_matrix = ids[mask]
        vision_weights_matrix = weights[mask]
        vision_valid = vision_ids_matrix.ge(0)
        vision_ids = vision_ids_matrix[vision_valid].to(torch.int64)
        vision_weights = vision_weights_matrix[vision_valid].to(torch.float32)
        vision_hist = torch.bincount(vision_ids, minlength=128)
        vision_mass = torch.zeros(128, dtype=torch.float32, device=weights.device)
        if vision_ids.numel():
            vision_mass.scatter_add_(0, vision_ids, vision_weights)
        row = {
            "pid": os.getpid(),
            "forward_id": int(getattr(_STATE, "forward", -1)),
            "request_id": control.get("request_id"),
            "domain": control.get("domain"),
            "analysis_split": control.get("analysis_split"),
            "decode_step": getattr(_STATE, "step", None),
            "position_key": getattr(_STATE, "position_key", None),
            "dp_rank": int(os.environ.get("VLLM_DP_RANK", "0")),
            "tp_rank": int(get_tensor_model_parallel_rank()),
            "source_ep_rank": int(get_ep_group().rank_in_group),
            "ep_world_size": int(get_ep_group().world_size),
            "layer": int(getattr(_STATE, "layer", -1)),
            "rows": int(ids.shape[0]),
            "visual_rows": int(mask.sum().item()),
            "valid_pairs": int(valid.sum().item()),
            "evidence": "INSTRUMENTED_PHYSICAL_ROUTER_TRACE",
        }
        if row["decode_step"] is None:
            row.update({
                "record_type": "prefill",
                "all_expert_histogram": histogram.cpu().tolist(),
                "all_expert_weight_sum": mass.cpu().tolist(),
                "vision_expert_histogram": vision_hist.cpu().tolist(),
                "vision_expert_weight_sum": vision_mass.cpu().tolist(),
                "text_expert_histogram": (histogram - vision_hist).cpu().tolist(),
                "text_expert_weight_sum": (mass - vision_mass).cpu().tolist(),
            })
        else:
            row.update({
                "record_type": "decode",
                "route_ids": ids.detach().to(device="cpu", dtype=torch.int16).tolist(),
                "route_weights": weights.detach().to(device="cpu", dtype=torch.float32).tolist(),
            })
        _TRACE.append(row)
        return weights, ids

    Qwen3VLMoeForConditionalGeneration.embed_input_ids = embed
    Qwen3MoeLLMModel.forward = model
    Qwen3MoeSparseMoeBlock.__init__ = block_init
    Qwen3MoeSparseMoeBlock.forward = block
    BaseRouter.select_experts = select
    atexit.register(_flush)


install()
if os.environ.get("MLLM_VDC_TRACE") == "1":
    os.environ["MLLM_VDC_HOOK_IMPORTED"] = "1"
