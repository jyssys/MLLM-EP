"""Opt-in visual-only ragged top-k adapter for the installed vLLM runtime.

The hook is inert unless ``MLLM_REX_ENABLE=1``.  It deliberately keeps the
router's fixed-width top-8 tensor contract and uses -1 ids for invalid visual
slots, which DeepEP and the installed sparse Triton expert path skip.
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

from poc_mllm_resolution_expert_tradeoff.contract import IMAGE_TOKEN_ID
from poc_mllm_resolution_expert_tradeoff.routing import apply_visual_k

_STATE = threading.local()
_TRACE: list[dict[str, Any]] = []
_TRACE_LOCK = threading.Lock()
_CURRENT_LAYER = -1
_DEBUG_COUNTS: dict[str, int] = {}
_FORWARD_COUNTER = 0


def _debug(event: str, payload: dict[str, Any]) -> None:
    target = os.environ.get("MLLM_REX_TRACE_DIR")
    if not target or _DEBUG_COUNTS.get(event, 0) >= 4:
        return
    _DEBUG_COUNTS[event] = _DEBUG_COUNTS.get(event, 0) + 1
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"debug.pid{os.getpid()}.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": event, "pid": os.getpid(), **payload}) + "\n")


def _flush_trace(*, clear: bool = False) -> None:
    target = os.environ.get("MLLM_REX_TRACE_DIR")
    if not target or not _TRACE:
        return
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"route_summary.pid{os.getpid()}.jsonl"
    with path.open("a") as handle:
        for row in _TRACE:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if clear:
        _TRACE.clear()


def install() -> None:
    if os.environ.get("MLLM_REX_ENABLE") != "1":
        return

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
    )
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from vllm.model_executor.models.qwen3_vl_moe import (
        Qwen3MoeLLMModel,
        Qwen3VLMoeForConditionalGeneration,
    )
    from vllm.distributed import (
        get_ep_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    k_visual = int(os.environ.get("MLLM_REX_K_VISUAL", "8"))
    trace_enabled = bool(os.environ.get("MLLM_REX_TRACE_DIR"))
    save_ids = os.environ.get("MLLM_REX_SAVE_IDS") == "1"
    _debug("install", {"k_visual": k_visual, "trace_enabled": trace_enabled})
    original_model_forward = Qwen3MoeLLMModel.forward
    original_embed_input_ids = Qwen3VLMoeForConditionalGeneration.embed_input_ids
    original_block_init = Qwen3MoeSparseMoeBlock.__init__
    original_block_forward = Qwen3MoeSparseMoeBlock.forward
    original_select = BaseRouter.select_experts
    original_capture = RoutedExpertsCapturer.capture

    def sequence_parallel_mask(mask: torch.Tensor) -> torch.Tensor:
        """Apply vLLM's token-axis TP chunking to a one-dimensional mask.

        vLLM's generic helper pads a two-dimensional hidden tensor with the
        four-value ``F.pad`` form.  Reusing it for a 1-D boolean mask fails
        when an asynchronous decode wave shrinks to an odd token count.
        """
        tp_size = int(get_tensor_model_parallel_world_size())
        tp_rank = int(get_tensor_model_parallel_rank())
        remainder = int(mask.shape[0]) % tp_size
        if remainder:
            padded = torch.zeros(
                int(mask.shape[0]) + tp_size - remainder,
                dtype=mask.dtype,
                device=mask.device,
            )
            padded[: mask.shape[0]] = mask
            mask = padded
        chunk = int(mask.shape[0]) // tp_size
        return mask.narrow(0, tp_rank * chunk, chunk)

    def embed_input_ids(
        self: Any,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ):
        # vLLM's V1 multimodal runner deliberately calls the language model
        # with input_ids=None after it has merged image embeddings.  Capture
        # the runner's exact is_multimodal mask at that merge boundary rather
        # than attempting to recover modality from the embedding values.
        _STATE.pending_visual_mask = (
            None if is_multimodal is None else is_multimodal.reshape(-1)
        )
        return original_embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def model_forward(self: Any, input_ids: torch.Tensor | None, *args: Any, **kwargs: Any):
        global _FORWARD_COUNTER
        prior = getattr(_STATE, "forward_visual_mask", None)
        inputs_embeds = kwargs.get("inputs_embeds")
        pending = getattr(_STATE, "pending_visual_mask", None)
        if input_ids is not None:
            forward_mask = input_ids.reshape(-1).eq(IMAGE_TOKEN_ID)
        elif pending is not None and inputs_embeds is not None:
            forward_mask = torch.zeros(
                inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device
            )
            copied = min(int(pending.shape[0]), int(forward_mask.shape[0]))
            forward_mask[:copied] = pending[:copied]
        else:
            length = 0 if inputs_embeds is None else int(inputs_embeds.shape[0])
            device = None if inputs_embeds is None else inputs_embeds.device
            forward_mask = torch.zeros(length, dtype=torch.bool, device=device)
        _STATE.forward_visual_mask = forward_mask
        _STATE.actual_request = pending is not None or input_ids is not None
        if _STATE.actual_request:
            _FORWARD_COUNTER += 1
        _STATE.forward_id = _FORWARD_COUNTER
        _debug("model_forward", {
            "input_ids_none": input_ids is None,
            "input_ids_shape": None if input_ids is None else list(input_ids.shape),
            "pending_mask_shape": None if pending is None else list(pending.shape),
            "forward_mask_shape": list(forward_mask.shape),
            "image_token_count": int(forward_mask.sum().item()),
        })
        try:
            return original_model_forward(self, input_ids, *args, **kwargs)
        finally:
            _STATE.forward_visual_mask = prior
            _STATE.pending_visual_mask = None
            _STATE.actual_request = False
            if trace_enabled:
                _flush_trace(clear=True)

    def block_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_block_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        match = re.search(r"\.layers\.(\d+)\.mlp$", prefix)
        self._mllm_rex_layer = int(match.group(1)) if match else -1

    def block_forward(self: Any, hidden_states: torch.Tensor):
        forward_mask = getattr(_STATE, "forward_visual_mask", None)
        if forward_mask is None:
            visual_mask = torch.zeros(
                hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device
            )
        else:
            visual_mask = forward_mask
            # Qwen3-VL EP4 runs TP2 source shards.  In the installed runtime
            # the language-model input ids can remain global while the hidden
            # rows entering the block are already TP-local, even when the
            # block-level flag does not request another SP chunk.
            if visual_mask.shape[0] != hidden_states.shape[0]:
                visual_mask = sequence_parallel_mask(visual_mask)
            if visual_mask.shape[0] != hidden_states.shape[0]:
                raise RuntimeError(
                    f"cannot align modality mask {visual_mask.shape} to hidden "
                    f"rows {hidden_states.shape}"
                )
        _STATE.visual_mask = visual_mask
        _STATE.layer = int(getattr(self, "_mllm_rex_layer", -1))
        _debug("block_forward", {
            "hidden_shape": list(hidden_states.shape),
            "mask_shape": list(visual_mask.shape),
            "visual_rows": int(visual_mask.sum().item()),
        })
        try:
            return original_block_forward(self, hidden_states)
        finally:
            _STATE.visual_mask = None

    def select(
        self: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ):
        weights, ids = original_select(
            self, hidden_states, router_logits, input_ids=input_ids
        )
        visual_mask = getattr(_STATE, "visual_mask", None)
        # DeepEP's runner can enter sequence-parallel context after the Qwen
        # block wrapper. Align once more at the exact router boundary.
        if visual_mask is not None and visual_mask.shape[0] != weights.shape[0]:
            visual_mask = sequence_parallel_mask(visual_mask)
        _debug("router", {
            "weights_shape": list(weights.shape),
            "mask_shape": None if visual_mask is None else list(visual_mask.shape),
            "visual_rows": 0 if visual_mask is None else int(visual_mask.sum().item()),
        })
        if visual_mask is None or visual_mask.shape[0] != weights.shape[0]:
            if k_visual != weights.shape[-1]:
                raise RuntimeError(
                    "visual-row mask unavailable for a non-baseline K; refusing "
                    f"to change text/decode rows: mask={getattr(visual_mask, 'shape', None)}, "
                    f"weights={weights.shape}"
                )
            return weights, ids

        changed_weights, changed_ids = apply_visual_k(
            weights, ids, visual_mask, k_visual
        )
        if trace_enabled and bool(getattr(_STATE, "actual_request", False)):
            # Diagnostic runs explicitly accept one synchronization per routed
            # layer. Clean timing runs never set MLLM_REX_TRACE_DIR.
            visual_rows = int(visual_mask.sum().item())
            visual_ids = changed_ids[visual_mask]
            valid = visual_ids.ge(0)
            histogram = torch.bincount(
                visual_ids[valid].to(torch.int64), minlength=128
            ).cpu().tolist()
            all_valid = changed_ids.ge(0)
            all_histogram = torch.bincount(
                changed_ids[all_valid].to(torch.int64), minlength=128
            ).cpu().tolist()
            owner = torch.where(
                all_valid, torch.div(changed_ids, 32, rounding_mode="floor"), -1
            )
            # A token hidden row is logically transmitted once to each
            # destination rank even when several of its experts share it.
            dest_presence = torch.stack(
                [owner.eq(rank).any(dim=-1) for rank in range(4)], dim=-1
            )
            source_ep_rank = int(get_ep_group().rank_in_group)
            remote_presence = dest_presence.clone()
            remote_presence[:, source_ep_rank] = False
            visual_owner = torch.where(
                valid, torch.div(visual_ids, 32, rounding_mode="floor"), -1
            )
            visual_dest_presence = torch.stack(
                [visual_owner.eq(rank).any(dim=-1) for rank in range(4)], dim=-1
            )
            visual_remote_presence = visual_dest_presence.clone()
            visual_remote_presence[:, source_ep_rank] = False
            with _TRACE_LOCK:
                _TRACE.append({
                        "pid": os.getpid(),
                        "forward_id": int(getattr(_STATE, "forward_id", -1)),
                        "dp_rank": int(os.environ.get("VLLM_DP_RANK", "0")),
                        "tp_rank": int(get_tensor_model_parallel_rank()),
                        "source_ep_rank": source_ep_rank,
                        "layer": int(getattr(_STATE, "layer", _CURRENT_LAYER)),
                        "all_rows": int(changed_ids.shape[0]),
                        "all_valid_pairs": int(all_valid.sum().item()),
                        "all_expert_histogram": all_histogram,
                        "all_rank_pair_histogram": [
                            int(sum(all_histogram[rank * 32:(rank + 1) * 32]))
                            for rank in range(4)
                        ],
                        "all_destination_rows": int(dest_presence.sum().item()),
                        "all_remote_destination_rows": int(remote_presence.sum().item()),
                        "visual_rows": visual_rows,
                        "k_visual": k_visual,
                        "valid_visual_pairs": int(valid.sum().item()),
                        "invalid_visual_slots": int((~valid).sum().item()),
                        "expert_histogram": histogram,
                        "visual_rank_pair_histogram": [
                            int(sum(histogram[rank * 32:(rank + 1) * 32]))
                            for rank in range(4)
                        ],
                        "visual_destination_rows": int(
                            visual_dest_presence.sum().item()
                        ),
                        "visual_remote_destination_rows": int(
                            visual_remote_presence.sum().item()
                        ),
                        "logical_dispatch_bf16_bytes": int(
                            dest_presence.sum().item() * 2048 * 2
                        ),
                        "logical_remote_dispatch_bf16_bytes": int(
                            remote_presence.sum().item() * 2048 * 2
                        ),
                        "evidence": "DIAGNOSTIC_INSTRUMENTED_ROUTE_SUMMARY",
                })
            if save_ids:
                import numpy as np

                route_root = Path(os.environ["MLLM_REX_TRACE_DIR"]) / "raw" / "routes"
                route_root.mkdir(parents=True, exist_ok=True)
                route_name = (
                    f"route_pid{os.getpid()}_f{int(getattr(_STATE, 'forward_id', -1))}"
                    f"_ep{source_ep_rank}_layer{int(getattr(_STATE, 'layer', -1))}.npy"
                )
                np.save(
                    route_root / route_name,
                    changed_ids.detach().to(device="cpu", dtype=torch.int16).numpy(),
                    allow_pickle=False,
                )
        return changed_weights, changed_ids

    def capture(self: Any, layer_id: int, topk_ids: torch.Tensor) -> None:
        global _CURRENT_LAYER
        _CURRENT_LAYER = int(layer_id)
        return original_capture(self, layer_id, topk_ids)

    Qwen3MoeLLMModel.forward = model_forward
    Qwen3VLMoeForConditionalGeneration.embed_input_ids = embed_input_ids
    Qwen3MoeSparseMoeBlock.__init__ = block_init
    Qwen3MoeSparseMoeBlock.forward = block_forward
    BaseRouter.select_experts = select
    RoutedExpertsCapturer.capture = capture
    atexit.register(_flush_trace)


install()
if os.environ.get("MLLM_REX_ENABLE") == "1":
    # Inherited by vLLM's engine/model-worker grandchildren and checked by
    # the launcher.  This makes a missing startup hook a hard error instead
    # of silently running the baseline top-8 path.
    os.environ["MLLM_REX_HOOK_IMPORTED"] = "1"

# Compose the Part-1 read-only CUDA profilers after the mixed-k adapter so the
# profiler measures the actual custom path instead of replacing it.  These
# branches remain inert in clean semantic/timing runs.
if os.environ.get("MLLM_EP_FOUNDATION_PROFILE") == "1":
    from poc_flashvep.live_traffic_matrix_validation.instrumentation import (
        install as install_ep_profile,
    )
    from poc_mllm_ep_foundation.profiling import install as install_full_profile

    install_ep_profile()
    install_full_profile()
elif os.environ.get("MLLM_EP_FOUNDATION_PROFILE_EP") == "1":
    from poc_flashvep.live_traffic_matrix_validation.instrumentation import (
        install as install_ep_profile,
    )

    install_ep_profile()
elif os.environ.get("MLLM_EP_FOUNDATION_PROFILE_FULL") == "1":
    from poc_mllm_ep_foundation.profiling import install as install_full_profile

    install_full_profile()
