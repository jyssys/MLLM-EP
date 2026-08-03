"""Phase 2 Qwen3-VL calibration hook interfaces.

These functions define the future integration contract only. They must not
register hooks or run model forwards during Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from calib.collect_stats import CalibrationHookSpec


@dataclass(frozen=True)
class CapturedLayerTensors:
    """Tensor payload captured for one decoder/MoE layer.

    Shapes are intentionally aligned with ``calib.collect_stats``:

    - ``router_logits``: ``[num_tokens, num_experts]``
    - ``attention``: ``[batch, heads, seq, seq]`` or an already-collapsed
      attention tensor accepted by calibration stats
    - ``hidden_states``: ``[num_tokens, hidden_dim]`` expert-input states
    - ``token_type``: ``[num_tokens]`` with ``0=text`` and nonzero vision ids
    - ``mm_token_type_ids``: optional Qwen3-VL multimodal ids, ``0/1/2`` for
      text/image/video before flattening
    """

    layer_idx: int
    router_logits: torch.Tensor
    attention: torch.Tensor
    hidden_states: torch.Tensor
    token_type: torch.Tensor
    mm_token_type_ids: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class ExpertAssignment:
    """Top-k routing tensors consumed by Phase 1 calibration/merge modules."""

    expert_assignment: torch.Tensor
    routing_weights: torch.Tensor
    router_logits: torch.Tensor


@dataclass(frozen=True)
class HookHandles:
    """Opaque future hook handles plus the shared capture buffer."""

    handles: tuple[Any, ...]
    captured_by_layer: MutableMapping[int, CapturedLayerTensors]


LayerCalibrationPayload = Mapping[str, torch.Tensor]


def register_calibration_hooks(
    model: Any,
    hook_plan: Sequence[CalibrationHookSpec],
    *,
    capture_router_logits: bool = True,
    capture_attentions: bool = True,
    capture_hidden_states: bool = True,
    capture_token_masks: bool = True,
) -> HookHandles:
    """Register Phase 2 Qwen3-VL calibration hooks.

    The concrete implementation will attach hooks at the points documented in
    ``docs/model_arch.md``:

    - model/generate entry to capture ``input_ids``, ``mm_token_type_ids``, and
      vision/text masks
    - each decoder layer self-attention module to capture per-layer attention
      weights for adaptive key/redundant vision-token selection
    - each ``Qwen3VLMoeTextSparseMoeBlock`` to capture gate/router logits and
      expert-input hidden states before dispatch

    Returns opaque hook handles and a layer-indexed capture buffer. Phase 1 does
    not register against a real HF model.
    """

    raise NotImplementedError("TODO(Phase2): register against real Qwen3-VL forward")


def extract_routing(
    captured: CapturedLayerTensors,
    *,
    top_k: int,
    num_experts: int | None = None,
    normalize_topk_prob: bool = True,
) -> ExpertAssignment:
    """Convert captured router logits into Phase 1 expert-assignment tensors.

    The future implementation will mirror Qwen3-VL-MoE routing: softmax over
    ``router_logits``, top-k expert selection, optional top-k probability
    normalization, then output:

    - ``expert_assignment``: ``[num_tokens, top_k]`` long tensor
    - ``routing_weights``: ``[num_tokens, top_k]`` float tensor
    - ``router_logits``: original ``[num_tokens, num_experts]`` tensor
    """

    raise NotImplementedError("TODO(Phase2): extract Qwen3-VL router top-k assignments")


def build_calibration_payload(
    captured: CapturedLayerTensors,
    routing: ExpertAssignment,
) -> dict[str, torch.Tensor]:
    """Build the plain tensor payload expected by ``collect_calibration_stats``.

    The returned dictionary is expected to contain ``expert_assignment``,
    ``token_type``, ``hidden_states``, and ``attention`` keys so Phase 1
    calibration aggregation can run unchanged after real hooks are implemented.
    """

    raise NotImplementedError("TODO(Phase2): materialize calibration payload from captured Qwen3-VL tensors")


def remove_hooks(handles: HookHandles) -> None:
    """Remove previously registered HF hook handles."""

    raise NotImplementedError("TODO(Phase2): remove real Qwen3-VL hook handles")

