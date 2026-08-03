"""Phase 2 DeepSpeed Expert Parallel integration interfaces."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol

import torch

from hooks.register_hooks import ExpertAssignment
from method1.placement import PlacementResult
from method2.merge import MergeResult


class MergeFn(Protocol):
    """Callable shape compatible with ``method2.merge.expert_aware_merge``."""

    def __call__(
        self,
        candidate_mask_or_indices: torch.Tensor,
        expert_assignment: torch.Tensor,
        hidden_states: torch.Tensor,
        importance: torch.Tensor,
        *,
        target_experts: Iterable[int] | torch.Tensor | None = None,
        similarity_threshold: float = 0.9,
        similarity: str = "hidden_cosine",
    ) -> MergeResult:
        """Return expert-aware merge clusters."""


class CapFn(Protocol):
    """Callable shape compatible with ``method2.cap.cap_merge_count``."""

    def __call__(
        self,
        loads: torch.Tensor,
        straggler_gpu: int,
        candidate_count: int,
        max_merge_ratio: float,
        *,
        reduction_per_merge: float = 1.0,
    ) -> dict[str, float | int]:
        """Return cap metadata for stopping at the second-place load."""


def ep_moe_forward_with_merge(
    hidden: torch.Tensor,
    router_out: ExpertAssignment,
    placement: PlacementResult | torch.Tensor,
    merge_fn: MergeFn,
    cap_fn: CapFn,
    *,
    importance: torch.Tensor,
    candidate_mask_or_indices: torch.Tensor | None = None,
    gpu_loads: torch.Tensor | None = None,
    straggler_gpus: torch.Tensor | None = None,
    max_merge_ratio: float = 1.0,
    similarity_threshold: float = 0.9,
    ep_group: object | None = None,
) -> torch.Tensor:
    """Run the future DeepSpeed-EP MoE forward with Method 2 merge inserted.

    Interface contract:

    1. Use ``router_out.expert_assignment`` and ``placement`` to identify
       per-GPU route counts and straggler GPUs.
    2. Select/receive redundant-token candidates and call ``cap_fn`` so merging
       stops once the straggler reaches the second-highest load.
    3. Call ``merge_fn`` with ``hidden``, ``importance``, and top-k expert
       assignments to build expert-local merge clusters.
    4. Insert the merged representation between router output and EP dispatch.
    5. Run DeepSpeed all-to-all dispatch, expert forward, and combine following
       the Qwen3-VL sparse block path documented in ``docs/model_arch.md``.

    Phase 1 does not execute all-to-all, real experts, or combine.
    """

    raise NotImplementedError("TODO(Phase2): DeepSpeed all-to-all dispatch")


def restore_ep_moe_forward(
    model: object,
    *,
    patched_modules: Mapping[str, object],
) -> None:
    """Restore any MoE forward methods patched for Phase 2 EP integration."""

    raise NotImplementedError("TODO(Phase2): restore patched DeepSpeed/Qwen3-VL MoE forwards")

