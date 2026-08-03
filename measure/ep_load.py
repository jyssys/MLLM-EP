"""Expert-parallel load aggregation utilities.

These helpers operate on already-captured routed expert ids from vanilla vLLM
EP. They do not modify model placement, dispatch, merge, or routing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EPLoadBreakdown:
    """Vision/text load counts for experts and EP ranks."""

    expert_vision: np.ndarray
    expert_text: np.ndarray
    rank_vision: np.ndarray
    rank_text: np.ndarray

    @property
    def expert_total(self) -> np.ndarray:
        return self.expert_vision + self.expert_text

    @property
    def rank_total(self) -> np.ndarray:
        return self.rank_vision + self.rank_text


def expert_to_rank(
    expert_ids: np.ndarray,
    *,
    num_experts: int = 128,
    ep_degree: int = 8,
) -> np.ndarray:
    """Map global expert ids to linear-placement EP ranks."""

    if num_experts % ep_degree != 0:
        raise ValueError("num_experts must be divisible by ep_degree")
    experts_per_rank = num_experts // ep_degree
    ranks = expert_ids // experts_per_rank
    if np.any((ranks < 0) | (ranks >= ep_degree)):
        raise ValueError("expert id is outside the configured EP rank range")
    return ranks


def count_expert_load(
    routed_experts: np.ndarray,
    vision_mask: np.ndarray,
    *,
    num_experts: int = 128,
    ep_degree: int = 8,
) -> EPLoadBreakdown:
    """Count top-k routed assignments by expert/rank and token modality.

    Args:
        routed_experts: Integer array with shape ``[tokens, layers, topk]``.
        vision_mask: Boolean array with shape ``[tokens]``. True marks image or
            video tokens; False marks text/control tokens.
        num_experts: Number of global MoE experts.
        ep_degree: Number of EP ranks.

    Returns:
        Per-expert and per-rank routed assignment counts. Counts include the
        top-k multiplicity, matching the MoE dispatch load.
    """

    routed = np.asarray(routed_experts)
    mask = np.asarray(vision_mask, dtype=bool)
    if routed.ndim != 3:
        raise ValueError("routed_experts must have shape [tokens, layers, topk]")
    if routed.shape[0] != mask.shape[0]:
        raise ValueError("vision_mask length must match routed token dimension")

    vision_ids = routed[mask].reshape(-1)
    text_ids = routed[~mask].reshape(-1)
    expert_vision = np.bincount(vision_ids, minlength=num_experts).astype(np.int64)
    expert_text = np.bincount(text_ids, minlength=num_experts).astype(np.int64)

    rank_vision = np.bincount(
        expert_to_rank(vision_ids, num_experts=num_experts, ep_degree=ep_degree),
        minlength=ep_degree,
    ).astype(np.int64)
    rank_text = np.bincount(
        expert_to_rank(text_ids, num_experts=num_experts, ep_degree=ep_degree),
        minlength=ep_degree,
    ).astype(np.int64)

    return EPLoadBreakdown(
        expert_vision=expert_vision[:num_experts],
        expert_text=expert_text[:num_experts],
        rank_vision=rank_vision[:ep_degree],
        rank_text=rank_text[:ep_degree],
    )


def load_imbalance(total_load: np.ndarray) -> dict[str, float | int]:
    """Return max/mean load imbalance metrics for a load vector."""

    total = np.asarray(total_load, dtype=np.float64)
    mean = float(total.mean()) if total.size else 0.0
    max_load = float(total.max()) if total.size else 0.0
    min_load = float(total.min()) if total.size else 0.0
    return {
        "mean_load": mean,
        "max_load": max_load,
        "min_load": min_load,
        "max_over_mean": float(max_load / mean) if mean else 0.0,
        "hot_index": int(total.argmax()) if total.size else -1,
    }

