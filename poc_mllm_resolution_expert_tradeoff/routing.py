from __future__ import annotations

import torch


def apply_visual_k(
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
    visual_mask: torch.Tensor,
    k_visual: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ragged visual-only k while preserving the fixed-width API.

    Unused visual slots use the installed vLLM/DeepEP invalid-route contract:
    expert id ``-1`` and weight zero.  Text rows are returned bit-identically.
    """
    if k_visual == weights.shape[-1]:
        return weights, expert_ids
    if k_visual < 1 or k_visual > weights.shape[-1]:
        raise ValueError((k_visual, weights.shape))
    if visual_mask.ndim != 1 or visual_mask.shape[0] != weights.shape[0]:
        raise ValueError((visual_mask.shape, weights.shape))

    kept_weights, kept_positions = torch.topk(
        weights, k=k_visual, dim=-1, largest=True, sorted=True
    )
    kept_ids = torch.gather(expert_ids, -1, kept_positions)
    kept_weights = kept_weights / kept_weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(kept_weights.dtype).tiny
    )

    candidate_weights = torch.zeros_like(weights)
    candidate_ids = torch.full_like(expert_ids, -1)
    candidate_weights[..., :k_visual] = kept_weights
    candidate_ids[..., :k_visual] = kept_ids
    row_mask = visual_mask.to(device=weights.device, dtype=torch.bool).unsqueeze(-1)
    return (
        torch.where(row_mask, candidate_weights, weights),
        torch.where(row_mask, candidate_ids, expert_ids),
    )

