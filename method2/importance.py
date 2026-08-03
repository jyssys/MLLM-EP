"""Cross-attention importance for vision tokens.

Only the raw attention variant is implemented in Phase 1. de-RoPE and CLS
terms are intentionally left as Phase 2 integration points.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from method2.derope import derope_attention


def _index_tensor(indices: torch.Tensor | list[int] | tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def compute_raw_cross_attention_importance(
    attention: torch.Tensor,
    text_indices: torch.Tensor | list[int] | tuple[int, ...],
    vision_indices: torch.Tensor | list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Compute ``a_j = mean_t A[text_t, vision_j]``.

    Supported attention shapes:
      - ``[seq, seq]``
      - ``[heads, seq, seq]``
      - ``[batch, heads, seq, seq]``

    Head and batch dimensions are averaged before the text-token mean.
    """

    if attention.ndim not in (2, 3, 4):
        raise ValueError("attention must have shape [S,S], [H,S,S], or [B,H,S,S]")
    text_idx = _index_tensor(text_indices, attention.device)
    vision_idx = _index_tensor(vision_indices, attention.device)
    if text_idx.numel() == 0 or vision_idx.numel() == 0:
        raise ValueError("text_indices and vision_indices must be non-empty")

    if attention.ndim == 2:
        block = attention.index_select(0, text_idx).index_select(1, vision_idx)
        return block.mean(dim=0)
    if attention.ndim == 3:
        block = attention.index_select(1, text_idx).index_select(2, vision_idx)
        return block.mean(dim=(0, 1))

    block = attention.index_select(2, text_idx).index_select(3, vision_idx)
    return block.mean(dim=(0, 1, 2))


def compute_cross_attention_importance(
    attention: torch.Tensor,
    text_indices: torch.Tensor | list[int] | tuple[int, ...],
    vision_indices: torch.Tensor | list[int] | tuple[int, ...],
    *,
    lambda_cls: float = 0.0,
    derope: bool = False,
    cls_attention: torch.Tensor | None = None,
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    position_ids_3d: torch.Tensor | None = None,
    mrope_section: Sequence[int] = (24, 20, 20),
    mrope_interleaved: bool = True,
) -> torch.Tensor:
    """General importance interface.

    Phase 1 supports only ``lambda_cls=0`` and ``derope=False``.
    """

    if derope:
        attention = derope_attention(
            q=q,
            k=k,
            cos=cos,
            sin=sin,
            position_ids_3d=position_ids_3d,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )
    if lambda_cls != 0.0 or cls_attention is not None:
        # TODO(Phase2): implement CLS attention term and tune lambda_cls.
        raise NotImplementedError("CLS-based importance is a Phase 2 item")
    return compute_raw_cross_attention_importance(attention, text_indices, vision_indices)


def split_key_redundant(
    importance: torch.Tensor,
    rho_key: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split vision tokens into top-rho key tokens and redundant remainder."""

    if not 0.0 < rho_key <= 1.0:
        raise ValueError("rho_key must be in (0, 1]")
    importance = importance.detach()
    if importance.ndim != 1:
        raise ValueError("importance must be a 1D tensor")
    n = importance.numel()
    k = max(1, math.ceil(n * rho_key))
    key_indices = torch.topk(importance, k=k, largest=True, sorted=True).indices
    key_mask = torch.zeros(n, dtype=torch.bool, device=importance.device)
    key_mask[key_indices] = True
    return key_mask, ~key_mask, key_indices
