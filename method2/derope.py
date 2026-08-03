"""Phase 2 de-RoPE importance interface for Qwen3-VL M-RoPE."""

from __future__ import annotations

from typing import Sequence

import torch


def derope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids_3d: torch.Tensor,
    *,
    mrope_section: Sequence[int] = (24, 20, 20),
    mrope_interleaved: bool = True,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return attention scores after inverting Qwen3-VL interleaved M-RoPE.

    Expected Phase 2 inputs:

    - ``q`` and ``k``: query/key states after rotary application, typically
      ``[batch, heads, seq, head_dim]``
    - ``cos`` and ``sin``: rotary caches produced for the same positions
    - ``position_ids_3d``: Qwen3-VL multimodal position ids with shape
      ``[4, batch, seq]`` where row 0 is text/cache position and rows 1-3 are
      temporal/height/width ids
    - ``mrope_section``: interleaved temporal/height/width section sizes, e.g.
      ``[24, 20, 20]``
    - ``mrope_interleaved``: whether to follow Qwen3-VL's
      ``apply_interleaved_mrope`` layout

    A plain 1D RoPE inverse is incorrect for image/video tokens because
    Qwen3-VL applies 3D interleaved M-RoPE across temporal, height, and width
    positions.
    """

    raise NotImplementedError("TODO(Phase2): invert interleaved M-RoPE on Q/K")

