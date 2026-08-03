"""Calibration statistics aggregation for dummy router outputs.

Phase 1 deliberately avoids real model forward passes. The functions here only
consume plain tensors that a later Qwen/DeepSpeed integration can produce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

TOKEN_TEXT = 0
TOKEN_VISION = 1
TOKEN_VISION_REDUNDANT = 1
TOKEN_VISION_KEY = 2


@dataclass(frozen=True)
class CalibrationConfig:
    """Small set of knobs for layer-wise calibration aggregation."""

    num_experts: int
    delta: float = 0.1
    dominant_ratio: float = 0.2
    attn_mode: str = "adaptive"


@dataclass(frozen=True)
class CalibrationHookSpec:
    """Phase 2 hook contract, modeled after MODE's frequency recorder."""

    target: str
    hook_type: str
    captures: tuple[str, ...]
    note: str


def _as_bool_mask(mask: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.zeros(n, dtype=torch.bool, device=device)
    if mask.shape != (n,):
        raise ValueError(f"mask must have shape ({n},), got {tuple(mask.shape)}")
    return mask.to(device=device, dtype=torch.bool)


def _count_assignments(
    expert_assignment: torch.Tensor,
    token_mask: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count top-k expert selections made by tokens in ``token_mask``."""

    if expert_assignment.ndim != 2:
        raise ValueError("expert_assignment must have shape [num_tokens, top_k]")
    selected = expert_assignment[token_mask]
    counts = torch.zeros(num_experts, dtype=torch.float32, device=expert_assignment.device)
    if selected.numel() == 0:
        return counts
    flat = selected.reshape(-1).to(torch.long)
    if flat.min() < 0 or flat.max() >= num_experts:
        raise ValueError("expert_assignment contains expert ids outside [0, num_experts)")
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return counts


def _normalize_counts(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum()
    if total <= 0:
        return torch.zeros_like(counts)
    return counts / total


def _classify(delta: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return 1 for vision-specialized, -1 for text-specialized, 0 for shared."""

    out = torch.zeros_like(delta, dtype=torch.int8)
    out[delta >= threshold] = 1
    out[delta <= -threshold] = -1
    return out


def _collapse_attention(attention: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Return a mean attention matrix with shape ``[query_tokens, n_tokens]``."""

    if attention.ndim == 2:
        matrix = attention.float()
    elif attention.ndim == 3:
        matrix = attention.float().mean(dim=0)
    elif attention.ndim == 4:
        matrix = attention.float().mean(dim=(0, 1))
    else:
        raise ValueError("attention must have shape [L,L], [H,L,L], or [B,H,L,L]")
    if matrix.shape[-1] != n_tokens:
        raise ValueError(
            f"attention key length must match token_type length {n_tokens}, got {matrix.shape[-1]}"
        )
    return matrix


def split_vision_tokens_by_attention(
    attention: torch.Tensor,
    token_type: torch.Tensor,
    dominant_ratio: float = 0.2,
    attn_mode: str = "adaptive",
    input_ids: torch.Tensor | None = None,
    special_token_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Split vision tokens into MODE-style key and redundant sets.

    This is a CPU/dummy implementation of the logic used by MODE's
    ``record_freq.py``. In ``vision`` and ``adaptive`` modes, image-token
    importance is the mean attention from post-image text tokens to each image
    token. In ``left`` mode, it is the mean attention from all non-self tokens
    to each image token.

    MODE computes a bottom-k mask too, but its MoE frequency hook treats
    redundant image tokens as all non-dominant image tokens. We mirror that
    routing-statistics behavior here.
    """

    if not 0 < dominant_ratio <= 1:
        raise ValueError("dominant_ratio must be in (0, 1]")
    if attn_mode not in {"vision", "adaptive", "left"}:
        raise ValueError("attn_mode must be one of: vision, adaptive, left")
    if token_type.ndim != 1:
        raise ValueError("token_type must have shape [num_tokens]")

    device = token_type.device
    n_tokens = token_type.numel()
    matrix = _collapse_attention(attention.to(device), n_tokens)
    vision_mask = token_type != TOKEN_TEXT
    image_pos = vision_mask.nonzero(as_tuple=True)[0]

    key_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    redundant_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    importance_full = torch.zeros(n_tokens, dtype=torch.float32, device=device)

    if image_pos.numel() == 0:
        return {
            "key_mask": key_mask.cpu(),
            "redundant_mask": redundant_mask.cpu(),
            "vision_importance": importance_full.cpu(),
        }

    if attn_mode in {"vision", "adaptive"}:
        query_mask = ~vision_mask
        query_mask[: int(image_pos[0].item()) + 1] = False
        if input_ids is not None and special_token_ids is not None and len(special_token_ids) > 0:
            if input_ids.shape != (n_tokens,):
                raise ValueError(f"input_ids must have shape ({n_tokens},), got {tuple(input_ids.shape)}")
            ids = input_ids.to(device)
            special = torch.as_tensor(special_token_ids, dtype=ids.dtype, device=device)
            query_mask &= ~torch.isin(ids, special)
        query_pos = query_mask.nonzero(as_tuple=True)[0]
        if query_pos.numel() == 0:
            importance = torch.zeros(image_pos.numel(), dtype=torch.float32, device=device)
        else:
            importance = matrix[query_pos][:, image_pos].mean(dim=0)
    else:
        all_to_image = matrix[:, image_pos].clone()
        for column, pos in enumerate(image_pos):
            pos_idx = int(pos.item())
            if pos_idx < all_to_image.shape[0]:
                all_to_image[pos_idx, column] = 0.0
        importance = all_to_image.mean(dim=0)

    num_key = max(1, int(image_pos.numel() * dominant_ratio))
    top_idx = torch.topk(importance, num_key).indices
    key_mask[image_pos[top_idx]] = True
    redundant_mask = vision_mask & ~key_mask
    importance_full[image_pos] = importance

    return {
        "key_mask": key_mask.cpu(),
        "redundant_mask": redundant_mask.cpu(),
        "vision_importance": importance_full.cpu(),
    }


def collect_layer_stats(
    expert_assignment: torch.Tensor,
    token_type: torch.Tensor,
    hidden_states: torch.Tensor,
    num_experts: int,
    delta: float = 0.1,
    key_mask: torch.Tensor | None = None,
    redundant_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate Phase 1 calibration stats for one layer.

    Args:
        expert_assignment: Top-k expert ids per token, shape ``[N, K]``.
        token_type: Token modality labels, shape ``[N]``. ``0`` is text,
            ``1`` is redundant/general vision, and ``2`` is key vision.
        hidden_states: Hidden vectors, shape ``[N, D]``.
        num_experts: Number of experts in the layer.
        delta: Specialization threshold for ``f_vis - f_txt``.
        key_mask: Optional explicit key-vision mask. If omitted, token_type==2.
        redundant_mask: Optional explicit redundant-vision mask. If omitted,
            token_type==1.

    Returns:
        A plain dictionary of tensors. Counts reflect top-k selections, so one
        token can increment multiple experts.
    """

    if expert_assignment.ndim != 2:
        raise ValueError("expert_assignment must have shape [num_tokens, top_k]")
    n_tokens = expert_assignment.shape[0]
    if token_type.shape != (n_tokens,):
        raise ValueError(f"token_type must have shape ({n_tokens},), got {tuple(token_type.shape)}")
    if hidden_states.ndim != 2 or hidden_states.shape[0] != n_tokens:
        raise ValueError("hidden_states must have shape [num_tokens, hidden_dim]")

    device = expert_assignment.device
    token_type = token_type.to(device)
    hidden_states = hidden_states.to(device)

    text_mask = token_type == TOKEN_TEXT
    vision_mask = token_type != TOKEN_TEXT
    key_mask_t = _as_bool_mask(key_mask, n_tokens, device)
    redundant_mask_t = _as_bool_mask(redundant_mask, n_tokens, device)
    if key_mask is None:
        key_mask_t = token_type == TOKEN_VISION_KEY
    if redundant_mask is None:
        redundant_mask_t = token_type == TOKEN_VISION_REDUNDANT

    n_vis = _count_assignments(expert_assignment, vision_mask, num_experts)
    n_txt = _count_assignments(expert_assignment, text_mask, num_experts)
    n_total = _count_assignments(expert_assignment, torch.ones(n_tokens, dtype=torch.bool, device=device), num_experts)
    f_vis = _normalize_counts(n_vis)
    f_txt = _normalize_counts(n_txt)
    f_total = _normalize_counts(n_total)
    specialization_delta = f_vis - f_txt

    n_key = _count_assignments(expert_assignment, key_mask_t & vision_mask, num_experts)
    n_red = _count_assignments(expert_assignment, redundant_mask_t & vision_mask, num_experts)
    f_key = _normalize_counts(n_key)
    f_red = _normalize_counts(n_red)

    # TODO(Phase2): use collected centroids only if redundant-token rerouting
    # is reintroduced. Phase 1 collects them as insurance and never reroutes.
    hidden_dim = hidden_states.shape[-1]
    centroid_sum = torch.zeros(num_experts, hidden_dim, dtype=hidden_states.dtype, device=device)
    centroid_count = torch.zeros(num_experts, dtype=hidden_states.dtype, device=device)
    flat_experts = expert_assignment.reshape(-1).to(torch.long)
    repeated_hidden = hidden_states[:, None, :].expand(-1, expert_assignment.shape[1], -1).reshape(-1, hidden_dim)
    centroid_sum.index_add_(0, flat_experts, repeated_hidden)
    centroid_count.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=hidden_states.dtype))
    centroid = centroid_sum / centroid_count.clamp_min(1).unsqueeze(-1)

    return {
        "N_total": n_total.cpu(),
        "N_vis": n_vis.cpu(),
        "N_image": n_vis.cpu(),
        "N_txt": n_txt.cpu(),
        "N_text": n_txt.cpu(),
        "f_total": f_total.cpu(),
        "f_vis": f_vis.cpu(),
        "f_image": f_vis.cpu(),
        "f_txt": f_txt.cpu(),
        "f_text": f_txt.cpu(),
        "delta": specialization_delta.cpu(),
        "specialization": _classify(specialization_delta, delta).cpu(),
        "N_key": n_key.cpu(),
        "N_dominant_image": n_key.cpu(),
        "N_red": n_red.cpu(),
        "N_redundant_image": n_red.cpu(),
        "f_key": f_key.cpu(),
        "f_dominant_image": f_key.cpu(),
        "f_red": f_red.cpu(),
        "f_redundant_image": f_red.cpu(),
        "centroid": centroid.cpu(),
        "centroid_count": centroid_count.cpu(),
    }


def collect_layer_stats_from_attention(
    expert_assignment: torch.Tensor,
    token_type: torch.Tensor,
    hidden_states: torch.Tensor,
    num_experts: int,
    attention: torch.Tensor,
    delta: float = 0.1,
    dominant_ratio: float = 0.2,
    attn_mode: str = "adaptive",
    input_ids: torch.Tensor | None = None,
    special_token_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate one layer using MODE-style attention-derived key masks."""

    split = split_vision_tokens_by_attention(
        attention=attention,
        token_type=token_type,
        dominant_ratio=dominant_ratio,
        attn_mode=attn_mode,
        input_ids=input_ids,
        special_token_ids=special_token_ids,
    )
    stats = collect_layer_stats(
        expert_assignment=expert_assignment,
        token_type=token_type,
        hidden_states=hidden_states,
        num_experts=num_experts,
        delta=delta,
        key_mask=split["key_mask"],
        redundant_mask=split["redundant_mask"],
    )
    stats["vision_importance"] = split["vision_importance"]
    return stats


def collect_calibration_stats(
    layers: Mapping[int, Mapping[str, Any]],
    num_experts: int,
    delta: float = 0.1,
) -> dict[int, dict[str, torch.Tensor]]:
    """Aggregate stats for several layers from plain tensor dictionaries."""

    stats: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx, payload in layers.items():
        if "attention" in payload:
            stats[layer_idx] = collect_layer_stats_from_attention(
                expert_assignment=payload["expert_assignment"],
                token_type=payload["token_type"],
                hidden_states=payload["hidden_states"],
                num_experts=num_experts,
                attention=payload["attention"],
                delta=delta,
                dominant_ratio=float(payload.get("dominant_ratio", 0.2)),
                attn_mode=str(payload.get("attn_mode", "adaptive")),
                input_ids=payload.get("input_ids"),
                special_token_ids=payload.get("special_token_ids"),
            )
        else:
            stats[layer_idx] = collect_layer_stats(
                expert_assignment=payload["expert_assignment"],
                token_type=payload["token_type"],
                hidden_states=payload["hidden_states"],
                num_experts=num_experts,
                delta=delta,
                key_mask=payload.get("key_mask"),
                redundant_mask=payload.get("redundant_mask"),
            )
    return stats


def load_sharegpt4v_records(calib_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load ShareGPT4V/COCO-style calibration records from JSON or JSONL."""

    path = Path(calib_path)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records = payload.get("data", payload.get("annotations", []))
        else:
            records = payload
    if not isinstance(records, list):
        raise ValueError("calibration file must contain a JSON list or a dict with data/annotations")
    out: list[dict[str, Any]] = []
    for record in records[:limit]:
        if not isinstance(record, dict):
            raise ValueError("each calibration record must be a JSON object")
        if "image" not in record:
            raise ValueError("ShareGPT4V calibration records must include an image field")
        out.append(record)
    return out


def resolve_calibration_image_path(record: Mapping[str, Any], image_root: str | Path) -> Path:
    """Resolve MODE-style relative image paths against a calibration root."""

    image = record.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError("record['image'] must be a non-empty string")
    path = Path(image)
    if path.is_absolute():
        return path
    return Path(image_root) / path


def build_mode_hook_plan(
    model_type: str = "qwen3_vl_moe",
    attn_mode: str = "adaptive",
    dominant_ratio: float = 0.2,
    capture_centroid: bool = True,
) -> tuple[CalibrationHookSpec, ...]:
    """Describe the future real-model hook structure without installing it.

    This intentionally returns metadata only. Real Qwen forward hooks require
    GPU/model execution and remain a Phase 2 integration task.
    """

    if model_type != "qwen3_vl_moe":
        raise NotImplementedError("Phase 1 only documents hook planning for qwen3_vl_moe")
    if attn_mode not in {"vision", "adaptive", "left"}:
        raise ValueError("attn_mode must be one of: vision, adaptive, left")

    centroid_note = "also accumulate hidden-state sums/counts per routed expert" if capture_centroid else "frequency only"
    return (
        CalibrationHookSpec(
            target="model/generate entry",
            hook_type="input pre-hook or generate wrapper",
            captures=("input_ids", "image_token mask", "special token ids"),
            note="Capture token modality masks before decoder layers run.",
        ),
        CalibrationHookSpec(
            target="decoder_layers[*].self_attn",
            hook_type="pre/post forward hooks",
            captures=("attention probabilities",),
            note=f"Compute layer-local dominant image mask with attn_mode={attn_mode}, dominant_ratio={dominant_ratio}.",
        ),
        CalibrationHookSpec(
            target="decoder_layers[*].mlp / Qwen3VLMoeTextSparseMoeBlock",
            hook_type="forward hook",
            captures=("router logits or gate(hidden_states)", "top-k expert ids", "hidden_states"),
            note=f"Aggregate modality/key/redundant expert frequencies; {centroid_note}.",
        ),
    )


def save_layer_stats(stats: Mapping[str, torch.Tensor], output_path: str | Path) -> None:
    """Persist one layer of stats as a torch file.

    The storage format is intentionally simple for Phase 1. Phase 2 can wrap
    this with model-specific metadata once real calibration runs exist.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(stats), output_path)
