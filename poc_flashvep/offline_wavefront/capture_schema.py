"""Compact layer-24 capture schema for the offline wavefront PoC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaptureMetadata:
    schema_version: int
    model_path: str
    layer: int
    dtype: str
    original_token_count: int
    vision_token_count: int
    hidden_size: int
    expert_intermediate_size: int
    top_k: int
    global_num_experts: int
    ep_size: int
    local_experts_per_rank: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_TENSORS = (
    "post_attention_hidden",
    "topk_expert_ids",
    "topk_weights",
    "destination_rank",
    "local_expert_id",
)


def validate_capture(capture: dict[str, Any]) -> None:
    """Validate shape, route ownership, and compact-capture invariants."""
    metadata = capture.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("capture metadata is missing")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported capture schema")
    for name in REQUIRED_TENSORS:
        if not isinstance(capture.get(name), torch.Tensor):
            raise ValueError(f"capture tensor is missing: {name}")

    hidden = capture["post_attention_hidden"]
    ids = capture["topk_expert_ids"]
    weights = capture["topk_weights"]
    destination = capture["destination_rank"]
    local_id = capture["local_expert_id"]
    tokens = int(metadata["original_token_count"])
    top_k = int(metadata["top_k"])
    local_experts = int(metadata["local_experts_per_rank"])
    ep_size = int(metadata["ep_size"])

    if hidden.ndim != 2 or tuple(hidden.shape) != (
        tokens,
        int(metadata["hidden_size"]),
    ):
        raise ValueError(f"invalid hidden shape: {tuple(hidden.shape)}")
    expected_route_shape = (tokens, top_k)
    if tuple(ids.shape) != expected_route_shape:
        raise ValueError(f"invalid route shape: {tuple(ids.shape)}")
    for name, tensor in (
        ("topk_weights", weights),
        ("destination_rank", destination),
        ("local_expert_id", local_id),
    ):
        if tuple(tensor.shape) != expected_route_shape:
            raise ValueError(f"invalid {name} shape: {tuple(tensor.shape)}")
    if hidden.dtype != torch.bfloat16 or weights.dtype != torch.float32:
        raise ValueError("capture must contain BF16 hidden and FP32 routing weights")
    if ids.min().item() < 0 or ids.max().item() >= ep_size * local_experts:
        raise ValueError("global expert id is outside the EP4 expert range")
    if not torch.equal(destination, torch.div(ids, local_experts, rounding_mode="floor")):
        raise ValueError("destination_rank does not match global expert ids")
    if not torch.equal(local_id, torch.remainder(ids, local_experts)):
        raise ValueError("local_expert_id does not match global expert ids")
