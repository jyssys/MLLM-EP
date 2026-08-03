"""Runtime patch for layer-wise vLLM MoE expert placement.

The patch is intentionally narrow: it replaces each ``FusedMoE`` layer's
``_expert_map`` buffer from a JSON expert->rank table. Routing, fused kernels,
weights, precision, and all-to-all implementations are left unchanged.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


_PATCHED_ATTR = "_mllm_layerwise_placement_patched"


def layer_id_from_prefix(prefix: str) -> int | None:
    """Extract ``N`` from vLLM layer names containing ``layers.N``."""

    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else None


@lru_cache(maxsize=4)
def load_layerwise_rank_map(path: str) -> dict[int, dict[int, int]]:
    """Load ``{layer: {expert: rank}}`` JSON into integer-keyed dictionaries."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("custom expert map JSON must be an object")

    parsed: dict[int, dict[int, int]] = {}
    for layer_key, layer_value in raw.items():
        if not isinstance(layer_value, dict):
            raise ValueError(f"layer {layer_key!r} must map experts to ranks")
        parsed[int(layer_key)] = {
            int(expert_key): int(rank) for expert_key, rank in layer_value.items()
        }
    return parsed


def make_expert_map_for_rank(
    layer_mapping: dict[int, int],
    *,
    ep_rank: int,
    ep_size: int,
    global_num_experts: int,
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
    device: torch.device | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor | None, list[int]]:
    """Convert one layer's expert->rank map to vLLM's global->local map."""

    expected_experts = set(range(global_num_experts))
    observed_experts = set(layer_mapping)
    missing = sorted(expected_experts - observed_experts)
    extra = sorted(observed_experts - expected_experts)
    if missing or extra:
        raise ValueError(
            "custom expert map must cover exactly the global experts; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    counts = [0 for _ in range(ep_size)]
    for expert_id in range(global_num_experts):
        rank = layer_mapping[expert_id]
        if rank < 0 or rank >= ep_size:
            raise ValueError(f"rank {rank} for expert {expert_id} is outside EP range")
        counts[rank] += 1

    if global_num_experts % ep_size != 0:
        raise ValueError("custom placement requires an even expert/rank split")
    expected_per_rank = global_num_experts // ep_size
    if any(count != expected_per_rank for count in counts):
        raise ValueError(
            f"each rank must own {expected_per_rank} experts; got counts={counts}"
        )

    local_global_experts = [
        expert_id
        for expert_id in range(global_num_experts)
        if layer_mapping[expert_id] == ep_rank
    ]
    local_num_experts = len(local_global_experts)
    device_kwargs: dict[str, Any] = {}
    if device is not None:
        device_kwargs["device"] = device

    expert_map = torch.full(
        (global_num_experts,), -1, dtype=torch.int32, **device_kwargs
    )
    if local_global_experts:
        expert_map[
            torch.tensor(local_global_experts, dtype=torch.long, **device_kwargs)
        ] = torch.arange(local_num_experts, dtype=torch.int32, **device_kwargs)

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,),
            dtype=torch.int32,
            **device_kwargs,
        )
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [
                        local_num_experts + idx
                        for idx in range(num_fused_shared_experts)
                    ],
                    dtype=torch.int32,
                    **device_kwargs,
                ),
            ),
            dim=0,
        )

    return local_num_experts, expert_map, expert_mask, local_global_experts


def _append_audit(record: dict[str, Any]) -> None:
    audit_path = os.environ.get("VLLM_MOE_EXPERT_MAP_AUDIT_JSONL")
    if not audit_path:
        return
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def apply_vllm_custom_placement_patch() -> bool:
    """Patch vLLM ``FusedMoE`` so every layer can use its own expert map.

    The patch activates only when ``VLLM_MOE_EXPERT_MAP_JSON`` is set. It is safe
    to call multiple times. Returns True when the class was patched.
    """

    map_path = os.environ.get("VLLM_MOE_EXPERT_MAP_JSON")
    if not map_path:
        return False

    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    fused_moe_cls = fused_layer.FusedMoE
    if getattr(fused_moe_cls, _PATCHED_ATTR, False):
        return True

    original_init = fused_moe_cls.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)

        if os.environ.get("VLLM_MOE_TIMING_JSONL"):
            try:
                from vllm_moe_timing import apply_vllm_moe_timing_patch

                apply_vllm_moe_timing_patch()
            except Exception as exc:
                _append_audit(
                    {
                        "timing_patch_error": repr(exc),
                        "prefix": getattr(self, "layer_name", ""),
                    }
                )

        if not getattr(self, "use_ep", False):
            return

        layer_id = layer_id_from_prefix(getattr(self, "layer_name", ""))
        if layer_id is None:
            return

        layer_maps = load_layerwise_rank_map(os.environ["VLLM_MOE_EXPERT_MAP_JSON"])
        if layer_id not in layer_maps:
            raise ValueError(
                f"custom expert map has no entry for vLLM layer {layer_id}"
            )

        current_map = getattr(self, "_expert_map", None)
        device = current_map.device if current_map is not None else None
        local_num_experts, expert_map, expert_mask, local_global = (
            make_expert_map_for_rank(
                layer_maps[layer_id],
                ep_rank=int(self.ep_rank),
                ep_size=int(self.ep_size),
                global_num_experts=int(self.global_num_experts),
                num_fused_shared_experts=int(self.num_fused_shared_experts),
                return_expert_mask=bool(self.rocm_aiter_fmoe_enabled),
                device=device,
            )
        )

        if local_num_experts != int(self.local_num_experts):
            raise ValueError(
                "custom map changed the local expert count from "
                f"{self.local_num_experts} to {local_num_experts}; this patch "
                "supports balanced maps only"
            )

        self.register_buffer("_expert_map", expert_map)
        self.register_buffer("expert_mask", expert_mask)

        quant_method = getattr(self, "quant_method", None)
        _append_audit(
            {
                "layer": layer_id,
                "prefix": getattr(self, "layer_name", ""),
                "ep_rank": int(self.ep_rank),
                "ep_size": int(self.ep_size),
                "global_num_experts": int(self.global_num_experts),
                "local_num_experts": int(local_num_experts),
                "local_global_experts": local_global,
                "map_path": os.environ["VLLM_MOE_EXPERT_MAP_JSON"],
                "quant_method": type(quant_method).__name__ if quant_method else None,
                "disable_expert_map": getattr(quant_method, "disable_expert_map", None),
            }
        )

    fused_moe_cls.__init__ = patched_init
    setattr(fused_moe_cls, _PATCHED_ATTR, True)
    return True
