"""Replay exact visual-prefix and post-image-tail routes through DeepEP/Triton."""

from __future__ import annotations

import json
import os
import re
import statistics
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poc_flashvep.deepep_revalidation.operator_replay import ExpertSpec, _correctness
from poc_flashvep.tile_slack_mechanism.operator_replay import _run_variant

_INSTALLED = False
_RAN_RANKS: set[int] = set()
_CONTEXT = threading.local()
IMAGE_TOKEN_ID = 151655


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _current_layer() -> int:
    return int(getattr(_CONTEXT, "layer", -1))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _samples(result: Path, device: torch.device) -> list[dict[str, Any]]:
    manifest = json.loads((result / "workload_manifest.json").read_text())
    limit = int(os.environ.get("FLASHVEP_CAUSAL_MAX_REQUESTS", "24"))
    rows = []
    for pair in manifest["pairs"][:limit]:
        item = pair["vision"]
        with np.load(result / item["route_file"]) as archive:
            routes = archive["routed_experts"].astype(np.int64)
            token_ids = archive["prompt_token_ids"].astype(np.int64)
        image_positions = np.flatnonzero(token_ids == IMAGE_TOKEN_ID)
        if not len(image_positions):
            raise AssertionError((item["request_id"], "missing visual tokens"))
        # The structural vision-end token immediately after the repeated image
        # tokens remains in the prefix. Everything later is post-image language.
        prefix_end = min(len(token_ids), int(image_positions[-1]) + 2)
        if prefix_end >= len(token_ids):
            raise AssertionError((item["request_id"], "empty language tail"))
        rows.append(
            {
                "request_id": item["request_id"],
                "category": item["category"],
                "pair_id": int(pair["pair_id"]),
                "token_bucket": pair["token_bucket"],
                "routes": torch.from_numpy(routes).to(device),
                "prompt_tokens": len(token_ids),
                "vision_tokens": int(len(image_positions)),
                "prefix_tokens": prefix_end,
                "tail_tokens": len(token_ids) - prefix_end,
                "prefix_indices": list(range(prefix_end)),
                "tail_indices": list(range(prefix_end, len(token_ids))),
            }
        )
    return rows


def _timing_rows(
    kernel: Any,
    original_experts: Any,
    spec: ExpertSpec,
    capture: dict[str, Any],
    rank: int,
) -> list[dict[str, Any]]:
    result = Path(os.environ["FLASHVEP_CAUSAL_SOURCE_DIR"])
    warmups = int(os.environ.get("FLASHVEP_CAUSAL_WARMUPS", "2"))
    iterations = int(os.environ.get("FLASHVEP_CAUSAL_ITERATIONS", "7"))
    layers = [
        int(value)
        for value in os.environ.get(
            "FLASHVEP_CAUSAL_LAYERS", ",".join(map(str, range(48)))
        ).split(",")
    ]
    rows = []
    for sample in _samples(result, spec.w1.device):
        for layer in layers:
            routes = sample["routes"][:, layer, :]
            for component, indices in (
                ("vision_prefix", sample["prefix_indices"]),
                ("language_tail", sample["tail_indices"]),
            ):
                local_routes = routes[indices].contiguous()
                timing, _ = _run_variant(
                    "serial",
                    [list(range(len(local_routes)))],
                    local_routes,
                    capture,
                    kernel,
                    original_experts,
                    kernel.prepare_finalize.buffer,
                    spec,
                    rank,
                    warmups,
                    iterations,
                )
                rows.append(
                    {
                        **{
                            key: value
                            for key, value in sample.items()
                            if key not in ("routes", "prefix_indices", "tail_indices")
                        },
                        "layer": layer,
                        "rank": rank,
                        "component": component,
                        "route_tokens": len(local_routes),
                        "topk": 8,
                        "dispatch_ms": timing["dispatch_ms"],
                        "expert_ms": timing["expert_ms"],
                        "combine_ms": timing["combine_ms"],
                        "wall_ms": timing["wall_ms"],
                        "dispatch_median_ms": timing["dispatch_ms_stats"]["median_ms"],
                        "expert_median_ms": timing["expert_ms_stats"]["median_ms"],
                        "combine_median_ms": timing["combine_ms_stats"]["median_ms"],
                        "wall_median_ms": timing["wall_ms_stats"]["median_ms"],
                        "route_identity": True,
                    }
                )
    return rows


def _diagnostic(
    kernel: Any,
    original_experts: Any,
    spec: ExpertSpec,
    capture: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    result = Path(os.environ["FLASHVEP_CAUSAL_SOURCE_DIR"])
    sample = _samples(result, spec.w1.device)[8]  # fixed medium request
    layer = 24
    routes = sample["routes"][:, layer, :]
    groups = [sample["prefix_indices"], sample["tail_indices"]]
    warmups, iterations = 5, 20
    serial, reference = _run_variant(
        "serial",
        groups,
        routes,
        capture,
        kernel,
        original_experts,
        kernel.prepare_finalize.buffer,
        spec,
        rank,
        warmups,
        iterations,
    )
    overlap, output = _run_variant(
        "overlap",
        groups,
        routes,
        capture,
        kernel,
        original_experts,
        kernel.prepare_finalize.buffer,
        spec,
        rank,
        warmups,
        iterations,
    )
    return {
        "rank": rank,
        "request_id": sample["request_id"],
        "layer": layer,
        "selection": "preregistered pair index 8, layer 24",
        "warmups": warmups,
        "iterations": iterations,
        "serial": serial,
        "overlap": overlap,
        "speedup": (
            serial["wall_ms_stats"]["median_ms"] / overlap["wall_ms_stats"]["median_ms"]
        ),
        "correctness": _correctness(reference, output),
    }


def _run(kernel: Any, original_experts: Any, spec: ExpertSpec) -> dict[str, Any]:
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    rank = int(ep.rank_in_group)
    if int(ep.world_size) != 4:
        raise AssertionError(ep.world_size)
    if type(kernel.prepare_finalize).__name__ != "DeepEPHTPrepareAndFinalize":
        raise AssertionError(type(kernel.prepare_finalize).__name__)
    capture = torch.load(
        os.environ["FLASHVEP_CAUSAL_CAPTURE_PATH"],
        map_location="cpu",
        weights_only=False,
    )
    mode = os.environ.get("FLASHVEP_CAUSAL_MODE", "timing")
    payload: dict[str, Any] = {
        "status": "ok",
        "rank": rank,
        "physical_gpu": [1, 2, 3, 4][rank],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expert_backend": type(kernel.fused_experts).__name__,
        "prepare_finalize_backend": type(kernel.prepare_finalize).__name__,
        "hidden_provenance": "validated real Qwen3-VL layer-24 BF16 capture",
        "input_replication": "same exact request routes on four EP sources",
        "mode": mode,
    }
    if mode == "timing":
        payload["observations"] = _timing_rows(
            kernel, original_experts, spec, capture, rank
        )
    elif mode == "diagnostic":
        payload["diagnostic"] = _diagnostic(
            kernel, original_experts, spec, capture, rank
        )
    else:
        raise ValueError(mode)
    return payload


def _write(rank: int, payload: dict[str, Any]) -> None:
    output = Path(os.environ["FLASHVEP_CAUSAL_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank{rank}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_causal_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = _current_layer()
        _CONTEXT.layer = int(getattr(self, "_flashvep_causal_layer", -1))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        rank = int(get_ep_group().rank_in_group)
        if (
            rank not in _RAN_RANKS
            and _current_layer() == 24
            and type(self.prepare_finalize).__name__ == "DeepEPHTPrepareAndFinalize"
        ):
            _RAN_RANKS.add(rank)
            names = (
                "in_dtype",
                "a1q",
                "a1q_scale",
                "w1",
                "w2",
                "topk_weights",
                "topk_ids",
                "activation",
                "global_num_experts",
                "local_num_experts",
                "expert_map",
                "apply_router_weight_on_input",
                "expert_tokens_meta",
            )
            values = dict(zip(names, args, strict=False))
            values.update(kwargs)
            spec = ExpertSpec(
                in_dtype=values["in_dtype"],
                w1=values["w1"],
                w2=values["w2"],
                activation=values["activation"],
                global_num_experts=int(values["global_num_experts"]),
                local_num_experts=int(values["local_num_experts"]),
                expert_map=values["expert_map"],
                apply_router_weight_on_input=bool(
                    values["apply_router_weight_on_input"]
                ),
            )
            try:
                _write(rank, _run(self, original_experts, spec))
            except BaseException as error:
                _write(
                    rank,
                    {
                        "status": "error",
                        "rank": rank,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
        return original_experts(self, *args, **kwargs)

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
