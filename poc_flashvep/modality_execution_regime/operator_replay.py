"""Replay captured Vision/Text routes through actual DeepEP and Triton experts."""

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

from poc_flashvep.deepep_revalidation.operator_replay import ExpertSpec
from poc_flashvep.tile_slack_mechanism.operator_replay import _run_variant


_INSTALLED = False
_RAN_RANKS: set[int] = set()
_CONTEXT = threading.local()


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _current_layer() -> int:
    return int(getattr(_CONTEXT, "layer", -1))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _samples(result: Path, device: torch.device) -> list[dict[str, Any]]:
    manifest = json.loads((result / "workload_manifest.json").read_text())
    rows = []
    for pair in manifest["pairs"]:
        for modality in ("vision", "text"):
            item = pair[modality]
            with np.load(result / item["route_file"]) as archive:
                routes = torch.from_numpy(archive["routed_experts"].astype(np.int64))
            if routes.ndim != 3 or routes.shape[1:] != (48, 8):
                raise AssertionError((item["request_id"], routes.shape))
            rows.append({
                "request_id": item["request_id"], "modality": modality,
                "category": item["category"], "pair_id": pair["pair_id"],
                "token_bucket": pair["token_bucket"],
                "prompt_tokens": int(item["prompt_tokens"]),
                "routes": routes.to(device),
            })
    return rows


def _shape(routes: torch.Tensor, rank: int, replication: int = 4) -> dict[str, Any]:
    ids = routes.detach().to("cpu").numpy().astype(np.int64)
    low, high = rank * 32, (rank + 1) * 32
    local = ids[(ids >= low) & (ids < high)] - low
    histogram = np.bincount(local, minlength=32).astype(np.int64) * replication
    destinations = ((ids >= low) & (ids < high)).any(axis=1)
    dispatched_rows = int(destinations.sum()) * replication
    return {
        "expert_histogram": histogram.tolist(),
        "total_assignments": int(histogram.sum()),
        "dispatched_rows": dispatched_rows,
    }


def _runtime_config(kernel: Any, spec: ExpertSpec, dispatched_rows: int) -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        try_get_optimal_moe_config,
    )

    experts = kernel.fused_experts
    dtype_name = experts.quant_config.config_name(spec.w1.dtype)
    config = try_get_optimal_moe_config(
        spec.w1.size(), spec.w2.size(), 8, dtype_name, dispatched_rows,
        block_shape=experts.block_shape,
    )
    return {key: int(value) for key, value in config.items() if isinstance(value, (int, bool))}


def _run(kernel: Any, original_experts: Any, spec: ExpertSpec) -> dict[str, Any]:
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    rank = int(ep.rank_in_group)
    if int(ep.world_size) != 4:
        raise AssertionError(f"expected EP4, got {ep.world_size}")
    if type(kernel.prepare_finalize).__name__ != "DeepEPHTPrepareAndFinalize":
        raise AssertionError(type(kernel.prepare_finalize).__name__)
    result = Path(os.environ["FLASHVEP_MODALITY_RESULT_DIR"])
    capture = torch.load(
        os.environ["FLASHVEP_MODALITY_CAPTURE_PATH"], map_location="cpu",
        weights_only=False,
    )
    warmups = int(os.environ.get("FLASHVEP_MODALITY_WARMUPS", "3"))
    iterations = int(os.environ.get("FLASHVEP_MODALITY_ITERATIONS", "15"))
    rows = []
    for sample in _samples(result, spec.w1.device):
        for layer in range(48):
            routes = sample["routes"][:, layer, :]
            shape = _shape(routes, rank)
            config = _runtime_config(kernel, spec, shape["dispatched_rows"])
            timing, _ = _run_variant(
                "serial", [list(range(len(routes)))], routes, capture, kernel,
                original_experts, kernel.prepare_finalize.buffer, spec, rank,
                warmups, iterations,
            )
            rows.append({
                **{key: value for key, value in sample.items() if key != "routes"},
                "layer": layer, "rank": rank, **shape,
                "runtime_m": shape["dispatched_rows"],
                "runtime_config": config,
                "expert_ms": timing["expert_ms"],
                "expert_median_ms": _median(timing["expert_ms"]),
                "dispatch_median_ms": timing["dispatch_ms_stats"]["median_ms"],
                "combine_median_ms": timing["combine_ms_stats"]["median_ms"],
                "route_identity": True,
            })
    return {
        "status": "ok", "rank": rank, "physical_gpu": [4, 5, 6, 7][rank],
        "settings": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "expert_backend": type(kernel.fused_experts).__name__,
            "prepare_finalize_backend": type(kernel.prepare_finalize).__name__,
            "communication_backend": "DeepEP high-throughput",
            "input_replication": 4,
            "hidden_provenance": "validated real Qwen3-VL layer-24 BF16 capture",
            "warmups": warmups, "iterations": iterations,
            "runtime_config_source": "TritonExperts.try_get_optimal_moe_config",
        },
        "observations": rows,
    }


def _write(rank: int, value: dict[str, Any]) -> None:
    directory = Path(os.environ["FLASHVEP_MODALITY_REPLAY_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{rank}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_modality_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = _current_layer()
        _CONTEXT.layer = int(getattr(self, "_flashvep_modality_layer", -1))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        rank = int(get_ep_group().rank_in_group)
        if (
            rank not in _RAN_RANKS and _current_layer() == 24
            and type(self.prepare_finalize).__name__ == "DeepEPHTPrepareAndFinalize"
        ):
            _RAN_RANKS.add(rank)
            names = (
                "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                "topk_ids", "activation", "global_num_experts",
                "local_num_experts", "expert_map", "apply_router_weight_on_input",
                "expert_tokens_meta",
            )
            values = dict(zip(names, args, strict=False)); values.update(kwargs)
            spec = ExpertSpec(
                in_dtype=values["in_dtype"], w1=values["w1"], w2=values["w2"],
                activation=values["activation"],
                global_num_experts=int(values["global_num_experts"]),
                local_num_experts=int(values["local_num_experts"]),
                expert_map=values["expert_map"],
                apply_router_weight_on_input=bool(values["apply_router_weight_on_input"]),
            )
            try:
                _write(rank, _run(self, original_experts, spec))
            except BaseException as error:
                _write(rank, {"status": "error", "rank": rank, "error": repr(error), "traceback": traceback.format_exc()})
                raise
        return original_experts(self, *args, **kwargs)

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
