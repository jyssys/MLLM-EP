"""Bounded GPU grouped-MoE replay for fixed vs exact route-oracle cuts.

The replay uses immutable Qwen3-VL route IDs and the validated layer-24
activation/weight capture.  Only contiguous cut points differ between the two
variants.  It is deliberately not a serving-scheduler implementation.
"""
from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poc_flashvep.deepep_revalidation.operator_replay import (
    ExpertSpec,
    _correctness,
)
from poc_flashvep.tile_slack_mechanism.operator_replay import _run_variant
_INSTALLED = False
_RAN_RANKS: set[int] = set()
_CONTEXT: dict[str, int] = {}
SELECTED_REQUESTS = (
    "coins", "cat", "logo", "coffee", "coffee_rocket", "model_card",
    "retina", "method",
)
LAYERS = (0, 12, 24, 36, 47)
BUDGETS = (128, 256)


# Keep the worker hook dependency-free.  The analysis module also imports
# pandas/matplotlib, neither of which is installed in the validated vLLM
# runtime environment.  These are the same pre-registered boundary/DP rules
# used by the offline analysis, reproduced locally for the GPU worker.
def _block_m(n: int) -> int:
    return 16 if n <= 32 else 32 if n <= 96 else 64 if n <= 512 else 128


def _chunks_fixed(n: int, budget: int) -> list[int]:
    ends = list(range(0, n, budget))
    if not ends or ends[-1] != n:
        ends.append(n)
    return ends


def _valid_range(start: int, n: int, budget: int) -> tuple[int, int]:
    lo = start + max(1, int(np.ceil(0.75 * budget)))
    hi = min(n, start + int(np.floor(1.25 * budget)))
    return lo, hi


def _visual_prefix_counts(routes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, layers, _ = routes.shape
    out = np.zeros((layers, 128, n + 1), dtype=np.int32)
    for pos in range(n):
        out[:, :, pos + 1] = out[:, :, pos]
        if bool(mask[pos]):
            for layer in range(layers):
                out[layer, routes[pos, layer], pos + 1] += 1
    return out


def _chunks_oracle(routes: np.ndarray, mask: np.ndarray, budget: int,
                   prefix: np.ndarray) -> list[int]:
    """Exact bounded partition DP minimizing route-aware vision tile count."""
    n = len(mask)
    dp = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=np.int64)
    dp[0] = 0.0
    for start in range(n):
        if not np.isfinite(dp[start]):
            continue
        lo, hi = _valid_range(start, n, budget)
        ends = range(lo, hi + 1) if lo <= hi else ()
        for end in ends:
            bm = _block_m(end - start)
            h = prefix[:, :, end] - prefix[:, :, start]
            value = dp[start] + float(((h + bm - 1) // bm).sum())
            if value < dp[end] - 1e-9:
                dp[end] = value
                prev[end] = start
    if prev[n] < 0:
        return _chunks_fixed(n, budget)
    ends = [n]
    cur = n
    while cur > 0:
        cur = int(prev[cur])
        ends.append(cur)
    return list(reversed(ends))


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _load_routes(base: Path) -> list[dict[str, Any]]:
    manifest = json.loads((base / "workload_manifest.json").read_text())
    rows_by_id: dict[str, dict[str, Any]] = {}
    for pair in manifest["pairs"]:
        item = pair["vision"]
        request_id = item["request_id"]
        if request_id not in SELECTED_REQUESTS:
            continue
        with np.load(base / item["route_file"]) as archive:
            route_np = archive["routed_experts"].astype(np.int64)
            token_ids = archive["prompt_token_ids"].astype(np.int64)
        if route_np.shape[1:] != (48, 8) or len(token_ids) != len(route_np):
            raise AssertionError((request_id, route_np.shape, token_ids.shape))
        rows_by_id[request_id] = {
            "request_id": request_id,
            "category": item["category"],
            "pair_id": int(pair["pair_id"]),
            "routes": route_np,
            "token_ids": token_ids,
        }
    rows = [rows_by_id[request_id] for request_id in SELECTED_REQUESTS
            if request_id in rows_by_id]
    if tuple(row["request_id"] for row in rows) != SELECTED_REQUESTS:
        raise AssertionError("selected request manifest is incomplete or reordered")
    return rows


def _runtime_spec(values: dict[str, Any]) -> ExpertSpec:
    return ExpertSpec(
        in_dtype=values["in_dtype"], w1=values["w1"], w2=values["w2"],
        activation=values["activation"],
        global_num_experts=int(values["global_num_experts"]),
        local_num_experts=int(values["local_num_experts"]),
        expert_map=values["expert_map"],
        apply_router_weight_on_input=bool(values["apply_router_weight_on_input"]),
    )


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "p25_ms": float(np.quantile(arr, .25)),
        "p75_ms": float(np.quantile(arr, .75)),
        "mean_ms": float(np.mean(arr)),
        "cv": float(np.std(arr) / max(np.mean(arr), 1e-12)),
    }


def _one_rank_replay(kernel: Any, original_experts: Any, spec: ExpertSpec,
                     rank: int) -> dict[str, Any]:
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    if int(ep.world_size) != 4:
        raise AssertionError(f"expected EP4, got {ep.world_size}")
    if type(kernel.prepare_finalize).__name__ != "DeepEPHTPrepareAndFinalize":
        raise AssertionError(type(kernel.prepare_finalize).__name__)
    base = Path(os.environ["FLASHVEP_CHUNK_ROUTE_DIR"])
    capture = torch.load(
        os.environ["FLASHVEP_CHUNK_CAPTURE"], map_location="cpu", weights_only=False
    )
    samples = _load_routes(base)
    warmups = int(os.environ.get("FLASHVEP_CHUNK_WARMUPS", "5"))
    iterations = int(os.environ.get("FLASHVEP_CHUNK_ITERATIONS", "20"))
    buffer = kernel.prepare_finalize.buffer
    observations: list[dict[str, Any]] = []
    for sample in samples:
        route_np = sample["routes"]
        mask = sample["token_ids"] == 151655
        prefix = _visual_prefix_counts(route_np, mask)
        for budget in BUDGETS:
            cuts = {
                "fixed": _chunks_fixed(len(route_np), budget),
                "oracle": _chunks_oracle(route_np, mask, budget, prefix),
            }
            for layer in LAYERS:
                routes = torch.from_numpy(route_np[:, layer, :]).to(
                    spec.w1.device, dtype=torch.int64, non_blocking=True
                ).contiguous()
                outputs: dict[str, torch.Tensor] = {}
                for strategy in ("fixed", "oracle"):
                    ends = cuts[strategy]
                    groups = [list(range(st, en)) for st, en in zip(ends[:-1], ends[1:])]
                    if sum(map(len, groups)) != len(route_np):
                        raise AssertionError("chunk groups do not partition route")
                    timing, output = _run_variant(
                        "serial", groups, routes, capture, kernel, original_experts,
                        buffer, spec, rank, warmups, iterations,
                    )
                    outputs[strategy] = output
                    correctness = ({"passed": True} if strategy == "fixed" else
                                   _correctness(outputs["fixed"], output))
                    per_chunk = []
                    for sample_timing in timing["samples"]:
                        per_chunk.append(sample_timing.get("per_wave", []))
                    observations.append({
                        "request_id": sample["request_id"],
                        "category": sample["category"],
                        "pair_id": sample["pair_id"],
                        "budget": budget,
                        "layer": layer,
                        "rank": rank,
                        "strategy": strategy,
                        "total_tokens": int(len(route_np)),
                        "vision_tokens": int(mask.sum()),
                        "boundaries": ends,
                        "chunk_sizes": [int(en - st) for st, en in zip(ends[:-1], ends[1:])],
                        "chunks": len(ends) - 1,
                        "warmups": warmups,
                        "iterations": iterations,
                        "wall_stats": _stats([float(x["wall_ms"]) for x in timing["samples"]]),
                        "expert_stats": _stats([float(x["expert_ms"]) for x in timing["samples"]]),
                        "dispatch_stats": _stats([float(x["dispatch_ms"]) for x in timing["samples"]]),
                        "combine_stats": _stats([float(x["combine_ms"]) for x in timing["samples"]]),
                        "per_chunk_last_iteration": per_chunk[-1],
                        "correctness": correctness,
                        "route_identity": True,
                        "token_partition_identity": True,
                    })
                del routes, outputs
                torch.cuda.empty_cache()
    return {
        "status": "ok", "rank": rank,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_mapping": [1, 2, 3, 4],
        "settings": {
            "backend": type(kernel.fused_experts).__name__,
            "prepare_finalize": type(kernel.prepare_finalize).__name__,
            "communication": "DeepEP high-throughput",
            "warmups": warmups, "iterations": iterations,
            "route_source": str(base),
            "capture_source": os.environ["FLASHVEP_CHUNK_CAPTURE"],
            "activation_provenance": "validated Qwen3-VL layer-24 capture cycled to route length",
            "selected_requests": SELECTED_REQUESTS,
            "selected_layers": LAYERS,
        },
        "observations": observations,
    }


def _write(rank: int, payload: dict[str, Any]) -> None:
    out = Path(os.environ["FLASHVEP_CHUNK_REPLAY_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rank{rank}.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_chunk_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = _CONTEXT.get("layer", -1)
        _CONTEXT["layer"] = int(getattr(self, "_flashvep_chunk_layer", -1))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT["layer"] = previous

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        from vllm.distributed import get_ep_group
        rank = int(get_ep_group().rank_in_group)
        if (rank not in _RAN_RANKS and _CONTEXT.get("layer", -1) == 24
                and type(self.prepare_finalize).__name__ == "DeepEPHTPrepareAndFinalize"):
            _RAN_RANKS.add(rank)
            names = (
                "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                "topk_ids", "activation", "global_num_experts", "local_num_experts",
                "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
            )
            values = dict(zip(names, args, strict=False)); values.update(kwargs)
            try:
                _write(rank, _one_rank_replay(self, original_experts, _runtime_spec(values), rank))
            except BaseException as exc:
                _write(rank, {"status": "error", "rank": rank, "error": repr(exc),
                               "traceback": traceback.format_exc()})
                raise
        return original_experts(self, *args, **kwargs)

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
