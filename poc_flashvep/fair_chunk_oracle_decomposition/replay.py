"""GPU replay for the fair chunk-oracle decomposition."""
from __future__ import annotations

import json
import hashlib
import os
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poc_flashvep.chunk_oracle_gpu_scale_validation import replay as base

_INSTALLED = False
_RAN: set[int] = set()
_CONTEXT: dict[str, int] = {}
SHORT_IDS = ("coins", "cat", "logo", "coffee", "coffee_rocket", "model_card", "retina", "method")
LONG_IDS = ("long_6img_natural_fine", "long_8img_mixed", "long_10img_chart_mixed", "long_12img_broad")
SHORT_LAYERS = (0, 12, 24, 36, 47)
LONG_LAYERS = (24,)
STRATEGIES = ("fixed", "balanced", "same_count", "strict", "relaxed")


def _load_routes() -> list[dict[str, Any]]:
    short = Path(os.environ["FAIR_SHORT_ROUTE_DIR"])
    long = Path(os.environ["FAIR_LONG_ROUTE_DIR"])
    manifest = json.loads((short / "workload_manifest.json").read_text())
    by_id = {p["vision"]["request_id"]: p["vision"] for p in manifest["pairs"]}
    rows: list[dict[str, Any]] = []
    for sid in SHORT_IDS:
        item = by_id[sid]
        with np.load(short / item["route_file"]) as z:
            routes, tokens = z["routed_experts"].astype(np.int64), z["prompt_token_ids"].astype(np.int64)
        rows.append({"request_id": sid, "category": item["category"], "routes": routes, "token_ids": tokens, "source": "short", "layers": SHORT_LAYERS})
    lm = json.loads((long / "sample_manifest.json").read_text())
    lby = {s["sample_id"]: s for s in lm["samples"]}
    for sid in LONG_IDS:
        with np.load(long / f"routing.{sid}.npz") as z:
            routes, tokens = z["routed_experts"].astype(np.int64), z["prompt_token_ids"].astype(np.int64)
        rows.append({"request_id": sid, "category": lby[sid]["category"], "routes": routes, "token_ids": tokens, "source": "long", "layers": LONG_LAYERS})
    return rows


def _stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "p25_ms": float(np.quantile(a, .25)), "p75_ms": float(np.quantile(a, .75)), "mean_ms": float(np.mean(a)), "cv": float(np.std(a) / max(np.mean(a), 1e-12))}


def _one_rank(kernel: Any, original_experts: Any, spec: Any, rank: int) -> dict[str, Any]:
    from vllm.distributed import get_ep_group
    ep = get_ep_group()
    if int(ep.world_size) != 4 or type(kernel.prepare_finalize).__name__ != "DeepEPHTPrepareAndFinalize":
        raise AssertionError(f"expected DeepEP EP4, got world={ep.world_size} prepare={type(kernel.prepare_finalize).__name__}")
    cuts_all = json.loads(Path(os.environ["FAIR_CUTS"]).read_text())["samples"]
    capture = torch.load(os.environ["FAIR_CAPTURE"], map_location="cpu", weights_only=False)
    samples = _load_routes()
    warmups, iterations = int(os.environ.get("FAIR_WARMUPS", "5")), int(os.environ.get("FAIR_ITERATIONS", "20"))
    prewarm = os.environ.get("FAIR_PREWARM", "1") not in {"0", "false", "False"}
    buffer = kernel.prepare_finalize.buffer
    observations: list[dict[str, Any]] = []
    for sample in samples:
        route_np, token_ids = sample["routes"], sample["token_ids"]
        for budget in (128, 256, 512, 1024):
            cuts = cuts_all[sample["request_id"]][str(budget)]
            for layer in sample["layers"]:
                routes = torch.from_numpy(route_np[:, layer, :]).to(spec.w1.device, dtype=torch.int64, non_blocking=True).contiguous()
                groups_by_strategy: dict[str, list[list[int]]] = {}
                for strategy in STRATEGIES:
                    ends = [int(x) for x in cuts[strategy]]
                    groups = [list(range(st, en)) for st, en in zip(ends[:-1], ends[1:])]
                    if sum(map(len, groups)) != len(route_np) or (any((en - st) > budget for st, en in zip(ends[:-1], ends[1:])) and strategy not in {"fixed", "relaxed"}):
                        raise AssertionError((sample["request_id"], budget, strategy, ends[:4], ends[-4:]))
                    groups_by_strategy[strategy] = groups
                # Prewarm all strategies so the first measured method does
                # not absorb one-time allocator/autotune/cache costs.  The
                # measured blocks use a deterministic per-observation order
                # shared by all EP ranks.
                if prewarm:
                    for strategy in STRATEGIES:
                        base._run_variant("serial", groups_by_strategy[strategy], routes, capture, kernel, original_experts, buffer, spec, rank, warmups, 1)
                seed_material = f"{sample['request_id']}:{budget}:{layer}".encode()
                seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
                order = list(STRATEGIES)
                np.random.default_rng(seed).shuffle(order)
                outputs: dict[str, torch.Tensor] = {}
                measured: dict[str, tuple[dict[str, Any], torch.Tensor, list[int]]] = {}
                for strategy in order:
                    timing, output = base._run_variant("serial", groups_by_strategy[strategy], routes, capture, kernel, original_experts, buffer, spec, rank, 0, iterations)
                    measured[strategy] = (timing, output, [int(x) for x in cuts[strategy]])
                    outputs[strategy] = output
                for strategy in STRATEGIES:
                    timing, output, ends = measured[strategy]
                    observations.append({"request_id": sample["request_id"], "source": sample["source"], "category": sample["category"], "budget": budget, "layer": layer, "rank": rank, "strategy": strategy, "measurement_order": order, "prewarmed": prewarm, "total_tokens": len(route_np), "vision_tokens": int((token_ids == 151655).sum()), "boundaries": ends, "chunk_sizes": [en - st for st, en in zip(ends[:-1], ends[1:])], "chunks": len(ends) - 1, "warmups": warmups, "iterations": iterations, "wall_stats": _stats([x["wall_ms"] for x in timing["samples"]]), "expert_stats": _stats([x["expert_ms"] for x in timing["samples"]]), "dispatch_stats": _stats([x["dispatch_ms"] for x in timing["samples"]]), "combine_stats": _stats([x["combine_ms"] for x in timing["samples"]]), "correctness": {"passed": True} if strategy == "fixed" else base._correctness(outputs["fixed"], output), "route_identity": True, "token_partition_identity": True})
                del routes, outputs
                torch.cuda.empty_cache()
    out = Path(os.environ["FAIR_REPLAY_DIR"]); out.mkdir(parents=True, exist_ok=True)
    (out / f"rank{rank}.json").write_text(json.dumps({"status": "ok", "rank": rank, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "physical_gpu_mapping": [1, 2, 3, 4], "settings": {"backend": type(kernel.fused_experts).__name__, "prepare_finalize": type(kernel.prepare_finalize).__name__, "communication": "DeepEP high-throughput", "warmups": warmups, "iterations": iterations, "prewarm_all_strategies": prewarm, "measurement_order": "deterministic_sha256_per_observation", "layers_short": SHORT_LAYERS, "layers_long": LONG_LAYERS}, "observations": observations}, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    original_init, original_forward, original_experts = Qwen3MoeDecoderLayer.__init__, Qwen3MoeDecoderLayer.forward, FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        m = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
        self._fair_layer = int(m.group(1)) if m else -1

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        old = _CONTEXT.get("layer", -1); _CONTEXT["layer"] = int(getattr(self, "_fair_layer", -1))
        try: return original_forward(self, *args, **kwargs)
        finally: _CONTEXT["layer"] = old

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        from vllm.distributed import get_ep_group
        rank = int(get_ep_group().rank_in_group)
        if rank not in _RAN and _CONTEXT.get("layer", -1) == 24:
            _RAN.add(rank)
            names = ("in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights", "topk_ids", "activation", "global_num_experts", "local_num_experts", "expert_map", "apply_router_weight_on_input", "expert_tokens_meta")
            values = dict(zip(names, args, strict=False)); values.update(kwargs)
            try: _one_rank(self, original_experts, base._runtime_spec(values), rank)
            except BaseException:
                traceback.print_exc(); raise
        return original_experts(self, *args, **kwargs)

    Qwen3MoeDecoderLayer.__init__, Qwen3MoeDecoderLayer.forward = patched_init, patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
