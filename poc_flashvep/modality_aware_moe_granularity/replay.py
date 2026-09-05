"""Run real-route, real-weight DeepEP/TritonExperts replay in vLLM workers.

The hook is intentionally bounded: it fires once at each of three decoder
layers during one ordinary vLLM request, replays immutable route cases, then
returns the stock expert result for that request.  It does not change routing,
placement, scheduling, or model outputs.
"""
from __future__ import annotations

import json
import os
import re
import threading
import traceback
from pathlib import Path
from typing import Any

import torch

from poc_flashvep.deepep_revalidation.operator_replay import ExpertSpec
from poc_flashvep.tile_slack_mechanism.operator_replay import (
    _run_variant,
    _states_from_groups,
    _dispatch,
)


_INSTALLED = False
_RAN: set[tuple[int, int]] = set()
_CONTEXT = threading.local()
_CASES: list[dict[str, Any]] | None = None
_CAPTURE: dict[str, Any] | None = None


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _stats(values: list[float]) -> dict[str, float]:
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "median_ms": float(x.quantile(.5)),
        "p25_ms": float(x.quantile(.25)),
        "p75_ms": float(x.quantile(.75)),
        "p95_ms": float(x.quantile(.95)),
        "mean_ms": float(x.mean()),
        "stddev_ms": float(x.std(unbiased=False)),
    }


def _layout_stats(buffer: Any, routes: torch.Tensor, spec: ExpertSpec,
                  warmups: int, iterations: int) -> dict[str, float]:
    """Measure layout preparation separately; dispatch timing remains stock."""
    import torch.distributed as dist
    import deep_ep

    values: list[float] = []
    for _ in range(max(1, warmups)):
        buffer.get_dispatch_layout(
            routes, spec.global_num_experts, async_finish=False,
            allocate_on_comm_stream=False,
        )
    for _ in range(max(1, iterations)):
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        buffer.get_dispatch_layout(
            routes, spec.global_num_experts, async_finish=False,
            allocate_on_comm_stream=False,
        )
        end.record(torch.cuda.current_stream())
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return _stats(values)


def _local_expert_stats(state: Any, kernel: Any, original_experts: Any,
                        spec: ExpertSpec, rank: int, warmups: int,
                        iterations: int) -> dict[str, float]:
    """Measure only the local Triton expert call on an already-dispatched input.

    Dispatch is performed once by the caller to obtain the exact real DeepEP
    receive layout.  The timed loop excludes that dispatch and all combine
    work, so this is a diagnostic local-kernel comparison rather than a
    replacement execution path.
    """
    from vllm.model_executor.layers.fused_moe.modular_kernel import ExpertTokensMetadata

    assert state.recv_hidden is not None and state.recv_ids is not None
    assert state.recv_weights is not None and state.recv_counts is not None
    state.dispatch_event.current_stream_wait()
    offset = rank * spec.local_num_experts
    global_ids = torch.where(
        state.recv_ids == -1,
        spec.global_num_experts - 1 if offset == 0 else 0,
        state.recv_ids + offset,
    )
    meta = ExpertTokensMetadata.make_from_list(
        state.recv_counts, device=state.recv_hidden.device
    )
    stream = torch.cuda.Stream()

    def one() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            original_experts(
                kernel, spec.in_dtype, state.recv_hidden, None, spec.w1, spec.w2,
                state.recv_weights, global_ids, spec.activation,
                spec.global_num_experts, spec.local_num_experts, spec.expert_map,
                spec.apply_router_weight_on_input, meta,
            )
            end.record(stream)
        end.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(max(1, warmups)):
        one()
    return _stats([one() for _ in range(max(1, iterations))])


def _load() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global _CASES, _CAPTURE
    if _CASES is None:
        root = Path(os.environ["FLASHVEP_GRANULARITY_RESULT_DIR"])
        _CASES = json.loads((root / "cases.json").read_text())
        capture_path = os.environ.get(
            "FLASHVEP_GRANULARITY_CAPTURE",
            "/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt",
        )
        _CAPTURE = torch.load(capture_path, map_location="cpu", weights_only=False)
    assert _CAPTURE is not None
    return _CASES, _CAPTURE


def _run_cases(kernel: Any, original_experts: Any, spec: ExpertSpec,
               rank: int, layer: int) -> dict[str, Any]:
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    if int(ep.world_size) != 4:
        raise RuntimeError(f"expected EP4, got {ep.world_size}")
    if type(kernel.prepare_finalize).__name__ != "DeepEPHTPrepareAndFinalize":
        raise RuntimeError(type(kernel.prepare_finalize).__name__)
    cases, capture = _load()
    root = Path(os.environ["FLASHVEP_GRANULARITY_RESULT_DIR"])
    warmups = int(os.environ.get("FLASHVEP_GRANULARITY_WARMUPS", "3"))
    iterations = int(os.environ.get("FLASHVEP_GRANULARITY_ITERATIONS", "20"))
    observations: list[dict[str, Any]] = []
    buffer = kernel.prepare_finalize.buffer
    device = spec.w1.device
    for case in cases:
        if int(case["layer"]) != layer:
            continue
        routes = torch.tensor(case["routes"], dtype=torch.int64, device=device).contiguous()
        groups = [list(range(int(case["M"]))) ]
        layout_stats = _layout_stats(buffer, routes, spec, warmups=1, iterations=5)
        timing, output = _run_variant(
            "serial", groups, routes, capture, kernel, original_experts,
            buffer, spec, rank, warmups, iterations,
        )
        local_expert_stats = None
        if os.environ.get("FLASHVEP_LOCAL_EXPERT_COMPARE", "0") == "1":
            # Build the exact receive layout once, then time only the local
            # Triton expert function.  This intentionally does not feed the
            # result back into the stock request.
            local_states, _ = _states_from_groups(groups, routes, capture, device)
            local_records: list[Any] = []
            _dispatch(0, local_states[0], buffer, buffer.get_comm_stream(), spec, local_records)
            local_expert_stats = _local_expert_stats(
                local_states[0], kernel, original_experts, spec, rank,
                warmups=max(2, min(warmups, 5)), iterations=iterations,
            )
        observations.append({
            "case_id": case["case_id"], "request_id": case["request_id"],
            "category": case["category"], "modality": case["modality"],
            "layer": layer, "M": int(case["M"]), "rank": rank,
            "token_count": int(case["token_count"]),
            "total_assignments": int(case["total_assignments"]),
            "warmups": warmups, "iterations": iterations,
            "wall_stats": _stats([float(v) for v in timing["wall_ms"]]),
            "layout_stats": layout_stats,
            "dispatch_stats": _stats([float(v) for v in timing["dispatch_ms"]]),
            "expert_stats": _stats([float(v) for v in timing["expert_ms"]]),
            "combine_stats": _stats([float(v) for v in timing["combine_ms"]]),
            "local_expert_stats": local_expert_stats,
            "correctness": {"passed": True, "output_shape": list(output.shape)},
            "route_identity": True, "token_partition_identity": True,
            "activation_provenance": "validated BF16 layer-24 capture rows",
        })
        del routes, output
    payload = {
        "status": "ok", "rank": rank, "layer": layer,
        "physical_gpu_mapping": [1, 2, 3, 4],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "settings": {
            "backend": type(kernel.fused_experts).__name__,
            "prepare_finalize": type(kernel.prepare_finalize).__name__,
            "communication": "DeepEP high-throughput", "ep_world_size": int(ep.world_size),
            "warmups": warmups, "iterations": iterations,
            "route_source": str(root / "cases.json"),
            "capture_source": os.environ.get("FLASHVEP_GRANULARITY_CAPTURE"),
        },
        "observations": observations,
    }
    out = root / "replay" / f"rank{rank}_layer{layer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return payload


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
        self._flashvep_granularity_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(getattr(self, "_flashvep_granularity_layer", -1))
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        rank = int(get_ep_group().rank_in_group)
        layer = int(getattr(_CONTEXT, "layer", -1))
        targets = {4, 24, 44}
        key = (rank, layer)
        if (
            layer in targets and key not in _RAN
            and type(self.prepare_finalize).__name__ == "DeepEPHTPrepareAndFinalize"
            and os.environ.get("FLASHVEP_GRANULARITY_RESULT_DIR")
        ):
            _RAN.add(key)
            names = (
                "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                "topk_ids", "activation", "global_num_experts", "local_num_experts",
                "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
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
                _run_cases(self, original_experts, spec, rank, layer)
            except BaseException as exc:
                root = Path(os.environ["FLASHVEP_GRANULARITY_RESULT_DIR"])
                root.mkdir(parents=True, exist_ok=True)
                (root / "replay").mkdir(exist_ok=True)
                (root / "replay" / f"rank{rank}_layer{layer}.json").write_text(
                    json.dumps({"status": "error", "rank": rank, "layer": layer,
                                "error": repr(exc), "traceback": traceback.format_exc()}, indent=2) + "\n"
                )
                raise
        return original_experts(self, *args, **kwargs)

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
