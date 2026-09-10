"""Read-only worker instrumentation and full-unit pairwise replay.

The hook captures a complete Qwen3 MoE language-attention invocation and runs
it, without token slicing, next to real DeepEP/Triton stages. The original
model forward finishes before diagnostic replays begin, so a benchmark never
overwrites an in-flight model dispatch buffer.
"""

from __future__ import annotations

import atexit
import json
import os
import random
import re
import statistics
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from poc_flashvep.deepep_revalidation.operator_replay import (
    ExpertSpec,
    MicroState,
    _correctness,
    _event,
    _elapsed,
)


_LOCK = threading.Lock()
_CONTEXT = threading.local()
_CONTROL: dict[str, Any] = {}
_CONTROL_SIG: tuple[int, int, int] | None = None
_TIMINGS: list[dict[str, Any]] = []
_LOGITS: dict[int, torch.Tensor] = {}
_ATTN_MODULE: Any = None
_ATTN_POSITIONS: torch.Tensor | None = None
_ATTN_HIDDEN: torch.Tensor | None = None
_ORIGINAL_ATTN: Any = None
_LATEST_PREP: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
_KERNEL: Any = None
_SPEC: ExpertSpec | None = None
_ORIGINAL_EXPERTS: Any = None
_PAIR_RAN = False
_POLICY_RAN: set[str] = set()
_FLUSHED = False


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _target_layer() -> int:
    return int(os.environ.get("MODPHASE_LAYER", "24"))


def _control() -> dict[str, Any]:
    global _CONTROL, _CONTROL_SIG
    path = Path(os.environ["MODPHASE_CONTROL"])
    if not path.exists():
        return {}
    stat = path.stat()
    sig = (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))
    if sig != _CONTROL_SIG:
        _CONTROL = json.loads(path.read_text(encoding="utf-8"))
        _CONTROL_SIG = sig
    return _CONTROL


def _ep_rank() -> int:
    from vllm.distributed import get_ep_group

    return int(get_ep_group().rank_in_group)


def _write(name: str, value: object) -> None:
    root = Path(os.environ["MODPHASE_RAW"])
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(x) for x in values)
    def quantile(q: float) -> float:
        p = (len(ordered) - 1) * q
        lo = int(p); hi = min(lo + 1, len(ordered) - 1); w = p - lo
        return ordered[lo] * (1 - w) + ordered[hi] * w
    return {
        "n": len(ordered), "median_ms": float(statistics.median(ordered)),
        "p10_ms": quantile(0.1), "p90_ms": quantile(0.9),
        "mean_ms": float(statistics.fmean(ordered)),
        "cv": float(statistics.stdev(ordered) / statistics.fmean(ordered))
        if len(ordered) > 1 and statistics.fmean(ordered) else 0.0,
    }


def _sync() -> None:
    dist.barrier(group=dist.group.WORLD)
    torch.cuda.synchronize()


def _record(stage: str, start: torch.cuda.Event, end: torch.cuda.Event, tokens: int) -> None:
    entry = dict(getattr(_CONTEXT, "entry", {}))
    if not entry:
        return
    with _LOCK:
        _TIMINGS.append({
            "wave": int(entry.get("wave", -1)), "mode": entry.get("mode"),
            "modality": entry.get("modality"), "iteration": int(entry.get("iteration", -1)),
            "ep_rank": _ep_rank(), "layer": int(getattr(_CONTEXT, "layer", -1)),
            "stage": stage, "tokens": int(tokens), "start": start, "end": end,
        })


def _tokens(args: tuple[Any, ...], kwargs: dict[str, Any], index: int) -> int:
    value = kwargs.get("hidden_states")
    if value is None and len(args) > index:
        value = args[index]
    return int(value.shape[0]) if isinstance(value, torch.Tensor) else -1


def _attention(stream: torch.cuda.Stream, wait: torch.cuda.Event | None = None) -> tuple[torch.Tensor, torch.cuda.Event, torch.cuda.Event]:
    assert _ORIGINAL_ATTN is not None and _ATTN_MODULE is not None
    assert _ATTN_POSITIONS is not None and _ATTN_HIDDEN is not None
    start, end = _event(), _event()
    with torch.cuda.stream(stream):
        if wait is not None:
            stream.wait_event(wait)
        start.record(stream)
        torch.cuda.nvtx.range_push("MODPHASE_FULL_ATTENTION")
        output = _ORIGINAL_ATTN(
            _ATTN_MODULE,
            positions=_ATTN_POSITIONS,
            hidden_states=_ATTN_HIDDEN,
        )
        torch.cuda.nvtx.range_pop()
        end.record(stream)
    return output, start, end


def _fresh_state(hidden: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, repeat: int = 1) -> MicroState:
    return MicroState(
        hidden.repeat((repeat, 1)).contiguous(),
        weights.repeat((repeat, 1)).contiguous(),
        ids.repeat((repeat, 1)).to(torch.int64).contiguous(),
    )


def _dispatch(state: MicroState, buffer: Any, spec: ExpertSpec) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    import deep_ep

    comm = buffer.get_comm_stream()
    start, end = _event(), _event(); start.record(comm)
    layout = buffer.get_dispatch_layout(
        state.ids, spec.global_num_experts, async_finish=True,
        allocate_on_comm_stream=False,
    )
    num_rank, num_rdma, num_exp, in_rank, layout_event = layout
    dispatched = buffer.dispatch(
        x=state.hidden, handle=None, num_tokens_per_rank=num_rank,
        num_tokens_per_rdma_rank=num_rdma, is_token_in_rank=in_rank,
        num_tokens_per_expert=num_exp, topk_idx=state.ids.to(deep_ep.topk_idx_t),
        topk_weights=state.weights, expert_alignment=1,
        config=deep_ep.Buffer.get_dispatch_config(dist.get_world_size()),
        previous_event=layout_event, async_finish=True,
        allocate_on_comm_stream=False,
    )
    (state.recv_hidden, state.recv_ids, state.recv_weights, state.recv_counts,
     state.handle, state.dispatch_event) = dispatched
    end.record(comm)
    return start, end


def _expert(state: MicroState, stream: torch.cuda.Stream, spec: ExpertSpec, rank: int) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    import deep_ep
    from vllm.model_executor.layers.fused_moe.modular_kernel import ExpertTokensMetadata
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate,
    )

    assert _KERNEL is not None and _ORIGINAL_EXPERTS is not None
    assert state.recv_hidden is not None and state.recv_ids is not None
    assert state.recv_weights is not None and state.recv_counts is not None
    start, end = _event(), _event()
    with torch.cuda.stream(stream):
        state.dispatch_event.current_stream_wait()
        offset = rank * spec.local_num_experts
        global_ids = torch.where(
            state.recv_ids == -1,
            spec.global_num_experts - 1 if offset == 0 else 0,
            state.recv_ids + offset,
        )
        meta = ExpertTokensMetadata.make_from_list(state.recv_counts, device=state.recv_hidden.device)
        start.record(stream)
        raw = _ORIGINAL_EXPERTS(
            _KERNEL, spec.in_dtype, state.recv_hidden, None, spec.w1, spec.w2,
            state.recv_weights, global_ids, spec.activation, spec.global_num_experts,
            spec.local_num_experts, spec.expert_map, spec.apply_router_weight_on_input, meta,
        )
        reducer = _KERNEL.fused_experts.finalize_weight_and_reduce_impl()
        if isinstance(reducer, TopKWeightAndReduceDelegate):
            reducer = TopKWeightAndReduceContiguous()
        state.expert_output = reducer.apply(
            output=None, fused_expert_output=raw, topk_weights=state.recv_weights,
            topk_ids=global_ids,
            apply_router_weight_on_input=spec.apply_router_weight_on_input,
        ).clone()
        end.record(stream)
        state.expert_event = deep_ep.Buffer.capture()
    return start, end


def _combine(state: MicroState, buffer: Any) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    import deep_ep

    assert state.expert_output is not None
    comm = buffer.get_comm_stream()
    with torch.cuda.stream(comm):
        state.expert_event.current_stream_wait()
    start, end = _event(), _event(); start.record(comm)
    state.combined, _, state.combine_event = buffer.combine(
        x=state.expert_output, handle=state.handle, topk_weights=None,
        config=deep_ep.Buffer.get_combine_config(dist.get_world_size()),
        async_finish=True, allocate_on_comm_stream=False,
    )
    end.record(comm)
    return start, end


def _wait_dispatch(state: MicroState) -> torch.cuda.Event:
    state.dispatch_event.current_stream_wait()
    done = _event(); done.record(torch.cuda.current_stream()); done.synchronize()
    return done


def _cleanup_after_dispatch(state: MicroState, buffer: Any) -> None:
    _, _, event = buffer.combine(
        x=state.recv_hidden, handle=state.handle, topk_weights=None,
        config=__import__("deep_ep").Buffer.get_combine_config(dist.get_world_size()),
        async_finish=True, allocate_on_comm_stream=False,
    )
    event.current_stream_wait(); done = _event(); done.record(); done.synchronize()


def _setup_to_expert(state: MicroState, buffer: Any, spec: ExpertSpec, rank: int, expert_stream: torch.cuda.Stream) -> None:
    _dispatch(state, buffer, spec)
    _wait_dispatch(state)
    _expert(state, expert_stream, spec, rank)
    state.expert_event.current_stream_wait(); done = _event(); done.record(); done.synchronize()


def _run_stage(
    phase: str, state: MicroState, buffer: Any, spec: ExpertSpec, rank: int,
    expert_stream: torch.cuda.Stream, origin: torch.cuda.Event | None = None,
) -> tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]:
    default = torch.cuda.current_stream()
    if origin is None:
        origin = _event(); origin.record(default)
    if phase == "dispatch":
        start, internal_end = _dispatch(state, buffer, spec)
        state.dispatch_event.current_stream_wait()
        end = _event(); end.record(default); end.synchronize()
        _cleanup_after_dispatch(state, buffer)
        return start, end, internal_end
    if phase in ("expert", "combine"):
        _dispatch(state, buffer, spec); _wait_dispatch(state)
        if phase == "expert":
            start, end = _expert(state, expert_stream, spec, rank)
            end.synchronize()
            # Preserve the expert output until cleanup combine consumes it.
            state.expert_event.current_stream_wait()
            _, _, cleanup = buffer.combine(
                x=state.expert_output, handle=state.handle, topk_weights=None,
                config=__import__("deep_ep").Buffer.get_combine_config(dist.get_world_size()),
                async_finish=True, allocate_on_comm_stream=False,
            )
            cleanup.current_stream_wait(); done = _event(); done.record(); done.synchronize()
            return start, end, end
        _expert(state, expert_stream, spec, rank)
        state.expert_event.current_stream_wait(); ready = _event(); ready.record(); ready.synchronize()
        start, internal_end = _combine(state, buffer)
        state.combine_event.current_stream_wait()
        end = _event(); end.record(default); end.synchronize()
        return start, end, internal_end
    if phase == "moe":
        start = origin
        _dispatch(state, buffer, spec)
        _expert(state, expert_stream, spec, rank)
        _combine(state, buffer)
        state.combine_event.current_stream_wait()
        end = _event(); end.record(default); end.synchronize()
        return start, end, end
    raise ValueError(phase)


def _fixed_workload(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    capture = torch.load(os.environ["MODPHASE_CAPTURE"], map_location="cpu", weights_only=False)
    repeat = int(os.environ.get("MODPHASE_FIXED_REPEAT", "3"))
    return (
        capture["post_attention_hidden"].to(device).repeat((repeat, 1)).contiguous(),
        capture["topk_weights"].to(device).repeat((repeat, 1)).contiguous(),
        capture["topk_expert_ids"].to(device).repeat((repeat, 1)).to(torch.int64).contiguous(),
    )


def _measure_attention_only(stream: torch.cuda.Stream) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    output, start, end = _attention(stream)
    end.synchronize()
    return _elapsed(start, end), output.detach().clone()


def _measure_stage_only(phase: str, hidden: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
                        buffer: Any, spec: ExpertSpec, rank: int, expert_stream: torch.cuda.Stream) -> float:
    torch.cuda.synchronize()
    state = _fresh_state(hidden, weights, ids)
    default = torch.cuda.current_stream()
    if phase in ("expert", "combine"):
        _dispatch(state, buffer, spec); _wait_dispatch(state)
        if phase == "combine":
            _expert(state, expert_stream, spec, rank)
            state.expert_event.current_stream_wait(); ready = _event(); ready.record(); ready.synchronize()
    origin = _event(); origin.record(default)
    if phase == "dispatch":
        _, _ = _dispatch(state, buffer, spec)
        state.dispatch_event.current_stream_wait(); end = _event(); end.record(default); end.synchronize()
        _cleanup_after_dispatch(state, buffer)
    elif phase == "expert":
        _, end = _expert(state, expert_stream, spec, rank); end.synchronize()
        state.expert_event.current_stream_wait()
        _, _, cleanup = buffer.combine(
            x=state.expert_output, handle=state.handle, topk_weights=None,
            config=__import__("deep_ep").Buffer.get_combine_config(dist.get_world_size()),
            async_finish=True, allocate_on_comm_stream=False,
        )
        cleanup.current_stream_wait(); cleanup_done = _event(); cleanup_done.record(); cleanup_done.synchronize()
    elif phase == "combine":
        _combine(state, buffer); state.combine_event.current_stream_wait()
        end = _event(); end.record(default); end.synchronize()
    elif phase == "moe":
        _dispatch(state, buffer, spec); _expert(state, expert_stream, spec, rank); _combine(state, buffer)
        state.combine_event.current_stream_wait(); end = _event(); end.record(default); end.synchronize()
    else:
        raise ValueError(phase)
    return _elapsed(origin, end)


def _measure_concurrent(phase: str, hidden: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
                        buffer: Any, spec: ExpertSpec, rank: int, attn_stream: torch.cuda.Stream,
                        expert_stream: torch.cuda.Stream) -> tuple[float, float, float, torch.Tensor]:
    torch.cuda.synchronize()
    state = _fresh_state(hidden, weights, ids)
    # Expert/combine setup must be outside the paired interval.
    if phase in ("expert", "combine"):
        _dispatch(state, buffer, spec); _wait_dispatch(state)
        if phase == "combine":
            _expert(state, expert_stream, spec, rank)
            state.expert_event.current_stream_wait(); ready = _event(); ready.record(); ready.synchronize()
    default = torch.cuda.current_stream()
    origin = _event(); origin.record(default)
    attn_output, attn_start, attn_end = _attention(attn_stream, origin)
    if phase == "dispatch":
        stage_start, _ = _dispatch(state, buffer, spec)
        state.dispatch_event.current_stream_wait(); stage_end = _event(); stage_end.record(default)
    elif phase == "expert":
        with torch.cuda.stream(expert_stream):
            expert_stream.wait_event(origin)
        stage_start, stage_end = _expert(state, expert_stream, spec, rank)
    elif phase == "combine":
        stage_start, _ = _combine(state, buffer)
        state.combine_event.current_stream_wait(); stage_end = _event(); stage_end.record(default)
    elif phase == "moe":
        stage_start = origin
        _dispatch(state, buffer, spec)
        _expert(state, expert_stream, spec, rank)
        _combine(state, buffer)
        state.combine_event.current_stream_wait(); stage_end = _event(); stage_end.record(default)
    else:
        raise ValueError(phase)
    default.wait_event(attn_end); default.wait_event(stage_end)
    joined = _event(); joined.record(default); joined.synchronize()
    # Complete or clean up the DeepEP handle after the timed region.
    if phase == "dispatch":
        _cleanup_after_dispatch(state, buffer)
    elif phase == "expert":
        state.expert_event.current_stream_wait()
        _, _, cleanup = buffer.combine(
            x=state.expert_output, handle=state.handle, topk_weights=None,
            config=__import__("deep_ep").Buffer.get_combine_config(dist.get_world_size()),
            async_finish=True, allocate_on_comm_stream=False,
        )
        cleanup.current_stream_wait(); done = _event(); done.record(); done.synchronize()
    torch.cuda.synchronize()
    return (_elapsed(origin, joined), _elapsed(attn_start, attn_end),
            _elapsed(stage_start, stage_end), attn_output.detach().clone())


def _run_pairwise() -> None:
    global _PAIR_RAN
    if _PAIR_RAN:
        return
    _PAIR_RAN = True
    assert _SPEC is not None and _KERNEL is not None
    rank = _ep_rank(); buffer = _KERNEL.prepare_finalize.buffer
    import deep_ep
    deep_ep.Buffer.set_num_sms(20)
    hidden, weights, ids = _fixed_workload(_SPEC.w1.device)
    warmups = int(os.environ.get("MODPHASE_PAIR_WARMUPS", "4"))
    iterations = int(os.environ.get("MODPHASE_PAIR_ITERATIONS", "12"))
    attn_stream = torch.cuda.Stream(); expert_stream = torch.cuda.Stream()
    rows: list[dict[str, Any]] = []
    attention_checks: list[dict[str, Any]] = []
    reference_attention: torch.Tensor | None = None
    for phase in ("dispatch", "combine", "expert", "moe"):
        for iteration in range(warmups + iterations):
            order = ["attention", "stage", "concurrent"]
            random.Random(20260911 + iteration).shuffle(order)
            values: dict[str, Any] = {}
            for item in order:
                _sync()
                if item == "attention":
                    values["attention_ms"], output = _measure_attention_only(attn_stream)
                    if reference_attention is None:
                        reference_attention = output
                    values["attention_output"] = output
                elif item == "stage":
                    values["stage_ms"] = _measure_stage_only(
                        phase, hidden, weights, ids, buffer, _SPEC, rank, expert_stream)
                else:
                    (values["concurrent_wall_ms"], values["attention_concurrent_ms"],
                     values["stage_concurrent_ms"], values["concurrent_output"]) = _measure_concurrent(
                        phase, hidden, weights, ids, buffer, _SPEC, rank, attn_stream, expert_stream)
            if iteration >= warmups:
                tx = float(values["attention_ms"]); ty = float(values["stage_ms"])
                txy = float(values["concurrent_wall_ms"])
                rows.append({
                    "phase": phase, "iteration": iteration - warmups,
                    "attention_ms": tx, "stage_ms": ty, "concurrent_wall_ms": txy,
                    "attention_concurrent_ms": float(values["attention_concurrent_ms"]),
                    "stage_concurrent_ms": float(values["stage_concurrent_ms"]),
                    "eta": (tx + ty - txy) / min(tx, ty),
                })
                attention_checks.append(_correctness(reference_attention, values["concurrent_output"]))
    _sync()
    _write(f"pairwise.rank{rank}.json", {
        "rank": rank, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "modality": _control().get("modality"), "target_layer": _target_layer(),
        "attention_tokens": int(_ATTN_HIDDEN.shape[0]), "stage_tokens_per_rank": int(hidden.shape[0]),
        "warmups": warmups, "iterations": iterations, "rows": rows,
        "attention_correctness": attention_checks,
        "timing": "same-device CUDA events; rank-critical aggregation is offline",
    })


def _run_full_moe(state: MicroState, buffer: Any, spec: ExpertSpec, rank: int,
                  expert_stream: torch.cuda.Stream) -> tuple[dict[str, float], torch.Tensor]:
    default = torch.cuda.current_stream(); torch.cuda.synchronize()
    origin = _event(); origin.record(default)
    d0, d1 = _dispatch(state, buffer, spec)
    e0, e1 = _expert(state, expert_stream, spec, rank)
    c0, c1 = _combine(state, buffer)
    state.combine_event.current_stream_wait(); done = _event(); done.record(default); done.synchronize()
    assert state.combined is not None
    return ({"total_ms": _elapsed(origin, done), "dispatch_ms": _elapsed(d0, d1),
             "expert_ms": _elapsed(e0, e1), "combine_ms": _elapsed(c0, c1)},
            state.combined.detach().clone())


def _run_policy(kind: str) -> None:
    if kind in _POLICY_RAN:
        return
    _POLICY_RAN.add(kind)
    assert _LATEST_PREP is not None and _SPEC is not None and _KERNEL is not None
    hidden, weights, ids = _LATEST_PREP
    rank = _ep_rank(); buffer = _KERNEL.prepare_finalize.buffer
    import deep_ep
    sms_values = [4, 8, 12, 16, 20]
    aggregations = [1, 2, 4] if kind != "decode" else [1, 4, 16, 64]
    warmups = int(os.environ.get("MODPHASE_POLICY_WARMUPS", "3"))
    iterations = int(os.environ.get("MODPHASE_POLICY_ITERATIONS", "10"))
    expert_stream = torch.cuda.Stream(); rows: list[dict[str, Any]] = []
    references: dict[int, torch.Tensor] = {}
    correctness: list[dict[str, Any]] = []
    for aggregation in aggregations:
        for iteration in range(warmups + iterations):
            order = list(sms_values); random.Random(9301 + aggregation * 97 + iteration).shuffle(order)
            for sms in order:
                _sync(); deep_ep.Buffer.set_num_sms(sms)
                state = _fresh_state(hidden, weights, ids, aggregation)
                timing, output = _run_full_moe(state, buffer, _SPEC, rank, expert_stream)
                if iteration >= warmups:
                    rows.append({"kind": kind, "aggregation": aggregation, "sms": sms,
                                 "iteration": iteration - warmups, "tokens_per_rank": int(state.hidden.shape[0]),
                                 **timing})
                    if aggregation not in references and sms == 20:
                        references[aggregation] = output
                    elif aggregation in references:
                        correctness.append({"aggregation": aggregation, "sms": sms,
                                            **_correctness(references[aggregation], output)})
    deep_ep.Buffer.set_num_sms(20); _sync()
    _write(f"policy.{kind}.rank{rank}.json", {
        "rank": rank, "kind": kind, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "base_tokens_per_rank": int(hidden.shape[0]), "rows": rows,
        "correctness": correctness,
        "aggregation_semantics": "concatenate whole request/layer units; no token-axis split",
    })


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    # sitecustomize is also imported by driver/coordinator interpreters that
    # never initialize an EP group and never collect worker records.
    if not _TIMINGS and not _LOGITS:
        return
    if _TIMINGS:
        _TIMINGS[-1]["end"].synchronize()
    rank = _ep_rank(); rows = []
    for item in _TIMINGS:
        row = {key: value for key, value in item.items() if key not in ("start", "end")}
        row["duration_ms"] = _elapsed(item["start"], item["end"])
        rows.append(row)
    _write(f"timing.rank{rank}.json", {
        "rank": rank, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rows": rows, "semantics": "same-device CUDA duration only",
    })
    if _LOGITS:
        root = Path(os.environ["MODPHASE_RAW"]); root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / f"logits.rank{rank}.npz",
                            **{f"wave_{wave}": value.numpy() for wave, value in _LOGITS.items()})


def install() -> None:
    global _ORIGINAL_ATTN, _ORIGINAL_EXPERTS, _KERNEL, _SPEC, _LATEST_PREP
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeAttention, Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock,
    )
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    from vllm.v1.worker import gpu_model_runner as gmr

    if getattr(Qwen3VLMoeForConditionalGeneration, "_modphase_installed", False):
        return
    Qwen3VLMoeForConditionalGeneration._modphase_installed = True
    original_outer = Qwen3VLMoeForConditionalGeneration.forward
    original_logits = Qwen3VLMoeForConditionalGeneration.compute_logits
    original_execute = gmr.GPUModelRunner.execute_model
    original_attn_init = Qwen3MoeAttention.__init__
    _ORIGINAL_ATTN = Qwen3MoeAttention.forward
    original_layer_init = Qwen3MoeDecoderLayer.__init__
    original_layer = Qwen3MoeDecoderLayer.forward
    original_moe = Qwen3MoeSparseMoeBlock.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    _ORIGINAL_EXPERTS = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    def active_timing() -> bool:
        return _control().get("mode") == "instrumented"

    def patched_attn_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_attn_init(self, *args, **kwargs)
        self._modphase_layer = _layer(str(kwargs.get("prefix", "")))

    def patched_layer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_layer_init(self, *args, **kwargs)
        self._modphase_layer = _layer(str(kwargs.get("prefix", args[1] if len(args) > 1 else "")))

    def patched_outer(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "entry", {})
        _CONTEXT.entry = dict(_control())
        try:
            return original_outer(self, *args, **kwargs)
        finally:
            _CONTEXT.entry = previous

    def patched_layer(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1); _CONTEXT.layer = int(self._modphase_layer)
        try:
            if not active_timing():
                return original_layer(self, *args, **kwargs)
            start, end = _event(), _event(); start.record()
            result = original_layer(self, *args, **kwargs); end.record()
            _record("layer_total", start, end, _tokens(args, kwargs, 1))
            return result
        finally:
            _CONTEXT.layer = previous

    def patched_attn(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _ATTN_MODULE, _ATTN_POSITIONS, _ATTN_HIDDEN
        layer = int(getattr(self, "_modphase_layer", -1)); mode = _control().get("mode")
        if layer == _target_layer() and mode in ("pairwise", "policy_decode"):
            positions = kwargs.get("positions", args[0] if args else None)
            hidden = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
            if isinstance(positions, torch.Tensor) and isinstance(hidden, torch.Tensor):
                _ATTN_MODULE = self; _ATTN_POSITIONS = positions.detach().clone()
                _ATTN_HIDDEN = hidden.detach().clone()
        if not active_timing():
            return _ORIGINAL_ATTN(self, *args, **kwargs)
        start, end = _event(), _event(); start.record()
        result = _ORIGINAL_ATTN(self, *args, **kwargs); end.record()
        _record("attention", start, end, _tokens(args, kwargs, 1))
        return result

    def patched_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _LATEST_PREP
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        weights = kwargs.get("topk_weights", args[1] if len(args) > 1 else None)
        ids = kwargs.get("topk_ids", args[2] if len(args) > 2 else None)
        if int(getattr(_CONTEXT, "layer", -1)) == _target_layer() and _control().get("mode") in ("pairwise", "policy_decode"):
            if all(isinstance(x, torch.Tensor) for x in (hidden, weights, ids)):
                _LATEST_PREP = (hidden.detach().clone(), weights.detach().clone(), ids.detach().clone())
        if not active_timing():
            return original_prepare(self, *args, **kwargs)
        start, end = _event(), _event(); start.record()
        result = original_prepare(self, *args, **kwargs); end.record()
        _record("dispatch", start, end, int(hidden.shape[0]) if isinstance(hidden, torch.Tensor) else -1)
        return result

    def patched_experts(self: Any, in_dtype: torch.dtype, a1q: torch.Tensor, a1q_scale: Any,
                        w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
                        topk_ids: torch.Tensor, activation: Any, global_num_experts: int,
                        local_num_experts: int, expert_map: Any,
                        apply_router_weight_on_input: bool, expert_tokens_meta: Any) -> torch.Tensor:
        global _KERNEL, _SPEC
        if int(getattr(_CONTEXT, "layer", -1)) == _target_layer():
            _KERNEL = self
            _SPEC = ExpertSpec(in_dtype, w1, w2, activation, int(global_num_experts),
                               int(local_num_experts), expert_map, bool(apply_router_weight_on_input))
        if not active_timing():
            return _ORIGINAL_EXPERTS(self, in_dtype, a1q, a1q_scale, w1, w2, topk_weights,
                                     topk_ids, activation, global_num_experts, local_num_experts,
                                     expert_map, apply_router_weight_on_input, expert_tokens_meta)
        start, end = _event(), _event(); start.record()
        result = _ORIGINAL_EXPERTS(self, in_dtype, a1q, a1q_scale, w1, w2, topk_weights,
                                   topk_ids, activation, global_num_experts, local_num_experts,
                                   expert_map, apply_router_weight_on_input, expert_tokens_meta)
        end.record(); _record("expert", start, end, int(a1q.shape[0]))
        return result

    def patched_finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not active_timing():
            return original_finalize(self, *args, **kwargs)
        start, end = _event(), _event(); start.record()
        result = original_finalize(self, *args, **kwargs); end.record()
        _record("combine", start, end, _tokens(args, kwargs, 2))
        return result

    def patched_moe(self: Any, *args: Any, **kwargs: Any) -> Any:
        mode = _control().get("mode"); layer = int(getattr(_CONTEXT, "layer", -1))
        if active_timing():
            start, end = _event(), _event(); start.record()
            result = original_moe(self, *args, **kwargs); end.record()
            _record("moe", start, end, _tokens(args, kwargs, 0))
        else:
            result = original_moe(self, *args, **kwargs)
        if layer == _target_layer() and _KERNEL is not None and _LATEST_PREP is not None:
            try:
                if mode == "pairwise":
                    _run_pairwise(); _run_policy(str(_control().get("modality", "prefill")))
                elif mode == "policy_decode" and int(_LATEST_PREP[0].shape[0]) <= 4:
                    _run_policy("decode")
            except BaseException as exc:
                _write(f"benchmark_error.rank{_ep_rank()}.json", {
                    "error": repr(exc), "traceback": traceback.format_exc(), "mode": mode,
                })
                raise
        return result

    def patched_logits(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_logits(self, *args, **kwargs)
        entry = _control(); wave = int(entry.get("wave", -1))
        if output is not None and entry.get("capture_logits") and wave not in _LOGITS and _ep_rank() in (0, 2):
            _LOGITS[wave] = output[-1].detach().to(torch.float16).cpu()
        return output

    def patched_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_execute(self, *args, **kwargs)
        if _control().get("flush"):
            _flush()
        return result

    Qwen3VLMoeForConditionalGeneration.forward = patched_outer
    Qwen3VLMoeForConditionalGeneration.compute_logits = patched_logits
    Qwen3MoeAttention.__init__ = patched_attn_init
    Qwen3MoeAttention.forward = patched_attn
    Qwen3MoeDecoderLayer.__init__ = patched_layer_init
    Qwen3MoeDecoderLayer.forward = patched_layer
    Qwen3MoeSparseMoeBlock.forward = patched_moe
    FusedMoEKernelModularImpl._prepare = patched_prepare
    FusedMoEKernelModularImpl._fused_experts = patched_experts
    FusedMoEKernelModularImpl._finalize = patched_finalize
    gmr.GPUModelRunner.execute_model = patched_execute
    atexit.register(_flush)
