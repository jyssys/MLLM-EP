"""Read-only vLLM worker hook for actual vision-block/DeepEP overlap.

The hook is deliberately one-shot: the first real Qwen3-VL vision request
captures a Vision Transformer block input, and the first layer-24 expert call
replays an existing exact route through the model's live DeepEP buffer.  No
router, placement, scheduler, or model output is modified.
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from poc_flashvep.deepep_revalidation.operator_replay import (
    ExpertSpec,
    _correctness,
    _event,
    _elapsed,
    _micro_states,
    _stats,
    _values_env,
)
from poc_flashvep.offline_wavefront.workload_builder import build_repeated_workload

_INSTALLED = False
_RAN: set[int] = set()
_VISION_LOCK = threading.Lock()
_VISION_BLOCK: Any = None
_VISION_INPUT: torch.Tensor | None = None
_VISION_ARGS: tuple[Any, ...] = ()
_VISION_KWARGS: dict[str, Any] = {}


def _write(name: str, value: Any) -> None:
    out = Path(os.environ["FLASHVEP_VISION_OVERLAP_RESULT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(value, indent=2, default=_json) + "\n", encoding="utf-8")


def _json(value: Any) -> Any:
    if isinstance(value, (torch.Tensor,)):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    return repr(value)


def _sync(ep: Any) -> None:
    # EP device_group is the only collective used by this diagnostic.  The
    # barrier is outside timed CUDA work and is not a production mechanism.
    dist.barrier(group=ep.device_group)
    torch.cuda.synchronize()


def _launch_encoder(stream: torch.cuda.Stream, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
    if _VISION_BLOCK is None or _VISION_INPUT is None:
        raise RuntimeError("vision block activation was not captured")
    with torch.cuda.stream(stream):
        start.record(stream)
        torch.cuda.nvtx.range_push("FLASHVEP_ENCODER_BLOCK")
        with torch.inference_mode():
            _VISION_BLOCK(_VISION_INPUT, *_VISION_ARGS, **_VISION_KWARGS)
        torch.cuda.nvtx.range_pop()
        end.record(stream)


def _wait_event(event: Any, stream: torch.cuda.Stream) -> None:
    with torch.cuda.stream(stream):
        event.current_stream_wait()


def _run_phase(
    phase: str,
    encoder: bool,
    kernel: Any,
    original_experts: Any,
    buffer: Any,
    spec: ExpertSpec,
    workload: Any,
    rank: int,
    ep_size: int,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure a real DeepEP phase alone and with one real encoder block."""
    import deep_ep

    ep = buffer
    comm_stream = ep.get_comm_stream()
    expert_stream = torch.cuda.Stream()
    enc_stream = torch.cuda.Stream()
    samples: list[dict[str, float]] = []

    # Establish a same-process encoder-only reference on the same CUDA stream
    # used by the concurrent trials.  This avoids treating the encoder time
    # observed under contention as its standalone cost.
    encoder_alone: list[float] = []
    for index in range(warmups + iterations):
        _sync(type("E", (), {"device_group": dist.group.WORLD})())
        # The hook runs while the live model is paused inside its first MoE
        # invocation.  Drain any pre-existing model work before establishing
        # an encoder-only reference; otherwise this diagnostic would fold
        # unrelated attention/vision tail work into the encoder event.
        torch.cuda.synchronize()
        enc_start, enc_end = _event(), _event()
        _launch_encoder(enc_stream, enc_start, enc_end)
        enc_end.synchronize()
        torch.cuda.synchronize()
        if index >= warmups:
            encoder_alone.append(_elapsed(enc_start, enc_end))

    def cycle(with_encoder: bool, measure: bool) -> dict[str, float]:
        state = _micro_states(workload, rank, ep_size, 1)[0]
        # Keep each paired trial free of work left by the live model or the
        # preceding cleanup collective.  This is outside the timed interval.
        torch.cuda.synchronize()
        origin = _event()
        origin.record(torch.cuda.current_stream())
        enc_start, enc_end = _event(), _event()
        if with_encoder and phase in ("dispatch", "expert"):
            _launch_encoder(enc_stream, enc_start, enc_end)
        if phase == "dispatch":
            torch.cuda.nvtx.range_push("FLASHVEP_DEEPEP_DISPATCH")
            # DeepEP's Python call must be issued from a compute/default
            # stream.  The C++ implementation internally launches on its
            # dedicated communication stream and asserts that the caller
            # stream differs from that stream.  Calling it inside
            # ``with torch.cuda.stream(comm_stream)`` therefore triggers the
            # event.hpp same-stream assertion.
            start_comm = _event(); start_comm.record(comm_stream)
            num_rank, num_rdma, num_exp, in_rank, layout_event = ep.get_dispatch_layout(
                state.ids, spec.global_num_experts, async_finish=True,
                allocate_on_comm_stream=False,
            )
            dispatched = ep.dispatch(
                x=state.hidden, handle=None, num_tokens_per_rank=num_rank,
                num_tokens_per_rdma_rank=num_rdma, is_token_in_rank=in_rank,
                num_tokens_per_expert=num_exp,
                topk_idx=state.ids.to(deep_ep.topk_idx_t), topk_weights=state.weights,
                expert_alignment=1,
                config=deep_ep.Buffer.get_dispatch_config(dist.get_world_size()),
                previous_event=layout_event, async_finish=True,
                allocate_on_comm_stream=False,
            )
            (state.recv_hidden, state.recv_ids, state.recv_weights,
             state.recv_counts, state.handle, state.dispatch_event) = dispatched
            end_comm = _event(); end_comm.record(comm_stream)
            state.dispatch_event.current_stream_wait()
            done = _event(); done.record(torch.cuda.current_stream())
            torch.cuda.nvtx.range_pop()
            done.synchronize()
            # Complete the handle with the dispatched activation before the
            # next iteration.  This cleanup collective is outside the timed
            # dispatch interval and prevents outstanding DeepEP state from
            # influencing the following sample.
            _, _, cleanup_event = ep.combine(
                x=state.recv_hidden, handle=state.handle, topk_weights=None,
                config=deep_ep.Buffer.get_combine_config(dist.get_world_size()),
                async_finish=True, allocate_on_comm_stream=False,
            )
            cleanup_event.current_stream_wait()
            cleanup_done = _event(); cleanup_done.record(torch.cuda.current_stream())
            cleanup_done.synchronize()
        else:
            # Prepare a valid combine handle/output.  Setup is intentionally
            # outside the timed combine interval.
            num_rank, num_rdma, num_exp, in_rank, layout_event = ep.get_dispatch_layout(
                state.ids, spec.global_num_experts, async_finish=True,
                allocate_on_comm_stream=False,
            )
            (state.recv_hidden, state.recv_ids, state.recv_weights,
             state.recv_counts, state.handle, state.dispatch_event) = ep.dispatch(
                x=state.hidden, handle=None, num_tokens_per_rank=num_rank,
                num_tokens_per_rdma_rank=num_rdma, is_token_in_rank=in_rank,
                num_tokens_per_expert=num_exp,
                topk_idx=state.ids.to(deep_ep.topk_idx_t), topk_weights=state.weights,
                expert_alignment=1,
                config=deep_ep.Buffer.get_dispatch_config(dist.get_world_size()),
                previous_event=layout_event, async_finish=True,
                allocate_on_comm_stream=False,
            )
            state.dispatch_event.current_stream_wait()
            with torch.cuda.stream(expert_stream):
                state.dispatch_event.current_stream_wait()
                from vllm.model_executor.layers.fused_moe.modular_kernel import ExpertTokensMetadata
                from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate
                offset = rank * spec.local_num_experts
                global_ids = torch.where(state.recv_ids == -1, spec.global_num_experts - 1 if offset == 0 else 0, state.recv_ids + offset)
                meta = ExpertTokensMetadata.make_from_list(state.recv_counts, device=state.recv_hidden.device)
                raw = original_experts(kernel, spec.in_dtype, state.recv_hidden, None, spec.w1, spec.w2,
                                       state.recv_weights, global_ids, spec.activation,
                                       spec.global_num_experts, spec.local_num_experts, spec.expert_map,
                                       spec.apply_router_weight_on_input, meta)
                wr = kernel.fused_experts.finalize_weight_and_reduce_impl()
                if isinstance(wr, TopKWeightAndReduceDelegate): wr = TopKWeightAndReduceContiguous()
                state.expert_output = wr.apply(output=None, fused_expert_output=raw,
                    topk_weights=state.recv_weights, topk_ids=global_ids,
                    apply_router_weight_on_input=spec.apply_router_weight_on_input).clone()
                state.expert_event = deep_ep.Buffer.capture()
            torch.cuda.synchronize()
            if phase == "expert":
                # Diagnostic negative control: pair the real encoder block
                # with the actual Triton expert execution.  The combine below
                # is cleanup only and is excluded from this phase's timing.
                done = _event(); done.record(torch.cuda.current_stream())
                done.synchronize()
                start_comm = _event(); start_comm.record(torch.cuda.current_stream())
                _, _, cleanup_event = ep.combine(
                    x=state.expert_output, handle=state.handle, topk_weights=None,
                    config=deep_ep.Buffer.get_combine_config(dist.get_world_size()),
                    async_finish=True, allocate_on_comm_stream=False,
                )
                cleanup_event.current_stream_wait()
                end_comm = _event(); end_comm.record(torch.cuda.current_stream())
                end_comm.synchronize()
            else:
                if with_encoder:
                    # For combine, the encoder is intentionally launched
                    # after setup dispatch/expert work so its useful work is
                    # paired with the actual combine communication only.
                    _launch_encoder(enc_stream, enc_start, enc_end)
                torch.cuda.nvtx.range_push("FLASHVEP_DEEPEP_COMBINE")
                state.expert_event.current_stream_wait()
                start_comm = _event(); start_comm.record(comm_stream)
                _, _, state.combine_event = ep.combine(
                    x=state.expert_output, handle=state.handle, topk_weights=None,
                    config=deep_ep.Buffer.get_combine_config(dist.get_world_size()),
                    async_finish=True, allocate_on_comm_stream=False,
                )
                end_comm = _event(); end_comm.record(comm_stream)
                state.combine_event.current_stream_wait()
                done = _event(); done.record(torch.cuda.current_stream())
                torch.cuda.nvtx.range_pop()
                done.synchronize()
        if with_encoder:
            enc_end.synchronize()
        # Cleanup any outstanding work before the next collective.
        torch.cuda.synchronize()
        value = {"phase_ms": _elapsed(origin, done),
                 "comm_ms": _elapsed(start_comm, end_comm)}
        if with_encoder:
            value["encoder_ms"] = _elapsed(enc_start, enc_end)
            value["concurrent_wall_ms"] = _elapsed(origin, done)
        return value

    for _ in range(warmups):
        _sync(type("E", (), {"device_group": dist.group.WORLD})())
        cycle(False, False)
        _sync(type("E", (), {"device_group": dist.group.WORLD})())
        cycle(True, False)
    for index in range(iterations):
        # Interleave alone/concurrent in a deterministic pattern and barrier
        # every cycle so no rank launches a second DeepEP collective early.
        _sync(type("E", (), {"device_group": dist.group.WORLD})())
        alone = cycle(False, True)
        _sync(type("E", (), {"device_group": dist.group.WORLD})())
        concurrent = cycle(True, True)
        samples.append({"alone_ms": alone["phase_ms"], "concurrent_ms": concurrent["phase_ms"],
                        "alone_comm_ms": alone["comm_ms"],
                        "concurrent_comm_ms": concurrent["comm_ms"],
                        "encoder_ms": concurrent.get("encoder_ms", 0.0)})
    return {"phase": phase, "warmups": warmups, "iterations": iterations,
            "samples": samples, "alone_stats": _stats([x["alone_ms"] for x in samples]),
            "concurrent_stats": _stats([x["concurrent_ms"] for x in samples]),
            "alone_comm_stats": _stats([x["alone_comm_ms"] for x in samples]),
            "concurrent_comm_stats": _stats([x["concurrent_comm_ms"] for x in samples]),
            "encoder_overlap_stats": _stats([x["encoder_ms"] for x in samples]),
            "encoder_alone_ms": encoder_alone,
            "encoder_alone_stats": _stats(encoder_alone)}


def _run_benchmark(kernel: Any, original_experts: Any, spec: ExpertSpec) -> dict[str, Any]:
    from vllm.distributed import get_ep_group
    ep_group = get_ep_group(); rank = int(ep_group.rank_in_group); ep_size = int(ep_group.world_size)
    buffer = kernel.prepare_finalize.buffer
    capture_path = Path(os.environ["FLASHVEP_DEEPEP_CAPTURE_PATH"])
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    base_hidden = capture["post_attention_hidden"].to(spec.w1.device)
    base_ids = capture["topk_expert_ids"].to(spec.w1.device)
    base_weights = capture["topk_weights"].to(spec.w1.device)
    batch = int(os.environ.get("FLASHVEP_VISION_OVERLAP_BATCH", "4"))
    workload = build_repeated_workload(base_hidden, base_ids, base_weights, batch)
    warmups = int(os.environ.get("FLASHVEP_VISION_OVERLAP_WARMUPS", "10"))
    iterations = int(os.environ.get("FLASHVEP_VISION_OVERLAP_ITERATIONS", "30"))
    result: dict[str, Any] = {"rank": rank, "ep_size": ep_size, "batch": batch,
                              "capture_metadata": capture["metadata"],
                              "vision_activation": {"shape": list(_VISION_INPUT.shape) if _VISION_INPUT is not None else None,
                                                     "dtype": str(_VISION_INPUT.dtype) if _VISION_INPUT is not None else None},
                              "encoder_block": type(_VISION_BLOCK).__name__ if _VISION_BLOCK is not None else None}
    for phase in ("dispatch", "combine", "expert"):
        result[phase] = _run_phase(phase, True, kernel, original_experts, buffer, spec, workload,
                                   rank, ep_size, warmups, iterations)
    _write(f"overlap_rank{rank}.json", result)
    return result


def install() -> None:
    global _INSTALLED
    if _INSTALLED: return
    _INSTALLED = True
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionBlock
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer as TextQwen3MoeDecoderLayer

    original_block = Qwen3_VisionBlock.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_init_text = TextQwen3MoeDecoderLayer.__init__
    original_forward_text = TextQwen3MoeDecoderLayer.forward

    def block_forward(self: Any, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        global _VISION_BLOCK, _VISION_INPUT, _VISION_ARGS, _VISION_KWARGS
        if _VISION_INPUT is None:
            with _VISION_LOCK:
                if _VISION_INPUT is None:
                    _VISION_BLOCK = self
                    _VISION_INPUT = x.detach().clone()
                    _VISION_ARGS = tuple(args)
                    _VISION_KWARGS = dict(kwargs)
        return original_block(self, x, *args, **kwargs)

    def _init_layer(original: Any, self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        import re
        m = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
        self._flashvep_overlap_layer = int(m.group(1)) if m else -1

    def init_layer_text(self: Any, *args: Any, **kwargs: Any) -> None:
        _init_layer(original_init_text, self, *args, **kwargs)

    def forward_layer_text(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original_forward_text(self, *args, **kwargs)

    def experts(self: Any, in_dtype: torch.dtype, a1q: torch.Tensor, a1q_scale: Any,
                w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.Tensor, activation: Any, global_num_experts: int,
                local_num_experts: int, expert_map: Any, apply_router_weight_on_input: bool,
                expert_tokens_meta: Any) -> torch.Tensor:
        from vllm.distributed import get_ep_group
        ep = get_ep_group(); rank = int(ep.rank_in_group)
        # The first real expert invocation is sufficient: the vision block
        # activation is captured before language-model MoE begins, and using
        # the first call avoids relying on model-specific layer class aliases.
        if rank not in _RAN and _VISION_INPUT is not None:
            _RAN.add(rank)
            spec = ExpertSpec(in_dtype, w1, w2, activation, int(global_num_experts), int(local_num_experts), expert_map, bool(apply_router_weight_on_input))
            try:
                _run_benchmark(self, original_experts, spec)
            except BaseException as exc:
                _write(f"overlap_rank{rank}.error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
                raise
        return original_experts(self, in_dtype, a1q, a1q_scale, w1, w2, topk_weights, topk_ids,
                                activation, global_num_experts, local_num_experts, expert_map,
                                apply_router_weight_on_input, expert_tokens_meta)

    Qwen3_VisionBlock.forward = block_forward
    TextQwen3MoeDecoderLayer.__init__ = init_layer_text
    TextQwen3MoeDecoderLayer.forward = forward_layer_text
    FusedMoEKernelModularImpl._fused_experts = experts
