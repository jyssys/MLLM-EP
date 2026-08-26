"""Cached-control and bounded-event instrumentation for A0/A1/A2/C."""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


_INSTALLED = False
_LOCK = threading.Lock()
_CONTEXT = threading.local()
_CONTROL_ENTRY: dict[str, Any] = {}
_CONTROL_SIGNATURE: tuple[int, int, int] | None = None
_FORWARDS: list[dict[str, Any]] = []
_STAGES: list[dict[str, Any]] = []
_LOGITS: dict[int, torch.Tensor] = {}
_PREFIX_EVENTS: dict[int, torch.cuda.Event] = {}
_PREFIX_EVENT_WAVES: dict[int, int] = {}
_PROFILED_WAVES: set[int] = set()
_COUNTERS: Counter[str] = Counter()
_FLUSHED = False


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _ep_rank() -> int:
    from vllm.distributed import get_ep_group

    return int(get_ep_group().rank_in_group)


def _new_event(*, timing: bool, category: str) -> torch.cuda.Event:
    _COUNTERS[f"{category}_events_created"] += 1
    _COUNTERS["cuda_events_created_total"] += 1
    return torch.cuda.Event(enable_timing=timing)


def _refresh_control() -> dict[str, Any]:
    """Read an atomically replaced control file at most once per wave."""
    global _CONTROL_ENTRY, _CONTROL_SIGNATURE
    path = Path(os.environ["FLASHVEP_WAVEFRONT_FORENSICS_CONTROL"])
    _COUNTERS["control_stat_calls"] += 1
    if not path.exists():
        return _CONTROL_ENTRY
    stat = path.stat()
    signature = (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))
    if signature == _CONTROL_SIGNATURE:
        return _CONTROL_ENTRY
    entry = json.loads(path.read_text(encoding="utf-8"))
    with _LOCK:
        if signature != _CONTROL_SIGNATURE:
            _CONTROL_ENTRY = entry
            _CONTROL_SIGNATURE = signature
            _COUNTERS["control_file_reads"] += 1
    return _CONTROL_ENTRY


def _control() -> dict[str, Any]:
    """Hot-path access is memory-only; only GPUModelRunner refreshes it."""
    _COUNTERS["cached_control_accesses"] += 1
    return _CONTROL_ENTRY


def _ubatch_id() -> int:
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_enabled

    return int(dbo_current_ubatch_id()) if dbo_enabled() else -1


def _tensor_tokens(args: tuple[Any, ...], kwargs: dict[str, Any], index: int) -> int:
    value = kwargs.get("hidden_states")
    if value is None and len(args) > index:
        value = args[index]
    if value is None:
        value = kwargs.get("inputs_embeds")
    return int(value.shape[0]) if isinstance(value, torch.Tensor) else -1


def _expected_tokens(entry: dict[str, Any], ubatch: int) -> int:
    variant = os.environ["FLASHVEP_WAVEFRONT_FORENSICS_VARIANT"]
    total = int(entry["prompt_tokens"])
    if variant == "A0":
        return total
    if variant == "A1":
        split = total // 2
        return split if ubatch == 0 else total - split
    return int(entry["prefix_tokens"] if ubatch == 0 else entry["tail_tokens"])


def _is_target_prefill(entry: dict[str, Any], tokens: int, ubatch: int) -> bool:
    if not entry or entry.get("phase") == "flush":
        return False
    if ubatch not in ((-1,) if entry["variant"] == "A0" else (0, 1)):
        return False
    return tokens == _expected_tokens(entry, ubatch)


def _stage_profile(entry: dict[str, Any], tokens: int, ubatch: int) -> bool:
    return bool(entry.get("stage_profile")) and _is_target_prefill(
        entry, tokens, ubatch
    )


def _stage_metadata(entry: dict[str, Any], layer: int, ubatch: int) -> dict[str, Any]:
    return {
        "wave": int(entry["wave"]),
        "request_id": entry["request_id"],
        "variant": entry["variant"],
        "phase": entry["phase"],
        "iteration": int(entry["iteration"]),
        "layer": int(layer),
        "ubatch_id": int(ubatch),
    }


def _record_stage(
    stage: str,
    entry: dict[str, Any],
    layer: int,
    ubatch: int,
    start: torch.cuda.Event,
    end: torch.cuda.Event,
    **extra: Any,
) -> None:
    with _LOCK:
        _STAGES.append(
            _stage_metadata(entry, layer, ubatch)
            | {"stage": stage, "start": start, "end": end}
            | extra
        )


def _resolve_event_rows(
    rows: list[dict[str, Any]], origins: dict[int, torch.cuda.Event]
) -> list[dict[str, Any]]:
    resolved = []
    for row in rows:
        origin = origins[int(row["wave"])]
        resolved.append(
            {key: value for key, value in row.items() if key not in ("start", "end")}
            | {
                "start_ms": float(origin.elapsed_time(row["start"])),
                "end_ms": float(origin.elapsed_time(row["end"])),
                "duration_ms": float(row["start"].elapsed_time(row["end"])),
            }
        )
    return resolved


def _flush() -> None:
    global _FLUSHED
    if not _FORWARDS:
        return
    with _LOCK:
        if _FLUSHED:
            return
        _FLUSHED = True
    torch.cuda.synchronize()
    origins: dict[int, torch.cuda.Event] = {}
    for row in _FORWARDS:
        if int(row["ubatch_id"]) in (-1, 0):
            origins.setdefault(int(row["wave"]), row["start"])
    forward_rows = _resolve_event_rows(_FORWARDS, origins)
    stage_rows = _resolve_event_rows(_STAGES, origins)
    output = Path(os.environ["FLASHVEP_WAVEFRONT_FORENSICS_RAW"])
    output.mkdir(parents=True, exist_ok=True)
    rank = _ep_rank()
    dependency_live_before_cleanup = len(_PREFIX_EVENTS)
    _PREFIX_EVENTS.clear()
    _PREFIX_EVENT_WAVES.clear()
    payload = {
        "status": "ok",
        "code_sha": os.environ["FLASHVEP_WAVEFRONT_FORENSICS_CODE_SHA"],
        "variant": os.environ["FLASHVEP_WAVEFRONT_FORENSICS_VARIANT"],
        "ep_rank": rank,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "forward_records": forward_rows,
        "stage_records": stage_rows,
        "counters": dict(_COUNTERS),
        "control_file_reads_inside_model_forward": 0,
        "dependency_events_live_before_cleanup": dependency_live_before_cleanup,
        "dependency_events_live_after_cleanup": len(_PREFIX_EVENTS),
        "dependency_event_policy": "one reusable event per decoder layer; cleared at flush",
    }
    (output / f"rank{rank}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_logits(output, rank)


def _write_logits(output: Path | None = None, rank: int | None = None) -> None:
    if not _LOGITS:
        return
    output = output or Path(os.environ["FLASHVEP_WAVEFRONT_FORENSICS_RAW"])
    output.mkdir(parents=True, exist_ok=True)
    rank = _ep_rank() if rank is None else rank
    np.savez_compressed(
        output / f"rank{rank}.logits.npz",
        **{f"wave_{wave}": tensor.numpy() for wave, tensor in _LOGITS.items()},
    )


def _profile_execute(
    original: Any, self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
):
    entry = _control()
    wave = int(entry.get("wave", -1))
    if not entry.get("torch_profile") or wave in _PROFILED_WAVES:
        return original(self, *args, **kwargs)
    _PROFILED_WAVES.add(wave)
    output = Path(os.environ["FLASHVEP_WAVEFRONT_FORENSICS_RAW"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            result = original(self, *args, **kwargs)
        profiler.export_chrome_trace(
            str(output / f"torch_trace_rank{_ep_rank()}_wave{wave}.json")
        )
        _COUNTERS["torch_profiles_completed"] += 1
        return result
    except BaseException:
        (output / f"torch_profile_rank{_ep_rank()}_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        _COUNTERS["torch_profile_errors"] += 1
        raise


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeAttention,
        Qwen3MoeDecoderLayer,
        Qwen3MoeSparseMoeBlock,
    )
    from vllm.model_executor.models.qwen3_vl_moe import (
        Qwen3VLMoeForConditionalGeneration,
    )
    from vllm.v1.worker import gpu_model_runner as gmr
    from vllm.v1.worker import gpu_ubatch_wrapper as guw
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    if getattr(Qwen3VLMoeForConditionalGeneration, "_flashvep_forensics", False):
        return
    Qwen3VLMoeForConditionalGeneration._flashvep_forensics = True

    variant = os.environ["FLASHVEP_WAVEFRONT_FORENSICS_VARIANT"]
    original_execute_model = gmr.GPUModelRunner.execute_model
    original_maybe_slices = gmr.maybe_create_ubatch_slices
    original_make_metadata = UBatchWrapper._make_ubatch_metadata
    original_run_ubatches = UBatchWrapper._run_ubatches
    original_attn_init = Qwen3MoeAttention.__init__
    original_attn_forward = Qwen3MoeAttention.forward
    original_layer_init = Qwen3MoeDecoderLayer.__init__
    original_layer_forward = Qwen3MoeDecoderLayer.forward
    original_moe_forward = Qwen3MoeSparseMoeBlock.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize
    original_model_forward = Qwen3VLMoeForConditionalGeneration.forward
    original_compute_logits = Qwen3VLMoeForConditionalGeneration.compute_logits

    def patched_execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _refresh_control()
        result = _profile_execute(original_execute_model, self, args, kwargs)
        if entry.get("flush_after"):
            _flush()
        return result

    def patched_maybe_slices(
        should_ubatch: bool,
        num_scheduled_tokens: Any,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_ubatches: int,
        split_point: Any = None,
    ) -> Any:
        entry = _control()
        if (
            variant in ("A2", "C")
            and entry
            and should_ubatch
            and entry.get("phase") != "flush"
        ):
            split_point = int(entry["prefix_tokens"])
            if not 0 < split_point < int(num_tokens_padded):
                raise RuntimeError(
                    ("invalid forensic split", split_point, num_tokens_padded)
                )
        return original_maybe_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            num_ubatches,
            split_point=split_point,
        )

    def patched_make_metadata(self: Any, *args: Any, **kwargs: Any) -> Any:
        metadata = original_make_metadata(self, *args, **kwargs)
        if variant == "C" and len(metadata) == 2:
            if not hasattr(self, "_flashvep_forensic_tail_stream"):
                self._flashvep_forensic_tail_stream = torch.cuda.Stream(
                    device=self.device
                )
                _COUNTERS["tail_compute_streams_created"] += 1
            metadata[1].context.compute_stream = self._flashvep_forensic_tail_stream
            metadata[1].context.current_stream = self._flashvep_forensic_tail_stream
        return metadata

    def patched_run_ubatches(self: Any, metadata: Any, model: Any) -> torch.Tensor:
        if variant != "C":
            return original_run_ubatches(self, metadata, model)
        from vllm.forward_context import override_forward_context

        results: list[tuple[int, torch.Tensor, torch.cuda.Event]] = []
        errors: list[BaseException] = []

        @torch.inference_mode()
        def worker(item: Any) -> None:
            try:
                with item.context:
                    result = model(
                        input_ids=item.input_ids,
                        positions=item.positions,
                        intermediate_tensors=item.intermediate_tensors,
                        inputs_embeds=item.inputs_embeds,
                    )
                done = _new_event(timing=False, category="completion")
                done.record(item.context.compute_stream)
                results.append((item.context.id, result, done))
            except BaseException as error:
                errors.append(error)
                item.context.cpu_signal_event.set()

        with override_forward_context(None):
            threads = [
                threading.Thread(target=worker, args=(item,)) for item in metadata
            ]
            for thread in threads:
                thread.start()
            self.ready_barrier.wait()
            metadata[0].context.cpu_wait_event.set()
            for thread in threads:
                thread.join()
        if errors:
            raise errors[0]
        current = torch.cuda.current_stream()
        for _, _, done in results:
            current.wait_event(done)
        return torch.cat([value for _, value, _ in sorted(results)], dim=0)

    def patched_attn_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_attn_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        self._flashvep_forensic_layer = _layer(prefix)

    def patched_attn_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        layer = int(self._flashvep_forensic_layer)
        ubatch = _ubatch_id()
        tokens = _tensor_tokens(args, kwargs, 1)
        target = _is_target_prefill(entry, tokens, ubatch)
        profile_stage = _stage_profile(entry, tokens, ubatch)
        start = _new_event(timing=True, category="stage") if profile_stage else None
        if start is not None:
            start.record(torch.cuda.current_stream())
        if variant == "C" and target and ubatch == 1:
            with _LOCK:
                dependency = _PREFIX_EVENTS.get(layer)
                dependency_wave = _PREFIX_EVENT_WAVES.get(layer)
            if dependency is None or dependency_wave != int(entry["wave"]):
                raise RuntimeError(
                    f"missing prefix attention event for wave={entry['wave']} layer={layer}"
                )
            torch.cuda.current_stream().wait_event(dependency)
            _COUNTERS["dependency_event_waits"] += 1
            if profile_stage and start is not None:
                wait_end = _new_event(timing=True, category="stage")
                wait_end.record(torch.cuda.current_stream())
                _record_stage("causal_wait", entry, layer, ubatch, start, wait_end)
                start = wait_end
        result = original_attn_forward(self, *args, **kwargs)
        if variant == "C" and target and ubatch == 0:
            with _LOCK:
                dependency = _PREFIX_EVENTS.get(layer)
                if dependency is None:
                    dependency = _new_event(timing=False, category="dependency")
                    _PREFIX_EVENTS[layer] = dependency
                    _COUNTERS["dependency_events_max_live"] = max(
                        _COUNTERS["dependency_events_max_live"], len(_PREFIX_EVENTS)
                    )
                dependency.record(torch.cuda.current_stream())
                _PREFIX_EVENT_WAVES[layer] = int(entry["wave"])
                _COUNTERS["dependency_event_records"] += 1
        if profile_stage and start is not None:
            end = _new_event(timing=True, category="stage")
            end.record(torch.cuda.current_stream())
            _record_stage("attention", entry, layer, ubatch, start, end)
        return result

    def patched_layer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_layer_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_forensic_layer = _layer(prefix)

    def patched_layer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        layer = int(self._flashvep_forensic_layer)
        ubatch = _ubatch_id()
        tokens = _tensor_tokens(args, kwargs, 1)
        profile_stage = _stage_profile(entry, tokens, ubatch)
        previous_layer = getattr(_CONTEXT, "layer", -1)
        previous_entry = getattr(_CONTEXT, "entry", {})
        previous_ubatch = getattr(_CONTEXT, "ubatch", -1)
        _CONTEXT.layer = layer
        _CONTEXT.entry = entry if profile_stage else {}
        _CONTEXT.ubatch = ubatch
        start = _new_event(timing=True, category="stage") if profile_stage else None
        if start is not None:
            start.record(torch.cuda.current_stream())
        try:
            result = original_layer_forward(self, *args, **kwargs)
        finally:
            if start is not None:
                end = _new_event(timing=True, category="stage")
                end.record(torch.cuda.current_stream())
                _record_stage("decoder_layer", entry, layer, ubatch, start, end)
            _CONTEXT.layer = previous_layer
            _CONTEXT.entry = previous_entry
            _CONTEXT.ubatch = previous_ubatch
        return result

    def patched_moe_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = dict(getattr(_CONTEXT, "entry", {}))
        if not entry:
            return original_moe_forward(self, *args, **kwargs)
        layer = int(getattr(_CONTEXT, "layer", -1))
        ubatch = int(getattr(_CONTEXT, "ubatch", -1))
        start = _new_event(timing=True, category="stage")
        end = _new_event(timing=True, category="stage")
        start.record(torch.cuda.current_stream())
        result = original_moe_forward(self, *args, **kwargs)
        end.record(torch.cuda.current_stream())
        _record_stage("moe_total", entry, layer, ubatch, start, end)
        return result

    def patched_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = dict(getattr(_CONTEXT, "entry", {}))
        if not entry:
            return original_prepare(self, *args, **kwargs)
        layer = int(getattr(_CONTEXT, "layer", -1))
        ubatch = int(getattr(_CONTEXT, "ubatch", -1))
        compute = torch.cuda.current_stream()
        comm = self.prepare_finalize.buffer.get_comm_stream()
        compute_start = _new_event(timing=True, category="stage")
        compute_end = _new_event(timing=True, category="stage")
        comm_start = _new_event(timing=True, category="stage")
        comm_end = _new_event(timing=True, category="stage")
        compute_start.record(compute)
        comm_start.record(comm)
        result = original_prepare(self, *args, **kwargs)
        compute_end.record(compute)
        comm_end.record(comm)
        _record_stage(
            "dispatch_compute", entry, layer, ubatch, compute_start, compute_end
        )
        _record_stage("dispatch_comm", entry, layer, ubatch, comm_start, comm_end)
        return result

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = dict(getattr(_CONTEXT, "entry", {}))
        if not entry:
            return original_experts(self, *args, **kwargs)
        layer = int(getattr(_CONTEXT, "layer", -1))
        ubatch = int(getattr(_CONTEXT, "ubatch", -1))
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
        start = _new_event(timing=True, category="stage")
        end = _new_event(timing=True, category="stage")
        start.record(torch.cuda.current_stream())
        result = original_experts(self, *args, **kwargs)
        end.record(torch.cuda.current_stream())
        metadata = values.get("expert_tokens_meta")
        histogram = (
            [int(value) for value in metadata.expert_num_tokens_cpu.tolist()]
            if metadata is not None and metadata.expert_num_tokens_cpu is not None
            else []
        )
        _record_stage(
            "expert",
            entry,
            layer,
            ubatch,
            start,
            end,
            input_tokens=int(values["a1q"].shape[0]),
            expert_assignments=int(sum(histogram)),
            active_experts=int(sum(value > 0 for value in histogram)),
        )
        return result

    def patched_finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = dict(getattr(_CONTEXT, "entry", {}))
        if not entry:
            return original_finalize(self, *args, **kwargs)
        layer = int(getattr(_CONTEXT, "layer", -1))
        ubatch = int(getattr(_CONTEXT, "ubatch", -1))
        compute = torch.cuda.current_stream()
        comm = self.prepare_finalize.buffer.get_comm_stream()
        compute_start = _new_event(timing=True, category="stage")
        compute_end = _new_event(timing=True, category="stage")
        comm_start = _new_event(timing=True, category="stage")
        comm_end = _new_event(timing=True, category="stage")
        compute_start.record(compute)
        comm_start.record(comm)
        result = original_finalize(self, *args, **kwargs)
        compute_end.record(compute)
        comm_end.record(comm)
        _record_stage(
            "combine_compute", entry, layer, ubatch, compute_start, compute_end
        )
        _record_stage("combine_comm", entry, layer, ubatch, comm_start, comm_end)
        return result

    def patched_model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        ubatch = _ubatch_id()
        tokens = _tensor_tokens(args, kwargs, 1)
        if not _is_target_prefill(entry, tokens, ubatch):
            return original_model_forward(self, *args, **kwargs)
        start = _new_event(timing=True, category="forward")
        end = _new_event(timing=True, category="forward")
        start.record(torch.cuda.current_stream())
        result = original_model_forward(self, *args, **kwargs)
        end.record(torch.cuda.current_stream())
        with _LOCK:
            _FORWARDS.append(
                {
                    "wave": int(entry["wave"]),
                    "request_id": entry["request_id"],
                    "variant": entry["variant"],
                    "phase": entry["phase"],
                    "iteration": int(entry["iteration"]),
                    "measured": bool(entry["measured"]),
                    "ubatch_id": ubatch,
                    "tokens": tokens,
                    "start": start,
                    "end": end,
                }
            )
        return result

    def patched_compute_logits(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_compute_logits(self, *args, **kwargs)
        entry = _control()
        wave = int(entry.get("wave", -1))
        if (
            result is not None
            and entry.get("phase") == "correctness"
            and _ep_rank() in (0, 2)
            and wave not in _LOGITS
        ):
            _LOGITS[wave] = result[-1].detach().to(torch.float16).cpu()
            if entry.get("flush_after"):
                _write_logits()
        return result

    gmr.GPUModelRunner.execute_model = patched_execute_model
    if variant in ("A2", "C"):
        gmr.maybe_create_ubatch_slices = patched_maybe_slices
    if variant == "C":
        guw.UBatchWrapper._make_ubatch_metadata = patched_make_metadata
        guw.UBatchWrapper._run_ubatches = patched_run_ubatches
    Qwen3MoeAttention.__init__ = patched_attn_init
    Qwen3MoeAttention.forward = patched_attn_forward
    Qwen3MoeDecoderLayer.__init__ = patched_layer_init
    Qwen3MoeDecoderLayer.forward = patched_layer_forward
    Qwen3MoeSparseMoeBlock.forward = patched_moe_forward
    FusedMoEKernelModularImpl._prepare = patched_prepare
    FusedMoEKernelModularImpl._fused_experts = patched_experts
    FusedMoEKernelModularImpl._finalize = patched_finalize
    Qwen3VLMoeForConditionalGeneration.forward = patched_model_forward
    Qwen3VLMoeForConditionalGeneration.compute_logits = patched_compute_logits
    atexit.register(_flush)
