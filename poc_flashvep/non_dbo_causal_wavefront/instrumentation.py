"""Single-owner, DBO-off split execution and bounded CUDA-event timing."""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


_LOCK = threading.Lock()
_CONTROL: dict[str, Any] = {}
_SIGNATURE: tuple[int, int, int] | None = None
_CONTEXT = threading.local()
_FORWARDS: list[dict[str, Any]] = []
_STAGES: list[dict[str, Any]] = []
_LOGITS: dict[int, torch.Tensor] = {}
_COUNTERS: Counter[str] = Counter()
_FLUSHED = False
_VLLM_CONFIG: Any = None


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _ep_rank() -> int:
    from vllm.distributed import get_ep_group

    return int(get_ep_group().rank_in_group)


def _refresh_control() -> dict[str, Any]:
    global _CONTROL, _SIGNATURE
    path = Path(os.environ["FLASHVEP_NON_DBO_WAVEFRONT_CONTROL"])
    stat = path.stat()
    signature = (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))
    if signature != _SIGNATURE:
        _CONTROL = json.loads(path.read_text(encoding="utf-8"))
        _SIGNATURE = signature
        _COUNTERS["control_file_reads"] += 1
    return _CONTROL


def _control() -> dict[str, Any]:
    _COUNTERS["cached_control_accesses"] += 1
    return _CONTROL


def _event(category: str) -> torch.cuda.Event:
    _COUNTERS[f"{category}_events_created"] += 1
    return torch.cuda.Event(enable_timing=True)


def _tokens(args: tuple[Any, ...], kwargs: dict[str, Any], index: int) -> int:
    value = kwargs.get("hidden_states")
    if value is None and len(args) > index:
        value = args[index]
    if value is None:
        value = kwargs.get("inputs_embeds")
    return int(value.shape[0]) if isinstance(value, torch.Tensor) else -1


def _target(entry: dict[str, Any], tokens: int) -> bool:
    return bool(entry) and tokens in {
        int(entry["prompt_tokens"]),
        int(entry["prefix_tokens"]),
        int(entry["tail_tokens"]),
    }


def _segment(tokens: int, entry: dict[str, Any]) -> str:
    explicit = getattr(_CONTEXT, "segment", None)
    if explicit is not None:
        return explicit
    return "full" if tokens == int(entry.get("prompt_tokens", -1)) else "unknown"


def _metadata(entry: dict[str, Any], layer: int, tokens: int) -> dict[str, Any]:
    return {
        "wave": int(entry["wave"]),
        "request_id": entry["request_id"],
        "phase": entry["phase"],
        "iteration": int(entry["iteration"]),
        "variant": entry["variant"],
        "layer": int(layer),
        "segment": _segment(tokens, entry),
        "tokens": int(tokens),
    }


def _record(
    stage: str,
    entry: dict[str, Any],
    layer: int,
    tokens: int,
    start: torch.cuda.Event,
    end: torch.cuda.Event,
    **extra: Any,
) -> None:
    with _LOCK:
        _STAGES.append(
            _metadata(entry, layer, tokens)
            | {"stage": stage, "start": start, "end": end}
            | extra
        )


def _resolve(rows: list[dict[str, Any]], origins: dict[int, torch.cuda.Event]):
    return [
        {key: value for key, value in row.items() if key not in ("start", "end")}
        | {
            "start_ms": float(origins[int(row["wave"])].elapsed_time(row["start"])),
            "end_ms": float(origins[int(row["wave"])].elapsed_time(row["end"])),
            "duration_ms": float(row["start"].elapsed_time(row["end"])),
        }
        for row in rows
    ]


def _write_logits() -> None:
    if not _LOGITS:
        return
    output = Path(os.environ["FLASHVEP_NON_DBO_WAVEFRONT_RAW"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"rank{_ep_rank()}.logits.npz",
        **{f"wave_{wave}": tensor.numpy() for wave, tensor in _LOGITS.items()},
    )


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED or not _FORWARDS:
        return
    _FLUSHED = True
    torch.cuda.synchronize()
    origins: dict[int, torch.cuda.Event] = {}
    for row in _FORWARDS:
        if row["segment"] in ("full", "prefix"):
            origins.setdefault(int(row["wave"]), row["start"])
    output = Path(os.environ["FLASHVEP_NON_DBO_WAVEFRONT_RAW"])
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "variant": os.environ["FLASHVEP_NON_DBO_WAVEFRONT_VARIANT"],
        "code_sha": os.environ["FLASHVEP_NON_DBO_WAVEFRONT_CODE_SHA"],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ep_rank": _ep_rank(),
        "dbo_configured": False,
        "host_owner_threads": 1,
        "forward_records": _resolve(_FORWARDS, origins),
        "stage_records": _resolve(_STAGES, origins),
        "counters": dict(_COUNTERS),
        "control_file_reads_inside_model_forward": 0,
    }
    (output / f"rank{_ep_rank()}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_logits()


def _slice(value: torch.Tensor | None, token_slice: slice, positions: bool = False):
    if value is None:
        return None
    if positions and value.ndim == 2:
        return value[:, token_slice]
    return value[token_slice]


def install() -> None:
    from vllm.forward_context import (
        DPMetadata,
        create_forward_context,
        get_forward_context,
        override_forward_context,
    )
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

    if getattr(Qwen3VLMoeForConditionalGeneration, "_non_dbo_stage_poc", False):
        return
    Qwen3VLMoeForConditionalGeneration._non_dbo_stage_poc = True

    variant = os.environ["FLASHVEP_NON_DBO_WAVEFRONT_VARIANT"]
    original_execute = gmr.GPUModelRunner.execute_model
    original_metadata_init = gmr.GPUModelRunner.initialize_metadata_builders
    original_slices = gmr.maybe_create_ubatch_slices
    original_outer = Qwen3VLMoeForConditionalGeneration.forward
    original_logits = Qwen3VLMoeForConditionalGeneration.compute_logits
    original_attn_init = Qwen3MoeAttention.__init__
    original_attn = Qwen3MoeAttention.forward
    original_layer_init = Qwen3MoeDecoderLayer.__init__
    original_layer = Qwen3MoeDecoderLayer.forward
    original_moe = Qwen3MoeSparseMoeBlock.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    def patched_metadata_init(
        self: Any, kv_cache_config: Any, kernel_block_sizes: list[int]
    ) -> None:
        original_metadata_init(self, kv_cache_config, kernel_block_sizes)
        if variant != "S":
            return
        if self.parallel_config.use_ubatching:
            raise RuntimeError("DBO unexpectedly enabled")
        # Explicit non-DBO prefix/tail slices need independent persistent
        # attention-builder buffers.  This does not enable vLLM ubatching or
        # its worker/thread execution; it only gives each sequential slice a
        # distinct metadata scope.
        for group_id, groups in enumerate(self.attn_groups):
            for group in groups:
                group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[group_id]
                    if group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=2,
                )
                if len(group.metadata_builders) != 2:
                    raise RuntimeError("failed to create two metadata scopes")
                _COUNTERS["non_dbo_metadata_builders"] += 2
        self.calculate_reorder_batch_threshold()

    def patched_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _VLLM_CONFIG
        _VLLM_CONFIG = self.vllm_config
        entry = _refresh_control()
        result = original_execute(self, *args, **kwargs)
        if entry.get("flush_after"):
            _flush()
        return result

    def patched_slices(
        should_ubatch: bool,
        num_scheduled_tokens: Any,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_ubatches: int,
        split_point: Any = None,
    ) -> Any:
        entry = _control()
        force = (
            variant == "S"
            and entry
            and int(num_tokens_padded) == int(entry["prompt_tokens"])
        )
        if force:
            _COUNTERS["forced_split_calls"] += 1
            return original_slices(
                True,
                num_scheduled_tokens,
                num_tokens_padded,
                num_reqs_padded,
                2,
                split_point=int(entry["prefix_tokens"]),
            )
        return original_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            num_ubatches,
            split_point=split_point,
        )

    def patched_outer(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        tokens = _tokens(args, kwargs, 1)
        if (
            variant != "S"
            or not _target(entry, tokens)
            or tokens != int(entry["prompt_tokens"])
        ):
            return original_outer(self, *args, **kwargs)
        parent = get_forward_context()
        slices = parent.ubatch_slices
        if (
            slices is None
            or len(slices) != 2
            or not isinstance(parent.attn_metadata, list)
        ):
            raise RuntimeError("forced split metadata was not constructed")
        if _VLLM_CONFIG.parallel_config.use_ubatching:
            raise RuntimeError("DBO unexpectedly enabled")

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        inputs_embeds = kwargs.get("inputs_embeds")
        outputs = []
        try:
            for index, ubatch_slice in enumerate(slices):
                token_slice = ubatch_slice.token_slice
                count = int(token_slice.stop - token_slice.start)
                across_dp = torch.tensor(
                    [count] * _VLLM_CONFIG.parallel_config.data_parallel_size,
                    dtype=torch.int32,
                    device="cpu",
                )
                dp_metadata = DPMetadata.make(
                    _VLLM_CONFIG.parallel_config, count, across_dp
                )
                slot_mapping = (
                    parent.slot_mapping[index]
                    if isinstance(parent.slot_mapping, list)
                    else None
                )
                context = create_forward_context(
                    parent.attn_metadata[index],
                    _VLLM_CONFIG,
                    dp_metadata=dp_metadata,
                    slot_mapping=slot_mapping,
                    additional_kwargs={"ubatch_token_slice": token_slice},
                    skip_compiled=True,
                )
                _CONTEXT.segment = "prefix" if index == 0 else "tail"
                with override_forward_context(context):
                    sliced_embeds = _slice(inputs_embeds, token_slice)
                    deepstack = (
                        self._get_deepstack_input_embeds(count)
                        if sliced_embeds is not None
                        else None
                    )
                    outputs.append(
                        self.language_model.model(
                            input_ids=_slice(input_ids, token_slice),
                            positions=_slice(positions, token_slice, positions=True),
                            intermediate_tensors=None,
                            inputs_embeds=sliced_embeds,
                            deepstack_input_embeds=deepstack,
                        )
                    )
        finally:
            _CONTEXT.segment = None
            self._finalize_ubatch_inputs()
        return torch.cat(outputs, dim=0)

    def patched_attn_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_attn_init(self, *args, **kwargs)
        self._non_dbo_layer = _layer(str(kwargs.get("prefix", "")))

    def patched_layer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_layer_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._non_dbo_layer = _layer(prefix)

    def patched_layer(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        tokens = _tokens(args, kwargs, 1)
        profile = bool(entry.get("stage_profile")) and _target(entry, tokens)
        old = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(self._non_dbo_layer)
        start = _event("stage") if profile else None
        if start is not None:
            start.record()
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            if start is not None:
                end = _event("stage")
                end.record()
                _record("decoder_layer", entry, self._non_dbo_layer, tokens, start, end)
            _CONTEXT.layer = old

    def timed(stage: str, original: Any, layer_from_self: bool = False):
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            entry = _control()
            token_index = 1 if stage in ("attention", "expert") else 0
            tokens = _tokens(args, kwargs, token_index)
            nested_modular_stage = stage in ("dispatch", "expert", "combine")
            profile = bool(entry.get("stage_profile")) and (
                _target(entry, tokens)
                or (
                    nested_modular_stage
                    and int(getattr(_CONTEXT, "layer", -1)) >= 0
                    and getattr(_CONTEXT, "segment", None) is not None
                )
            )
            if profile:
                _COUNTERS[f"{stage}_profiled_calls"] += 1
            start = _event("stage") if profile else None
            if start is not None:
                start.record()
            result = original(self, *args, **kwargs)
            if start is not None:
                end = _event("stage")
                end.record()
                layer = (
                    int(self._non_dbo_layer)
                    if layer_from_self
                    else int(getattr(_CONTEXT, "layer", -1))
                )
                _record(stage, entry, layer, tokens, start, end)
            return result

        return wrapper

    def patched_model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        tokens = _tokens(args, kwargs, 1)
        if not _target(entry, tokens):
            return original_outer(self, *args, **kwargs)
        segment = _segment(tokens, entry)
        start, end = _event("forward"), _event("forward")
        start.record()
        previous_segment = getattr(_CONTEXT, "segment", None)
        _CONTEXT.segment = segment
        try:
            result = patched_outer(self, *args, **kwargs)
            end.record()
        finally:
            _CONTEXT.segment = previous_segment
        with _LOCK:
            _FORWARDS.append(
                {
                    "wave": int(entry["wave"]),
                    "request_id": entry["request_id"],
                    "phase": entry["phase"],
                    "iteration": int(entry["iteration"]),
                    "variant": entry["variant"],
                    "segment": segment,
                    "tokens": tokens,
                    "start": start,
                    "end": end,
                }
            )
        return result

    def patched_compute_logits(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_logits(self, *args, **kwargs)
        entry = _control()
        if (
            result is not None
            and entry.get("phase") == "correctness"
            and _ep_rank() in (0, 2)
        ):
            _LOGITS[int(entry["wave"])] = result[-1].detach().to(torch.float16).cpu()
            if entry.get("flush_after"):
                _write_logits()
        return result

    gmr.GPUModelRunner.execute_model = patched_execute
    if variant == "S":
        gmr.GPUModelRunner.initialize_metadata_builders = patched_metadata_init
        gmr.maybe_create_ubatch_slices = patched_slices
    Qwen3VLMoeForConditionalGeneration.forward = patched_model_forward
    Qwen3VLMoeForConditionalGeneration.compute_logits = patched_compute_logits
    Qwen3MoeAttention.__init__ = patched_attn_init
    Qwen3MoeAttention.forward = timed("attention", original_attn, True)
    Qwen3MoeDecoderLayer.__init__ = patched_layer_init
    Qwen3MoeDecoderLayer.forward = patched_layer
    Qwen3MoeSparseMoeBlock.forward = timed("moe_total", original_moe)
    FusedMoEKernelModularImpl._prepare = timed("dispatch", original_prepare)
    FusedMoEKernelModularImpl._fused_experts = timed("expert", original_experts)
    FusedMoEKernelModularImpl._finalize = timed("combine", original_finalize)
    atexit.register(_flush)
