"""Worker-side live split, dependency events, and low-overhead timing."""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


_INSTALLED = False
_LOCK = threading.Lock()
_FORWARDS: list[dict[str, Any]] = []
_LAYERS: list[dict[str, Any]] = []
_PREFIX_ATTN_DONE: dict[tuple[int, int], torch.cuda.Event] = {}
_LOGITS: dict[int, torch.Tensor] = {}
_FLUSHED = False


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_LIVE_WAVEFRONT_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _ep_rank() -> int:
    from vllm.distributed import get_ep_group

    return int(get_ep_group().rank_in_group)


def _tensor_tokens(args: tuple[Any, ...], kwargs: dict[str, Any], index: int) -> int:
    value = kwargs.get("hidden_states")
    if value is None and len(args) > index:
        value = args[index]
    if value is None:
        value = kwargs.get("inputs_embeds")
    return int(value.shape[0]) if isinstance(value, torch.Tensor) else -1


def _is_target_prefill(entry: dict[str, Any], tokens: int, ubatch: int) -> bool:
    if not entry or entry.get("phase") == "flush":
        return False
    if os.environ["FLASHVEP_LIVE_WAVEFRONT_MODE"] == "wavefront" and ubatch in (0, 1):
        expected = int(entry["prefix_tokens"] if ubatch == 0 else entry["tail_tokens"])
    else:
        expected = int(entry["prompt_tokens"])
    return tokens == expected


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
        if row["ubatch_id"] in (-1, 0):
            origins.setdefault(row["wave"], row["start"])
    forward_rows = []
    for row in _FORWARDS:
        origin = origins[row["wave"]]
        forward_rows.append(
            {key: value for key, value in row.items() if key not in ("start", "end")}
            | {
                "start_ms": float(origin.elapsed_time(row["start"])),
                "end_ms": float(origin.elapsed_time(row["end"])),
                "duration_ms": float(row["start"].elapsed_time(row["end"])),
            }
        )
    layer_rows = []
    for row in _LAYERS:
        origin = origins[row["wave"]]
        layer_rows.append(
            {key: value for key, value in row.items() if key not in ("start", "end")}
            | {
                "start_ms": float(origin.elapsed_time(row["start"])),
                "end_ms": float(origin.elapsed_time(row["end"])),
                "duration_ms": float(row["start"].elapsed_time(row["end"])),
            }
        )
    output = Path(os.environ["FLASHVEP_LIVE_WAVEFRONT_RAW"])
    output.mkdir(parents=True, exist_ok=True)
    rank = _ep_rank()
    payload = {
        "status": "ok",
        "mode": os.environ["FLASHVEP_LIVE_WAVEFRONT_MODE"],
        "ep_rank": rank,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "forward_records": forward_rows,
        "layer_records": layer_rows,
        "dependency": "tail attention layer l waits for prefix attention layer l event",
        "separate_compute_streams": os.environ["FLASHVEP_LIVE_WAVEFRONT_MODE"]
        == "wavefront",
    }
    (output / f"rank{rank}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if _LOGITS:
        np.savez_compressed(
            output / f"rank{rank}.logits.npz",
            **{f"wave_{wave}": tensor.numpy() for wave, tensor in _LOGITS.items()},
        )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeAttention,
        Qwen3MoeDecoderLayer,
    )
    from vllm.model_executor.models.qwen3_vl_moe import (
        Qwen3VLMoeForConditionalGeneration,
    )
    from vllm.v1.worker import gpu_model_runner as gmr
    from vllm.v1.worker import gpu_ubatch_wrapper as guw
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_enabled

    if getattr(Qwen3VLMoeForConditionalGeneration, "_flashvep_live_wavefront", False):
        return
    Qwen3VLMoeForConditionalGeneration._flashvep_live_wavefront = True

    mode = os.environ["FLASHVEP_LIVE_WAVEFRONT_MODE"]
    original_maybe_slices = gmr.maybe_create_ubatch_slices
    original_make_metadata = UBatchWrapper._make_ubatch_metadata
    original_attn_init = Qwen3MoeAttention.__init__
    original_attn_forward = Qwen3MoeAttention.forward
    original_layer_init = Qwen3MoeDecoderLayer.__init__
    original_layer_forward = Qwen3MoeDecoderLayer.forward
    original_model_forward = Qwen3VLMoeForConditionalGeneration.forward
    original_compute_logits = Qwen3VLMoeForConditionalGeneration.compute_logits

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
            mode == "wavefront"
            and entry
            and should_ubatch
            and entry.get("phase") != "flush"
        ):
            split_point = int(entry["prefix_tokens"])
            if not 0 < split_point < int(num_tokens_padded):
                raise RuntimeError(
                    ("invalid live split", split_point, num_tokens_padded)
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
        if mode == "wavefront" and len(metadata) == 2:
            if not hasattr(self, "_flashvep_tail_stream"):
                self._flashvep_tail_stream = torch.cuda.Stream(device=self.device)
            metadata[1].context.compute_stream = self._flashvep_tail_stream
            metadata[1].context.current_stream = self._flashvep_tail_stream
        return metadata

    def patched_run_ubatches(self: Any, metadata: Any, model: Any) -> torch.Tensor:
        from vllm.forward_context import override_forward_context

        results: list[tuple[int, torch.Tensor, torch.cuda.Event]] = []
        errors: list[BaseException] = []

        @torch.inference_mode()
        def worker(item: Any) -> None:
            try:
                with item.context:
                    output = model(
                        input_ids=item.input_ids,
                        positions=item.positions,
                        intermediate_tensors=item.intermediate_tensors,
                        inputs_embeds=item.inputs_embeds,
                    )
                done = torch.cuda.Event()
                done.record(item.context.compute_stream)
                results.append((item.context.id, output, done))
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
        self._flashvep_wavefront_layer = _layer(prefix)

    def patched_attn_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        if mode != "wavefront" or not dbo_enabled():
            return original_attn_forward(self, *args, **kwargs)
        entry = _control()
        if not entry:
            return original_attn_forward(self, *args, **kwargs)
        wave = int(entry["wave"])
        layer = int(self._flashvep_wavefront_layer)
        ubatch = int(dbo_current_ubatch_id())
        if not _is_target_prefill(entry, _tensor_tokens(args, kwargs, 1), ubatch):
            return original_attn_forward(self, *args, **kwargs)
        key = (wave, layer)
        if ubatch == 1:
            with _LOCK:
                dependency = _PREFIX_ATTN_DONE.get(key)
            if dependency is None:
                raise RuntimeError(f"missing prefix attention event for {key}")
            torch.cuda.current_stream().wait_event(dependency)
        output = original_attn_forward(self, *args, **kwargs)
        if ubatch == 0:
            dependency = torch.cuda.Event()
            dependency.record(torch.cuda.current_stream())
            with _LOCK:
                _PREFIX_ATTN_DONE[key] = dependency
        return output

    def patched_layer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_layer_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_wavefront_layer = _layer(prefix)

    def patched_layer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        if not entry:
            return original_layer_forward(self, *args, **kwargs)
        ubatch = int(dbo_current_ubatch_id()) if dbo_enabled() else -1
        tokens = _tensor_tokens(args, kwargs, 1)
        trace = (
            bool(entry.get("timeline"))
            and entry.get("phase") == "measured"
            and _is_target_prefill(entry, tokens, ubatch)
        )
        start = torch.cuda.Event(enable_timing=True) if trace else None
        end = torch.cuda.Event(enable_timing=True) if trace else None
        if start is not None:
            start.record(torch.cuda.current_stream())
        output = original_layer_forward(self, *args, **kwargs)
        if end is not None:
            end.record(torch.cuda.current_stream())
            with _LOCK:
                _LAYERS.append(
                    {
                        "wave": int(entry["wave"]),
                        "request_id": entry["request_id"],
                        "layer": int(self._flashvep_wavefront_layer),
                        "ubatch_id": ubatch,
                        "start": start,
                        "end": end,
                    }
                )
        return output

    def patched_model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = _control()
        if not entry:
            return original_model_forward(self, *args, **kwargs)
        if entry.get("phase") == "flush":
            _flush()
            return original_model_forward(self, *args, **kwargs)
        ubatch = int(dbo_current_ubatch_id()) if dbo_enabled() else -1
        tokens = _tensor_tokens(args, kwargs, 1)
        if not _is_target_prefill(entry, tokens, ubatch):
            return original_model_forward(self, *args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        output = original_model_forward(self, *args, **kwargs)
        end.record(torch.cuda.current_stream())
        with _LOCK:
            _FORWARDS.append(
                {
                    "wave": int(entry["wave"]),
                    "request_id": entry["request_id"],
                    "phase": entry["phase"],
                    "iteration": int(entry["iteration"]),
                    "measured": bool(entry["measured"]),
                    "ubatch_id": ubatch,
                    "tokens": tokens,
                    "start": start,
                    "end": end,
                }
            )
        return output

    def patched_compute_logits(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_compute_logits(self, *args, **kwargs)
        entry = _control()
        if not entry:
            return output
        wave = int(entry.get("wave", -1))
        if (
            output is not None
            and entry.get("phase") == "correctness"
            and _ep_rank() in (0, 2)
            and wave not in _LOGITS
        ):
            _LOGITS[wave] = output[-1].detach().to(torch.float16).cpu()
        return output

    if mode == "wavefront":
        gmr.maybe_create_ubatch_slices = patched_maybe_slices
        guw.UBatchWrapper._make_ubatch_metadata = patched_make_metadata
        guw.UBatchWrapper._run_ubatches = patched_run_ubatches
        Qwen3MoeAttention.__init__ = patched_attn_init
        Qwen3MoeAttention.forward = patched_attn_forward
    Qwen3MoeDecoderLayer.__init__ = patched_layer_init
    Qwen3MoeDecoderLayer.forward = patched_layer_forward
    Qwen3VLMoeForConditionalGeneration.forward = patched_model_forward
    Qwen3VLMoeForConditionalGeneration.compute_logits = patched_compute_logits
    atexit.register(_flush)
