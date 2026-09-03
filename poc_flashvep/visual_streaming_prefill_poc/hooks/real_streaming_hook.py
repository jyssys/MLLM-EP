"""Minimal worker-local image-ready handoff for the 2-image PoC.

This is deliberately a fixed experimental path.  It does not change model
math, routing, placement, token order, or the stock path for ordinary
requests.  For a request whose control id starts with ``streaming_`` it
encodes image 1 on the normal stream, launches image 2 on a side CUDA stream,
and inserts a CUDA event wait only when the scheduler reaches image 2's token
range.  The same full request and stock vLLM KV-cache/chunking remain in use.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from . import streaming_hook as base

_INSTALLED = False


def _root() -> Path | None:
    value = os.environ.get("FLASHVEP_STREAMING_RESULT_DIR")
    return Path(value) if value else None


def _log(kind: str, row: dict[str, Any]) -> None:
    root = _root()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"real_streaming_pid{os.getpid()}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **row}, default=str) + "\n")
    except Exception:
        pass


def _one_item(model: Any, item: Any, device: Any, pin_memory: bool) -> torch.Tensor:
    from vllm.multimodal.utils import group_and_batch_mm_kwargs

    groups = list(group_and_batch_mm_kwargs(
        [item], device=device, pin_memory=pin_memory))
    if len(groups) != 1:
        raise RuntimeError(f"unexpected one-image groups: {len(groups)}")
    _, num_items, batch = groups[0]
    if num_items != 1:
        raise RuntimeError(f"unexpected one-image count: {num_items}")
    result = model.embed_multimodal(**batch)
    if result is None or len(result) != 1:
        raise RuntimeError(f"unexpected one-image output: {type(result)}")
    return result[0]


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_execute = GPUModelRunner._execute_mm_encoder
    original_gather = GPUModelRunner._gather_mm_embeddings

    def execute(self: Any, scheduler_output: Any) -> list[torch.Tensor]:
        active = base._active_id()
        if active.startswith("streaming_"):
            _log("execute_enter", {"active_id": active, "pid": os.getpid()})
        if not active.startswith("streaming_"):
            return original_execute(self, scheduler_output)

        hashes, mm_kwargs, refs = self._batch_mm_inputs_from_scheduler(scheduler_output)
        if not hashes:
            return original_execute(self, scheduler_output)

        # vLLM's encoder budget may schedule the two images in separate
        # iterations.  On the first (image-1) call, recover the second
        # feature from the same request state and launch it on the side
        # stream before the first LM chunk starts.  On the later image-2
        # call, simply return the already-published cache entry.
        if len(hashes) == 1 and len(mm_kwargs) == 1 and len(refs) == 1:
            req_id = refs[0][0]
            pending = getattr(self, "_flashvep_real_pending", None)
            # The scheduler eventually emits a second encoder-input call for
            # image 2.  It is already running (or complete) on the side
            # stream, so return its cached tensor and let gather() perform the
            # CUDA-event wait before consumption.
            if pending is not None and pending["hash"] == hashes[0]:
                return [self.encoder_cache[hashes[0]]]
            req_state = self.requests.get(req_id)
            features = list(getattr(req_state, "mm_features", ())) if req_state is not None else []
            if len(features) == 2:
                first_hash = hashes[0]
                second_feature = next(
                    (feature for feature in features
                     if getattr(feature, "identifier", None) != first_hash),
                    None,
                )
                if second_feature is not None:
                    second_hash = second_feature.identifier
                    if pending is not None and pending["hash"] == second_hash:
                        _log("execute_cached", {"active_id": active, "hash": second_hash,
                                                 "pid": os.getpid()})
                        return [self.encoder_cache[second_hash]]

                    second_item = (second_feature.modality, second_feature.data)
                    return _launch_pair(
                        self, active, first_hash, mm_kwargs[0], refs[0],
                        second_hash, second_item, (req_id, second_feature.mm_position),
                    )

            # If the request shape is not exactly the fixed two-image case,
            # retain stock behavior rather than changing encoder semantics.
            return original_execute(self, scheduler_output)

        if len(hashes) != 2 or len(mm_kwargs) != 2 or len(refs) != 2:
            return original_execute(self, scheduler_output)

        pending = getattr(self, "_flashvep_real_pending", None)
        if pending is not None and pending["hash"] == hashes[0]:
            # The second image was proactively launched during image-1's
            # scheduled encoder call.  Its readiness is enforced by gather.
            return [self.encoder_cache[hashes[0]]]

        result = _launch_pair(self, active, hashes[0], mm_kwargs[0], refs[0],
                            hashes[1], mm_kwargs[1], refs[1])
        _log("execute_exit", {"active_id": active, "count": len(result),
                               "pid": os.getpid()})
        return result

    def _launch_pair(self: Any, active: str, first_hash: str,
                     first_item: Any, first_ref: Any, second_hash: str,
                     second_item: Any, second_ref: Any) -> list[torch.Tensor]:
        """Encode image 1 synchronously and enqueue image 2 asynchronously."""

        model = self.model
        device = self.device
        # Qwen's multimodal path has no LoRA in this validated setup.  Run the
        # first image on the current stream and time it with a CUDA event.
        e1_start = torch.cuda.Event(enable_timing=True)
        e1_end = torch.cuda.Event(enable_timing=True)
        e1_start.record(torch.cuda.current_stream())
        first = _one_item(model, first_item, device, self.pin_memory)
        e1_end.record(torch.cuda.current_stream())
        e1_end.synchronize()
        _log("encoder", {"active_id": active, "image": 1,
                          "duration_ms": float(e1_start.elapsed_time(e1_end)),
                          "mode": "streaming", "pid": os.getpid()})

        # Launch the second real image encoder on a separate stream.  The
        # inner vision hook is instructed not to synchronize or CPU-copy here.
        side = torch.cuda.Stream(device=device)
        e2_start = torch.cuda.Event(enable_timing=True)
        e2_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(side):
            e2_start.record(side)
            base._ASYNC_VISION = True
            try:
                second = _one_item(model, second_item, device, self.pin_memory)
            finally:
                base._ASYNC_VISION = False
            e2_end.record(side)

        # Publish both tensors immediately, but never consume image 2 without
        # the event wait installed by gather below.
        self.encoder_cache[first_hash] = first
        self.encoder_cache[second_hash] = second
        self._flashvep_real_pending = {
            "hash": second_hash, "event": e2_end, "stream": side,
            "start": e2_start, "active_id": active,
            "image_start": second_ref[1].offset,
            "image_end": second_ref[1].offset + second_ref[1].length,
            "logged": False,
        }
        _log("handoff", {"active_id": active, "image": 1,
                          "image2_start": second_ref[1].offset,
                          "event_recorded": True, "pid": os.getpid()})
        # The caller only uses this return value for optional connector hooks;
        # match stock's list shape so no downstream behavior changes.
        return [first]

    def gather(self: Any, scheduler_output: Any, shift_computed_tokens: int = 0):
        pending = getattr(self, "_flashvep_real_pending", None)
        active = base._active_id()
        if active.startswith("streaming_"):
            _log("gather_enter", {
                "active_id": active,
                "pending": pending is not None,
                "total": int(getattr(scheduler_output, "total_num_scheduled_tokens", 0)),
                "pid": os.getpid(),
            })
        if pending is not None:
            for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
                req = self.requests.get(req_id)
                if req is None:
                    continue
                start = req.num_computed_tokens + shift_computed_tokens
                end = start + int(scheduled)
                if end > pending["image_start"] and start < pending["image_end"]:
                    current = torch.cuda.current_stream()
                    _log("handoff_wait_start", {
                        "active_id": pending["active_id"], "start": start,
                        "end": end, "image_start": pending["image_start"],
                        "image_end": pending["image_end"], "pid": os.getpid(),
                    })
                    current.wait_event(pending["event"])
                    pending["event"].synchronize()
                    _log("handoff_wait_done", {
                        "active_id": pending["active_id"], "pid": os.getpid(),
                    })
                    if not pending["logged"]:
                        _log("encoder", {"active_id": pending["active_id"],
                                          "image": 2,
                                          "duration_ms": float(pending["start"].elapsed_time(pending["event"])),
                                          "mode": "streaming", "pid": os.getpid()})
                        pending["logged"] = True
        result = original_gather(self, scheduler_output, shift_computed_tokens)
        if active.startswith("streaming_"):
            _log("gather_exit", {"active_id": active, "pid": os.getpid()})
        return result

    execute._flashvep_real_streaming_wrapped = True  # type: ignore[attr-defined]
    gather._flashvep_real_streaming_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner._execute_mm_encoder = execute
    GPUModelRunner._gather_mm_embeddings = gather
