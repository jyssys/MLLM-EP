"""Read-only CUDA-event instrumentation for DP-to-EP synchronization.

This layer composes the validated DeepEP hook and adds (a) layer entry/MoE
entry spans, (b) a diagnostic delay on one DP worker, and (c) event-wait spans
around DeepEP EventOverlap waits.  It never changes routes, tensors, or
scheduler decisions.  All event objects are resolved once at the flush wave.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from poc_flashvep.ep4_serving_straggler_regime import live_instrumentation as base

_INSTALLED = False
_AUX: list[dict[str, Any]] = []
_LOCK = threading.Lock()


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _layer(obj: Any) -> int:
    return int(getattr(obj, "_flashvep_serving_layer", -1))


def _delay_ms() -> float:
    # For a positive-control sweep the host publishes the delay in the same
    # request-boundary control record used by the validated hook.  Falling
    # back to the environment preserves the single-delay CLI behavior.
    entry = getattr(base._CONTEXT, "entry", {})
    if isinstance(entry, dict) and "delay_ms" in entry:
        try:
            return float(entry["delay_ms"])
        except (TypeError, ValueError):
            pass
    try:
        return float(os.environ.get("FLASHVEP_INJECT_DELAY_MS", "0"))
    except ValueError:
        return 0.0


def _inject_rank() -> int:
    try:
        return int(os.environ.get("FLASHVEP_INJECT_DP_RANK", "-1"))
    except ValueError:
        return -1


def _inject_layer() -> int:
    try:
        return int(os.environ.get("FLASHVEP_INJECT_LAYER", "24"))
    except ValueError:
        return 24


def _flush_aux() -> None:
    if not _AUX:
        return
    torch.cuda.synchronize()
    out = Path(os.environ["FLASHVEP_ASAP_RAW_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in _AUX:
        rec = item.get("record")
        if not isinstance(rec, dict):
            continue
        if item.get("layer_start") is None or item.get("moe_entry") is None or item.get("moe_done") is None:
            continue
        # The validated base hook retains the same record in its pending
        # list and serializes it at the next flush boundary.  Keep every
        # CUDA Event in this experiment-local side table only; putting an
        # Event into the shared base record would make the base JSON writer
        # fail before our resolver gets a chance to consume it.
        row = {k: v for k, v in rec.items()
               if k not in {"dispatch", "expert", "combine"}}
        row.update({k: v for k, v in item.items()
                    if k not in {"record", "layer_start", "moe_entry", "moe_done",
                                 "delay_start", "delay_end", "waits",
                                 "dispatch_events", "expert_events", "combine_events"}})
        row["pre_moe_cuda_ms"] = float(item["layer_start"].elapsed_time(item["moe_entry"]))
        row["ep_entry_to_done_ms"] = float(item["moe_entry"].elapsed_time(item["moe_done"]))
        row["layer_entry_to_ep_done_ms"] = float(item["layer_start"].elapsed_time(item["moe_done"]))
        if item.get("delay_start") is not None:
            row["injected_delay_cuda_ms"] = float(item["delay_start"].elapsed_time(item["delay_end"]))
        wait_values = []
        for start, end, kind in item.get("waits", []):
            value = float(start.elapsed_time(end))
            wait_values.append({"kind": kind, "cuda_ms": value})
        row["event_waits"] = wait_values
        row["event_wait_cuda_ms"] = float(sum(x["cuda_ms"] for x in wait_values))
        for stage in ("dispatch", "expert", "combine"):
            pair = item.get(f"{stage}_events")
            if isinstance(pair, dict) and pair.get("start") is not None:
                row[f"{stage}_cuda_ms"] = float(pair["start"].elapsed_time(pair["end"]))
        rows.append(row)
    # VLLM's DP worker processes do not always export torch.distributed RANK
    # to this hook.  Use the explicit DP rank first so concurrent workers do
    # not interleave their JSONL writes into one ``rankna`` file.
    rank = os.environ.get("VLLM_DP_RANK", os.environ.get("RANK", os.environ.get("LOCAL_RANK", "na")))
    path = out / f"asap_rank{rank}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=True) + "\n")
    (out / f"asap_rank{rank}.proof.json").write_text(
        json.dumps({"events": len(rows),
                    "wait_metric": "CUDA event recorded before/after EventOverlap.current_stream_wait",
                    "forced_sync_wait": os.environ.get("FLASHVEP_FORCE_SYNC_WAIT", "0") == "1",
                    "host_prepare_span": True,
                    "cross_gpu_absolute_subtraction": False,
                    "injected_delay_gpu_sleep": True}, indent=2) + "\n", encoding="utf-8")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    base.install()
    from deep_ep import EventOverlap
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_forward = Qwen3MoeDecoderLayer.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize
    original_wait = EventOverlap.current_stream_wait

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        layer = _layer(self)
        if layer == 0:
            control = base._control()
            if control.get("flush"):
                _flush_aux()
        start = _event(); start.record(torch.cuda.current_stream())
        base._CONTEXT.asap_layer_start = start
        delay_start = delay_end = None
        dp = int(os.environ.get("VLLM_DP_RANK", "-1"))
        delay = _delay_ms()
        if delay > 0 and dp == _inject_rank() and layer == _inject_layer():
            delay_start = _event(); delay_start.record(torch.cuda.current_stream())
            # torch.cuda._sleep executes on the current GPU stream.  The
            # cycle scale is calibrated by the paired CUDA events below.
            torch.cuda._sleep(max(1, int(delay * 1_000_000)))
            delay_end = _event(); delay_end.record(torch.cuda.current_stream())
        base._CONTEXT.asap_delay_start = delay_start
        base._CONTEXT.asap_delay_end = delay_end
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            base._CONTEXT.asap_layer_start = None
            base._CONTEXT.asap_delay_start = None
            base._CONTEXT.asap_delay_end = None

    def patched_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = getattr(base._CONTEXT, "entry", {})
        layer = int(getattr(base._CONTEXT, "layer", -1))
        if entry.get("instrument") and layer >= 0:
            from poc_flashvep.dp_ep_arrival_skew_two_topologies.topology_probe import write_once
            write_once()
        host_prepare_start = time.monotonic_ns()
        value = original_prepare(self, *args, **kwargs)
        host_prepare_end = time.monotonic_ns()
        if entry.get("instrument") and layer >= 0:
            rec = getattr(base._CONTEXT, "record", None)
            if rec is not None:
                moe_entry = _event()
                moe_entry.record(torch.cuda.current_stream())
                aux = {
                    "record": rec,
                    "pre_moe_layer": layer,
                    "layer_start": getattr(base._CONTEXT, "asap_layer_start", None),
                    "moe_entry": moe_entry,
                    "waits": [],
                    "host_moe_entry_ns": time.monotonic_ns(),
                    "host_prepare_start_ns": host_prepare_start,
                    "host_prepare_end_ns": host_prepare_end,
                    "delay_start": getattr(base._CONTEXT, "asap_delay_start", None),
                    "delay_end": getattr(base._CONTEXT, "asap_delay_end", None),
                    "delay_ms": _delay_ms(),
                    "dispatch_events": rec.get("dispatch"),
                }
                # Only JSON-safe scalars are added to the shared validated
                # record.  CUDA events stay in `aux`, never in `rec`.
                rec.update({"pre_moe_layer": layer,
                            "host_moe_entry_ns": aux["host_moe_entry_ns"]})
                base._CONTEXT.asap_aux = aux
        return value

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> Any:
        rec = getattr(base._CONTEXT, "record", None)
        aux = getattr(base._CONTEXT, "asap_aux", None)
        value = original_experts(self, *args, **kwargs)
        if rec is not None and isinstance(aux, dict):
            # The base hook has already recorded the expert span in rec.  Keep
            # the Event-bearing pair in the side table only for our resolver;
            # the shared JSON writer excludes this key by design.
            aux["expert_events"] = rec.get("expert")
        return value

    def patched_finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        rec = getattr(base._CONTEXT, "record", None)
        aux = getattr(base._CONTEXT, "asap_aux", None)
        value = original_finalize(self, *args, **kwargs)
        if rec is not None:
            moe_done = _event(); moe_done.record(torch.cuda.current_stream())
            if isinstance(aux, dict):
                aux["moe_done"] = moe_done
                aux["host_moe_done_ns"] = time.monotonic_ns()
                aux["prepare_host_ms"] = (aux["host_prepare_end_ns"] - aux["host_prepare_start_ns"]) / 1e6
                aux["combine_events"] = rec.get("combine")
                with _LOCK:
                    _AUX.append(aux)
            # The base finalizer clears `record`; clear our side reference as
            # well so a later wait cannot attach to a completed invocation.
            base._CONTEXT.asap_aux = None
        return value

    def patched_wait(self: Any) -> None:
        rec = getattr(base._CONTEXT, "record", None)
        aux = getattr(base._CONTEXT, "asap_aux", None)
        if rec is None:
            return original_wait(self)
        start = _event(); start.record(torch.cuda.current_stream())
        value = original_wait(self)
        # Positive-control mode makes the normally asynchronous stream wait
        # observable as an elapsed GPU interval.  It is never enabled for
        # natural serving measurements, because synchronizing here would
        # intentionally perturb overlap semantics.
        if os.environ.get("FLASHVEP_FORCE_SYNC_WAIT", "0") == "1":
            torch.cuda.current_stream().synchronize()
        end = _event(); end.record(torch.cuda.current_stream())
        if isinstance(aux, dict):
            aux.setdefault("waits", []).append((start, end, "deepep_event_overlap"))
        return value

    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._prepare = patched_prepare
    FusedMoEKernelModularImpl._fused_experts = patched_experts
    FusedMoEKernelModularImpl._finalize = patched_finalize
    # EventOverlap is intentionally left unmodified in the baseline run.  A
    # separate positive-control mode enables the wrapper below; keeping the
    # stock method on production runs avoids perturbing DeepEP's Python/CUDA
    # callback lifetime.
    if os.environ.get("FLASHVEP_CAPTURE_EVENT_WAITS", "0") == "1":
        EventOverlap.current_stream_wait = patched_wait
