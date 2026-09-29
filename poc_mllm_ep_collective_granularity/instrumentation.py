"""Small extensions to the validated communication instrumentation.

The earlier TRUE-EP4 hooks remain the timing implementation.  This module only
adds scheduler-phase metadata, resident-route identity, and correct outer
iteration labels for external-DP dummy participants.  It does not synchronize
the clean serving hot path.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from poc_mllm_ep_communication_motivation import instrumentation as base

_INSTALLED = False
_EVENTS: list[dict[str, Any]] = []
_LOCK = threading.Lock()


def _route_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    selected = ids.detach()[mask].to("cpu", non_blocking=False).contiguous()
    return hashlib.sha256(selected.numpy().tobytes()).hexdigest()


def _drain_events() -> None:
    if not _EVENTS:
        return
    root = Path(os.environ["MLLM_COMM_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"scheduler.pid{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for event in _EVENTS:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    _EVENTS.clear()


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    original_route_stats = base._route_stats
    original_drain = base._drain

    def drain() -> None:
        # Preserve the caller iteration alongside the worker's internal model
        # execution index.  The validated base hook drains once per 48-layer
        # pass, so all pending rows belong to the current control epoch.
        control = base._control()
        for record, _ in base._PENDING:
            record.setdefault("outer_iteration", control.get("outer_iteration"))
        original_drain()

    def route_stats(
        ids: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
        injected: torch.Tensor | None,
    ) -> dict[str, Any]:
        result = original_route_stats(ids, visual, text, injected)
        if injected is not None:
            resident = ~injected.bool()
            zeros = torch.zeros_like(resident)
            resident_result = original_route_stats(ids, resident, zeros, None)["visual"]
            resident_result["route_id_sha256"] = _route_hash(ids, resident)
            result["resident"] = resident_result
            result["injected"]["route_id_sha256"] = _route_hash(ids, injected.bool())
        return result

    base._route_stats = route_stats
    base._drain = drain
    base.install()
    atexit.register(_drain_events)

    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    timed_forward = GPUModelRunner._model_forward
    timed_apply = FusedMoEKernelModularImpl.apply

    @functools.wraps(timed_forward)
    def forward(self: Any, *args: Any, **kwargs: Any):
        begin_ns = time.perf_counter_ns()
        result = timed_forward(self, *args, **kwargs)
        context = getattr(base._TLS, "context", None)
        if context is not None and not getattr(base._TLS, "bypass", False):
            control = context.get("control", {})
            event = {
                "wave": control.get("wave"),
                "condition": control.get("condition"),
                "replicate": control.get("replicate"),
                "outer_iteration": control.get("outer_iteration"),
                "internal_iteration": context.get("iteration"),
                "phase": context.get("phase"),
                "decode_step": context.get("decode_step"),
                "states": context.get("states", []),
                "host_forward_begin_ns": begin_ns,
                "host_forward_end_ns": time.perf_counter_ns(),
            }
            with _LOCK:
                _EVENTS.append(event)
        return result

    @functools.wraps(timed_apply)
    def apply(self: Any, *args: Any, **kwargs: Any):
        context = getattr(base._TLS, "context", None)
        owns_context = context is None
        if owns_context:
            control = base._control()
            if not bool(control.get("capture_stage", False)):
                return timed_apply(self, *args, **kwargs)
            context = {
                "control": control,
                "iteration": int(control.get("outer_iteration", -1)),
                "phase": "external_dp_dummy",
                "states": [],
                "decode_step": int(control.get("decode_progress", -1)),
            }
            base._TLS.context = context
        assert context is not None
        control = context.get("control", {})
        replay = bool(control.get("moe_only_replay", False)) and (
            context.get("phase") == "prefill"
            or control.get("expected_phase") == "prefill"
        )
        replay_args = None
        replay_kwargs = None
        saved_route = getattr(base._TLS, "route_record", None)
        if replay:
            replay_args = list(args)
            replay_kwargs = dict(kwargs)
            if replay_args and isinstance(replay_args[0], torch.Tensor):
                replay_args[0] = replay_args[0].detach().clone()
            elif isinstance(replay_kwargs.get("hidden_states"), torch.Tensor):
                replay_kwargs["hidden_states"] = (
                    replay_kwargs["hidden_states"].detach().clone()
                )
            for index in (3, 4):
                if len(replay_args) > index and isinstance(
                    replay_args[index], torch.Tensor
                ):
                    replay_args[index] = replay_args[index].detach().clone()
        try:
            result = timed_apply(self, *args, **kwargs)
            if replay and replay_args is not None and replay_kwargs is not None:
                original_control = context["control"]
                context["control"] = {
                    **original_control,
                    "condition": f"{original_control['condition']}_MOE_REPLAY",
                    "moe_only_replay": False,
                }
                base._TLS.route_record = saved_route
                try:
                    timed_apply(self, *replay_args, **replay_kwargs)
                finally:
                    context["control"] = original_control
            return result
        finally:
            if owns_context:
                base._TLS.context = None

    GPUModelRunner._model_forward = forward
    FusedMoEKernelModularImpl.apply = apply


__all__ = ["install"]
