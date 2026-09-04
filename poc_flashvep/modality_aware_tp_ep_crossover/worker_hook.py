"""Low-overhead per-worker MoE timing for the crossover experiment.

The hook only wraps existing vLLM calls.  CUDA events are resolved once at
process exit, so no per-layer synchronization or routing change is introduced.
It works for both the TP-only and EP4 FusedMoE implementations.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

_INSTALLED = False
_CTX = threading.local()
_PENDING: list[dict[str, Any]] = []
_LAST_WAVE = None


def _layer(name: str) -> int:
    m = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", str(name))
    return int(m.group(1)) if m else -1


def _control() -> dict[str, Any]:
    path = os.environ.get("FLASHVEP_CROSSOVER_CONTROL")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _evt() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _span(pair: tuple[torch.cuda.Event, torch.cuda.Event] | None) -> float | None:
    if pair is None:
        return None
    return float(pair[0].elapsed_time(pair[1]))


def _rank() -> int:
    # vLLM's worker environment does not consistently expose LOCAL_RANK.
    # Once NCCL is initialized, the process-group rank is the authoritative
    # physical-local worker identifier for this TP4 run.
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    for key in ("LOCAL_RANK", "VLLM_TP_RANK", "VLLM_RANK", "RANK"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def _flush() -> None:
    if not _PENDING:
        return
    try:
        torch.cuda.synchronize()
        raw = Path(os.environ["FLASHVEP_CROSSOVER_RAW_DIR"])
        raw.mkdir(parents=True, exist_ok=True)
        # At interpreter shutdown torch.distributed may already be torn down;
        # prefer the rank captured at call time so file names remain unique.
        observed_ranks = {int(r.get("local_rank", -1)) for r in _PENDING
                          if r.get("local_rank") is not None}
        local_rank = max(observed_ranks) if observed_ranks else _rank()
        out = raw / f"rank{local_rank}_pid{os.getpid()}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for rec in _PENDING:
                row = dict(rec)
                for phase in ("full_moe", "dispatch", "expert", "combine"):
                    pair = row.pop(f"_{phase}_events", None)
                    row[f"{phase}_ms"] = _span(pair)
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        proof = {
            "status": "ok",
            "pid": os.getpid(),
            "local_rank": local_rank,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "timing": "CUDA events, one final synchronize at worker exit",
            "records": len(_PENDING),
            "routing_placement": "logical expert_id // 32 for EP4 diagnostics",
        }
        (raw / f"rank{local_rank}.proof.json").write_text(
            json.dumps(proof, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover
        path = os.environ.get("FLASHVEP_CROSSOVER_RAW_DIR")
        if path:
            Path(path, f"flush_error_{os.getpid()}.txt").write_text(repr(exc) + "\n")


def _write_resolved_now(rec: dict[str, Any]) -> None:
    """Persist one completed span for workers that are terminated by V1.

    V1's EngineCore commonly terminates GPU workers rather than running their
    Python ``atexit`` handlers.  With the opt-in capture flag, synchronize
    only the diagnostic worker after each MoE call, resolve CUDA events, and
    append the record immediately.  This is intentionally disabled by
    default; the volume experiment enables it only for the short capture run.
    """
    raw = Path(os.environ["FLASHVEP_CROSSOVER_RAW_DIR"])
    raw.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    row = dict(rec)
    for phase in ("full_moe", "dispatch", "expert", "combine"):
        pair = row.pop(f"_{phase}_events", None)
        row[f"{phase}_ms"] = _span(pair)
    local_rank = int(row.get("local_rank", _rank()))
    out = raw / f"rank{local_rank}_pid{os.getpid()}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    # Leave a lightweight proof for spawned vLLM workers.  This is useful
    # because DP/engine workers can start after Python sitecustomize runs.
    raw_dir = os.environ.get("FLASHVEP_CROSSOVER_RAW_DIR")
    if raw_dir:
        try:
            Path(raw_dir).mkdir(parents=True, exist_ok=True)
            Path(raw_dir, f"hook_install_{os.getpid()}.txt").write_text(
                "installed\n", encoding="utf-8"
            )
        except Exception:
            pass
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl

    original_forward = FusedMoE.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    def moe_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _LAST_WAVE
        control = _control()
        wave = control.get("wave")
        # A control file is updated by the synchronous driver before each
        # request.  All 48 layers in that request share this metadata.
        if wave != _LAST_WAVE:
            _LAST_WAVE = wave
            _CTX.meta = control
        meta = dict(getattr(_CTX, "meta", control))
        layer = _layer(getattr(self, "layer_name", ""))
        start, end = _evt(), _evt()
        start.record(torch.cuda.current_stream())
        prior = getattr(_CTX, "record", None)
        _CTX.record = {
            "wave": meta.get("wave"),
            "workload": meta.get("workload"),
            "iteration": meta.get("iteration"),
            "measured": bool(meta.get("measured", False)),
            "modality": meta.get("modality"),
            "layer": layer,
            "worker_pid": os.getpid(),
            "local_rank": _rank(),
            "dp_rank": int(os.environ.get("VLLM_DP_RANK", "-1")),
            "tp_rank": int(os.environ.get("VLLM_TP_RANK", "-1")),
            "_full_moe_events": (start, end),
        }
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            end.record(torch.cuda.current_stream())
            rec = getattr(_CTX, "record", None)
            if rec is not None:
                # Kernel wrappers populate phase pairs; no route mutation.
                if os.environ.get("FLASHVEP_CROSSOVER_SYNC_RECORD") == "1":
                    try:
                        _write_resolved_now(rec)
                    except Exception as exc:  # pragma: no cover
                        raw = os.environ.get("FLASHVEP_CROSSOVER_RAW_DIR")
                        if raw:
                            Path(raw, f"record_error_{os.getpid()}.txt").write_text(
                                repr(exc) + "\n", encoding="utf-8"
                            )
                else:
                    _PENDING.append(rec)
            _CTX.record = prior

    def timed_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        rec = getattr(_CTX, "record", None)
        if rec is None:
            return original_prepare(self, *args, **kwargs)
        s, e = _evt(), _evt(); s.record(torch.cuda.current_stream())
        result = original_prepare(self, *args, **kwargs)
        e.record(torch.cuda.current_stream())
        rec["_dispatch_events"] = (s, e)
        # The canonical signature has topk_ids at positional index 2.
        try:
            ids = args[2] if len(args) > 2 else kwargs.get("topk_ids")
            if ids is not None:
                flat = ids.detach().to("cpu").reshape(-1).tolist()
                hist = {}
                for value in flat:
                    hist[str(int(value))] = hist.get(str(int(value)), 0) + 1
                rec["expert_histogram"] = hist
                rec["total_assignments"] = len(flat)
                rec["active_experts"] = len(hist)
                rec["rank_histogram_ep4"] = {
                    str(rank): sum(v for key, v in hist.items() if int(key) // 32 == rank)
                    for rank in range(4)
                }
        except Exception:
            rec["route_capture"] = "unavailable"
        return result

    def timed_experts(self: Any, *args: Any, **kwargs: Any) -> Any:
        rec = getattr(_CTX, "record", None)
        if rec is None:
            return original_experts(self, *args, **kwargs)
        s, e = _evt(), _evt(); s.record(torch.cuda.current_stream())
        result = original_experts(self, *args, **kwargs)
        e.record(torch.cuda.current_stream())
        rec["_expert_events"] = (s, e)
        rec["expert_backend"] = type(self.fused_experts).__name__
        rec["prepare_finalize_backend"] = type(self.prepare_finalize).__name__
        return result

    def timed_finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        rec = getattr(_CTX, "record", None)
        if rec is None:
            return original_finalize(self, *args, **kwargs)
        s, e = _evt(), _evt(); s.record(torch.cuda.current_stream())
        result = original_finalize(self, *args, **kwargs)
        e.record(torch.cuda.current_stream())
        rec["_combine_events"] = (s, e)
        return result

    FusedMoE.forward = moe_forward
    FusedMoEKernelModularImpl._prepare = timed_prepare
    FusedMoEKernelModularImpl._fused_experts = timed_experts
    FusedMoEKernelModularImpl._finalize = timed_finalize
    atexit.register(_flush)
