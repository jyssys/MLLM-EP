"""Low-overhead live DeepEP timing and local routing histogram capture.

This is an experiment-local copy of the validated read-only hook.  It keeps a
CUDA-event span around the existing dispatch, expert, and combine calls and
resolves all events only at the explicit flush boundary.  A model forward is
one scheduler iteration, so repeated layer-0 entries with the same batch are
intentionally retained rather than collapsed.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch


_INSTALLED = False
_CONTEXT = threading.local()
_PENDING: list[dict[str, Any]] = []
_FLUSHED = False


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_MATRIX_CONTROL"])
    if not path.exists():
        return {}
    # The host replaces control.json atomically at request boundaries.  A
    # worker can still observe the old inode while it is being replaced; this
    # must be treated as an empty control record, never as a model/runtime
    # error.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _runtime_config(kernel: Any, values: dict[str, Any]) -> dict[str, int]:
    from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config

    experts = kernel.fused_experts
    config = try_get_optimal_moe_config(
        values["w1"].size(), values["w2"].size(), 8,
        experts.quant_config.config_name(values["a1q"].dtype),
        int(values["a1q"].shape[0]), block_shape=experts.block_shape,
    )
    return {key: int(value) for key, value in config.items()
            if isinstance(value, (int, bool))}


def _span(stage: dict[str, torch.cuda.Event]) -> dict[str, float]:
    start, end = stage["start"], stage["end"]
    return {"ms": float(start.elapsed_time(end))}


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    if not _PENDING:
        return
    torch.cuda.synchronize()
    from vllm.distributed import get_ep_group

    ep_rank = int(get_ep_group().rank_in_group)
    output = Path(os.environ["FLASHVEP_MATRIX_RAW_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank{ep_rank}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in _PENDING:
            row = {
                key: value for key, value in item.items()
                if key not in ("dispatch", "expert", "combine")
            }
            row["dispatch"] = _span(item["dispatch"])
            row["expert"] = _span(item["expert"])
            row["combine"] = _span(item["combine"])
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    proof = {
        "status": "ok",
        "ep_rank": ep_rank,
        "events": len(_PENDING),
        "timing_scope": {
            "dispatch": "FusedMoEKernelModularImpl._prepare",
            "expert": "FusedMoEKernelModularImpl._fused_experts",
            "combine": "FusedMoEKernelModularImpl._finalize",
        },
        "event_resolution": "one bounded torch.cuda.synchronize at flush",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deep_ep_collective_overlap": 0,
        "model_forwards_retained": True,
    }
    (output / f"rank{ep_rank}.proof.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf-8"
    )


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
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_serving_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        prior_layer = getattr(_CONTEXT, "layer", -1)
        prior_entry = getattr(_CONTEXT, "entry", {})
        layer = int(getattr(self, "_flashvep_serving_layer", -1))
        if layer == 0:
            entry = _control()
            if entry.get("flush"):
                _flush()
            _CONTEXT.entry = entry
            _CONTEXT.forward_index = int(getattr(_CONTEXT, "forward_index", 0)) + 1
        _CONTEXT.layer = layer
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = prior_layer
            if layer == 47:
                _CONTEXT.entry = prior_entry

    def patched_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry = dict(getattr(_CONTEXT, "entry", {}))
        layer = int(getattr(_CONTEXT, "layer", -1))
        if not entry.get("instrument") or layer < 0:
            return original_prepare(self, *args, **kwargs)
        compute = torch.cuda.current_stream()
        comm = self.prepare_finalize.buffer.get_comm_stream()
        cs, ce = _event(), _event()
        cs.record(compute)
        value = original_prepare(self, *args, **kwargs)
        ce.record(compute)
        _CONTEXT.record = {
            "wave": int(entry.get("wave", -1)),
            "batch_id": entry.get("batch_id"),
            "request_id": entry.get("batch_id", entry.get("request_id")),
            "condition": entry.get("condition"),
            "modality": entry.get("modality"),
            "concurrency": int(entry.get("concurrency", 1)),
            "phase": entry.get("phase"),
            "iteration": int(entry.get("iteration", -1)),
            "measured": bool(entry.get("measured", False)),
            "scheduler_iteration": int(getattr(_CONTEXT, "forward_index", 0)),
            "worker_dp_rank": int(os.environ.get("VLLM_DP_RANK", -1)),
            "ep_rank": int(get_ep_group().rank_in_group),
            "layer": layer,
            "dispatch": {"start": cs, "end": ce},
        }
        return value

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        record = getattr(_CONTEXT, "record", None)
        if record is None:
            return original_experts(self, *args, **kwargs)
        names = (
            "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
            "topk_ids", "activation", "global_num_experts", "local_num_experts",
            "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
        )
        values = dict(zip(names, args, strict=False))
        values.update(kwargs)
        metadata = values.get("expert_tokens_meta")
        if metadata is None or metadata.expert_num_tokens_cpu is None:
            raise RuntimeError("DeepEP expert CPU histogram unavailable")
        histogram = [int(x) for x in metadata.expert_num_tokens_cpu.tolist()]
        start, end = _event(), _event()
        stream = torch.cuda.current_stream()
        start.record(stream)
        output = original_experts(self, *args, **kwargs)
        end.record(stream)
        record.update({
            "expert_histogram": histogram,
            "total_assignments": int(sum(histogram)),
            "dispatched_rows": int(values["a1q"].shape[0]),
            "runtime_m": int(values["a1q"].shape[0]),
            "runtime_config": _runtime_config(self, values),
            "expert_backend": type(self.fused_experts).__name__,
            "prepare_finalize_backend": type(self.prepare_finalize).__name__,
            "expert": {"start": start, "end": end},
        })
        return output

    def patched_finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        record = getattr(_CONTEXT, "record", None)
        if record is None:
            return original_finalize(self, *args, **kwargs)
        compute = torch.cuda.current_stream()
        cs, ce = _event(), _event()
        cs.record(compute)
        value = original_finalize(self, *args, **kwargs)
        ce.record(compute)
        record["combine"] = {"start": cs, "end": ce}
        _PENDING.append(record)
        _CONTEXT.record = None
        return value

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._prepare = patched_prepare
    FusedMoEKernelModularImpl._fused_experts = patched_experts
    FusedMoEKernelModularImpl._finalize = patched_finalize
