"""Low-overhead live DeepEP dispatch/combine and expert trace capture.

The wrapper deliberately times existing ``_prepare``/``_finalize`` calls.  In
vLLM 0.20 those methods call DeepEP ``prepare_async``/``finalize_async`` and
their receivers, so no second collective or routing change is introduced.
CUDA events are resolved once at the final flush rather than synchronized per
layer.
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
_LAST_WAVE = -1
_FLUSHED = False


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_MATRIX_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _runtime_config(kernel: Any, values: dict[str, Any]) -> dict[str, int]:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        try_get_optimal_moe_config,
    )

    experts = kernel.fused_experts
    config = try_get_optimal_moe_config(
        values["w1"].size(), values["w2"].size(), 8,
        experts.quant_config.config_name(values["a1q"].dtype),
        int(values["a1q"].shape[0]), block_shape=experts.block_shape,
    )
    return {key: int(value) for key, value in config.items()
            if isinstance(value, (int, bool))}


def _span(stage: dict[str, torch.cuda.Event], origin: torch.cuda.Event) -> dict[str, float]:
    start, end = stage["start"], stage["end"]
    return {
        "ms": float(start.elapsed_time(end)),
        "start_ms": float(origin.elapsed_time(start)),
        "end_ms": float(origin.elapsed_time(end)),
    }


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    if not _PENDING:
        return
    # One bounded synchronization after all measured forwards.  This does not
    # serialize individual layers during the measured run.
    torch.cuda.synchronize()
    origin = torch.cuda.Event(enable_timing=True)
    origin.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    from vllm.distributed import get_ep_group

    ep_rank = int(get_ep_group().rank_in_group)
    output = Path(os.environ["FLASHVEP_MATRIX_RAW_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank{ep_rank}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in _PENDING:
            row = {key: value for key, value in item.items()
                   if key not in ("dispatch", "expert", "combine")}
            row["dispatch"] = _span(item["dispatch"], origin)
            row["expert"] = _span(item["expert"], origin)
            row["combine"] = _span(item["combine"], origin)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / f"rank{ep_rank}.proof.json").write_text(
        json.dumps({
            "status": "ok", "ep_rank": ep_rank, "events": len(_PENDING),
            "timing_scope": {
                "dispatch": "FusedMoEKernelModularImpl._prepare (DeepEP prepare/receiver)",
                "expert": "FusedMoEKernelModularImpl._fused_experts",
                "combine": "FusedMoEKernelModularImpl._finalize (DeepEP finalize/receiver)",
            },
            "event_resolution": "one final bounded torch.cuda.synchronize",
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deep_ep_collective_overlap": 0,
        }, indent=2) + "\n", encoding="utf-8")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_matrix_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _LAST_WAVE
        prior_layer = getattr(_CONTEXT, "layer", -1)
        prior_entry = getattr(_CONTEXT, "entry", {})
        layer = int(getattr(self, "_flashvep_matrix_layer", -1))
        if layer == 0:
            entry = _control()
            wave = int(entry.get("wave", -1))
            if entry.get("flush"):
                _flush()
            _CONTEXT.entry = entry if wave != _LAST_WAVE else {**entry, "instrument": False}
            _LAST_WAVE = wave
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
        cs, ce, ms, me = _event(), _event(), _event(), _event()
        cs.record(compute); ms.record(comm)
        value = original_prepare(self, *args, **kwargs)
        ce.record(compute); me.record(comm)
        _CONTEXT.record = {
            "wave": int(entry["wave"]), "request_id": entry["request_id"],
            "modality": entry["modality"], "pair_id": int(entry["pair_id"]),
            "token_bucket": entry["token_bucket"], "phase": entry["phase"],
            "iteration": int(entry["iteration"]), "measured": bool(entry["measured"]),
            "source_dp_rank": int(entry["source_dp_rank"]),
            "worker_dp_rank": int(os.environ["VLLM_DP_RANK"]),
            "ep_rank": int(get_ep_group().rank_in_group), "layer": layer,
            "dispatch": {"start": cs, "end": ce, "comm_start": ms, "comm_end": me},
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
        values = dict(zip(names, args, strict=False)); values.update(kwargs)
        metadata = values.get("expert_tokens_meta")
        if metadata is None or metadata.expert_num_tokens_cpu is None:
            raise RuntimeError("DeepEP expert CPU histogram unavailable")
        histogram = [int(x) for x in metadata.expert_num_tokens_cpu.tolist()]
        start, end = _event(), _event()
        stream = torch.cuda.current_stream(); start.record(stream)
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
        })
        record["expert"] = {"start": start, "end": end}
        return output

    def patched_finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        record = getattr(_CONTEXT, "record", None)
        if record is None:
            return original_finalize(self, *args, **kwargs)
        compute = torch.cuda.current_stream()
        comm = self.prepare_finalize.buffer.get_comm_stream()
        cs, ce, ms, me = _event(), _event(), _event(), _event()
        cs.record(compute); ms.record(comm)
        value = original_finalize(self, *args, **kwargs)
        ce.record(compute); me.record(comm)
        record["combine"] = {"start": cs, "end": ce,
                             "comm_start": ms, "comm_end": me}
        _PENDING.append(record)
        _CONTEXT.record = None
        return value

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._prepare = patched_prepare
    FusedMoEKernelModularImpl._fused_experts = patched_experts
    FusedMoEKernelModularImpl._finalize = patched_finalize
