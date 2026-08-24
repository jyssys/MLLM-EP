"""Targeted live marker and exact-input isolated TritonExperts replay."""

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
_LAST_WAVE = -1


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_FORENSIC_CONTROL"])
    return json.loads(path.read_text()) if path.exists() else {}


def _write(name: str, payload: dict[str, Any]) -> None:
    output = Path(os.environ["FLASHVEP_FORENSIC_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


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
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_forensic_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _LAST_WAVE
        previous_layer = getattr(_CONTEXT, "layer", -1)
        previous_entry = getattr(_CONTEXT, "entry", {})
        layer = int(getattr(self, "_flashvep_forensic_layer", -1))
        if layer == 0:
            entry = _control()
            wave = int(entry.get("wave", -1))
            _CONTEXT.entry = entry if wave != _LAST_WAVE else {**entry, "target": False}
            _LAST_WAVE = wave
        _CONTEXT.layer = layer
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous_layer
            if layer == 47:
                _CONTEXT.entry = previous_entry

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        entry = dict(getattr(_CONTEXT, "entry", {}))
        layer = int(getattr(_CONTEXT, "layer", -1))
        rank = int(get_ep_group().rank_in_group)
        targeted = bool(entry.get("target")) and layer == int(entry["layer"]) and rank == int(entry["rank"])
        if not targeted:
            return original_experts(self, *args, **kwargs)

        label = f"MLLM_FORENSIC_{entry['modality'].upper()}_L{layer}_R{rank}"
        # The common outer range lets Nsight defer CUDA collection until this
        # one bounded expert call; the inner label retains modality identity.
        torch.cuda.nvtx.range_push("MLLM_FORENSIC_TARGET")
        torch.cuda.nvtx.range_push(label)
        result = original_experts(self, *args, **kwargs)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        if os.environ.get("FLASHVEP_FORENSIC_MODE") != "replay":
            return result

        names = (
            "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
            "topk_ids", "activation", "global_num_experts", "local_num_experts",
            "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
        )
        values = dict(zip(names, args, strict=False)); values.update(kwargs)
        metadata = values["expert_tokens_meta"]
        histogram = [int(value) for value in metadata.expert_num_tokens_cpu.tolist()]
        warmups = int(os.environ.get("FLASHVEP_FORENSIC_WARMUPS", "20"))
        iterations = int(os.environ.get("FLASHVEP_FORENSIC_ITERATIONS", "100"))
        for _ in range(warmups):
            original_experts(self, *args, **kwargs)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record(torch.cuda.current_stream())
            original_experts(self, *args, **kwargs)
            end.record(torch.cuda.current_stream())
        ends[-1].synchronize()
        samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
        _write(entry["modality"], {
            "request_id": entry["request_id"], "modality": entry["modality"],
            "layer": layer, "rank": rank, "physical_gpu": [4, 5, 6, 7][rank],
            "total_assignments": sum(histogram), "active_experts": sum(value > 0 for value in histogram),
            "expert_histogram": histogram, "dispatched_rows": int(values["a1q"].shape[0]),
            "warmups": warmups, "iterations": iterations, "samples_ms": samples,
            "routing_changed": False, "input_provenance": "exact live-prefill post-DeepEP expert input",
        })
        return result

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
