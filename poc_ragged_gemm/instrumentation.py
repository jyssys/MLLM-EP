"""Opt-in live Qwen3 expert timing and exact local histogram capture."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

_CONTEXT = threading.local()
_PENDING: list[dict[str, Any]] = []
_LAST_WAVE = -1
_FLUSHED = False


def _control() -> dict[str, Any]:
    path = Path(os.environ["RAGGED_GEMM_CONTROL"])
    return json.loads(path.read_text()) if path.exists() else {}


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _runtime_config(kernel: Any, values: dict[str, Any]) -> dict[str, int]:
    experts = kernel.fused_experts
    if hasattr(experts, "quant_config"):
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
        config = try_get_optimal_moe_config(
            values["w1"].size(), values["w2"].size(), 8,
            experts.quant_config.config_name(values["a1q"].dtype),
            int(values["a1q"].shape[0]), block_shape=experts.block_shape,
        )
        return {k: int(v) for k, v in config.items() if isinstance(v, (int, bool))}
    # The auto-selected FlashInfer CUTLASS H100 build encodes M128 in its
    # generated kernel family (fused_moe_90/gemm_grouped/*_M128_*).
    return {"BLOCK_SIZE_M": 128}


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    if _PENDING:
        _PENDING[-1]["end"].synchronize()
    from vllm.distributed import get_ep_group
    rank = int(get_ep_group().rank_in_group)
    output = Path(os.environ["RAGGED_GEMM_RAW_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"rank{rank}.jsonl").open("w") as handle:
        for item in _PENDING:
            row = {k: v for k, v in item.items() if k not in {"start", "end"}}
            row["expert_ms"] = float(item["start"].elapsed_time(item["end"]))
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / f"rank{rank}.proof.json").write_text(json.dumps({
        "status": "ok", "ep_rank": rank, "events": len(_PENDING),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timed_scope": (
            "live selected FusedMoEExperts backend only, "
            "post-dispatch/pre-combine"
        ),
        "event_resolution": "one bounded synchronization at flush",
    }, indent=2) + "\n")


def install() -> None:
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._ragged_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _LAST_WAVE
        prev_layer = getattr(_CONTEXT, "layer", -1)
        prev_entry = getattr(_CONTEXT, "entry", {})
        layer = int(getattr(self, "_ragged_layer", -1))
        if layer == 0:
            entry = _control(); wave = int(entry.get("wave", -1))
            first = wave != _LAST_WAVE; _LAST_WAVE = wave
            if entry.get("flush"):
                _flush()
            _CONTEXT.entry = entry if first else {**entry, "instrument": False}
        _CONTEXT.layer = layer
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = prev_layer
            if layer == 47:
                _CONTEXT.entry = prev_entry

    def patched_experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        entry = dict(getattr(_CONTEXT, "entry", {})); layer = int(getattr(_CONTEXT, "layer", -1))
        if not entry.get("instrument") or layer < 0:
            return original_experts(self, *args, **kwargs)
        names = (
            "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
            "topk_ids", "activation", "global_num_experts", "local_num_experts",
            "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
        )
        values = dict(zip(names, args, strict=False)); values.update(kwargs)
        metadata = values.get("expert_tokens_meta")
        if metadata is not None and metadata.expert_num_tokens_cpu is not None:
            histogram = [int(x) for x in metadata.expert_num_tokens_cpu.tolist()]
            histogram_source = "expert_num_tokens_cpu"
        else:
            topk_ids = values["topk_ids"].to(torch.long)
            expert_map = values.get("expert_map")
            local_ids = topk_ids if expert_map is None else expert_map.to(torch.long)[topk_ids]
            local_ids = local_ids[local_ids >= 0]
            histogram = [int(x) for x in torch.bincount(
                local_ids.flatten(), minlength=int(values["local_num_experts"])
            ).cpu().tolist()]
            histogram_source = "topk_ids_mapped_by_expert_map"
        from vllm.distributed import get_ep_group
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        result = original_experts(self, *args, **kwargs)
        end.record(torch.cuda.current_stream())
        _PENDING.append({
            "start": start, "end": end, "wave": int(entry["wave"]),
            "workload": entry["workload"], "prefill_tokens": int(entry["prefill_tokens"]),
            "batch_size": int(entry["batch_size"]), "repeat": int(entry["repeat"]),
            "layer": layer, "ep_rank": int(get_ep_group().rank_in_group),
            "histogram": histogram, "N": int(sum(histogram)),
            "histogram_source": histogram_source,
            "runtime_m": int(values["a1q"].shape[0]), "runtime_config": _runtime_config(self, values),
            "global_num_experts": int(values["global_num_experts"]),
            "local_num_experts": int(values["local_num_experts"]),
            "expert_backend": type(self.fused_experts).__name__,
        })
        return result

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
