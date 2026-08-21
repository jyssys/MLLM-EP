"""Low-perturbation CUDA-event timing of live post-DeepEP TritonExperts calls."""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
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


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_LIVE_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _runtime_config(kernel: Any, values: dict[str, Any]) -> dict[str, int]:
    from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config

    experts = kernel.fused_experts
    config = try_get_optimal_moe_config(
        values["w1"].size(), values["w2"].size(), 8,
        experts.quant_config.config_name(values["a1q"].dtype),
        int(values["a1q"].shape[0]), block_shape=experts.block_shape,
    )
    return {key: int(value) for key, value in config.items() if isinstance(value, (int, bool))}


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    if _PENDING:
        _PENDING[-1]["end"].synchronize()
    from vllm.distributed import get_ep_group

    ep_rank = int(get_ep_group().rank_in_group)
    output = Path(os.environ["FLASHVEP_LIVE_RAW_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank{ep_rank}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in _PENDING:
            row = {key: value for key, value in item.items() if key not in ("start", "end")}
            row["expert_ms"] = float(item["start"].elapsed_time(item["end"]))
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    proof = {
        "status": "ok", "ep_rank": ep_rank, "events": len(_PENDING),
        "expert_backend": "TritonExperts",
        "timed_scope": "FusedMoEKernelModularImpl._fused_experts after DeepEP dispatch and before combine",
        "event_resolution": "single bounded synchronization after all measured prefills",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output / f"rank{ep_rank}.proof.json").write_text(json.dumps(proof, indent=2) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__
    original_forward = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._flashvep_live_layer = _layer(prefix)

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _LAST_WAVE
        previous_layer = getattr(_CONTEXT, "layer", -1)
        previous_entry = getattr(_CONTEXT, "entry", {})
        layer = int(getattr(self, "_flashvep_live_layer", -1))
        if layer == 0:
            entry = _control()
            wave = int(entry.get("wave", -1))
            first_model_call = wave != _LAST_WAVE
            _LAST_WAVE = wave
            if entry.get("flush"):
                _flush()
            # A vLLM DP wave can contain later decode/padding model calls even
            # with max_tokens=1.  The first call is the only real prefill and
            # is the preregistered measurement target.
            _CONTEXT.entry = entry if first_model_call else {**entry, "instrument": False}
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
        if not entry.get("instrument") or layer < 0:
            return original_experts(self, *args, **kwargs)
        names = (
            "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
            "topk_ids", "activation", "global_num_experts", "local_num_experts",
            "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
        )
        values = dict(zip(names, args, strict=False)); values.update(kwargs)
        metadata = values.get("expert_tokens_meta")
        if metadata is None or metadata.expert_num_tokens_cpu is None:
            raise RuntimeError("live DeepEP expert CPU histogram is unavailable")
        histogram = [int(value) for value in metadata.expert_num_tokens_cpu.tolist()]
        if len(histogram) != int(values["local_num_experts"]):
            raise AssertionError((len(histogram), values["local_num_experts"]))
        from vllm.distributed import get_ep_group

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        result = original_experts(self, *args, **kwargs)
        end.record(torch.cuda.current_stream())
        config = _runtime_config(self, values)
        _PENDING.append({
            "start": start, "end": end, "wave": int(entry["wave"]),
            "request_id": entry["request_id"], "modality": entry["modality"],
            "pair_id": int(entry["pair_id"]), "token_bucket": entry["token_bucket"],
            "phase": entry["phase"], "iteration": int(entry["iteration"]),
            "measured": bool(entry["measured"]), "source_dp_rank": int(entry["source_dp_rank"]),
            "worker_dp_rank": int(os.environ["VLLM_DP_RANK"]),
            "ep_rank": int(get_ep_group().rank_in_group), "layer": layer,
            "expert_histogram": histogram,
            "total_assignments": int(sum(histogram)),
            "dispatched_rows": int(values["a1q"].shape[0]),
            "runtime_m": int(values["a1q"].shape[0]),
            "runtime_config": config,
            "expert_backend": type(self.fused_experts).__name__,
            "prepare_finalize_backend": type(self.prepare_finalize).__name__,
        })
        return result

    Qwen3MoeDecoderLayer.__init__ = patched_init
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_experts
