"""Low-overhead sampled TRUE DeepEP decode stage timing/counter hook."""
from __future__ import annotations

import atexit
import functools
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

_INSTALLED = False
_TLS = threading.local()
_PENDING: list[tuple[dict[str, Any], list[torch.cuda.Event]]] = []
_FLUSHED = False


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_RANK_STAGE_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED or not _PENDING:
        return
    _FLUSHED = True
    torch.cuda.synchronize()
    root = Path(os.environ["MLLM_RANK_STAGE_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"worker_pid{os.getpid()}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record, events in _PENDING:
            record.update({
                "dispatch_ms": float(events[0].elapsed_time(events[1])),
                "expert_execution_ms": float(events[1].elapsed_time(events[2])),
                "combine_ms": float(events[2].elapsed_time(events[3])),
                "moe_ms": float(events[0].elapsed_time(events[3])),
            })
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(_flush)
    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_execute = GPUModelRunner.execute_model
    original_forward = GPUModelRunner._model_forward
    original_init = Qwen3MoeDecoderLayer.__init__
    original_layer = Qwen3MoeDecoderLayer.forward
    original_apply = FusedMoEKernelModularImpl.apply
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    @functools.wraps(original_execute)
    def execute(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        prior = getattr(_TLS, "context", None)
        scheduled = dict(scheduler_output.num_scheduled_tokens)
        _TLS.context = {"scheduled": scheduled, "selected": False, "phase": "unknown"}
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            _TLS.context = prior

    @functools.wraps(original_forward)
    def model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = getattr(_TLS, "context", None)
        if context is not None:
            states = []
            for request_id, scheduled in context["scheduled"].items():
                index = self.input_batch.req_id_to_index.get(request_id)
                if index is None:
                    continue
                computed = int(self.input_batch.num_computed_tokens_cpu[index])
                prompt = int(self.input_batch.num_prompt_tokens[index])
                states.append({"request_id": request_id, "scheduled": int(scheduled),
                               "computed": computed, "prompt": prompt})
            control = _control()
            decode = bool(states) and all(row["computed"] >= row["prompt"] for row in states)
            decode_step = min((row["computed"] - row["prompt"] + 1 for row in states), default=-1)
            context.update({"states": states, "phase": "decode" if decode else "prefill",
                            "decode_step": decode_step, "batch": len(states),
                            "wave": int(control.get("wave", -1)),
                            "condition": control.get("condition"),
                            "selected": bool(control.get("capture")) and decode
                                        and decode_step in set(map(int, control.get("steps", [])))})
        return original_forward(self, *args, **kwargs)

    def decoder_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
        self._rank_stage_layer = int(match.group(1)) if match else -1

    def decoder_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        prior = getattr(_TLS, "layer", -1)
        _TLS.layer = int(getattr(self, "_rank_stage_layer", -1))
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            _TLS.layer = prior

    def apply(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = getattr(_TLS, "context", None)
        if not context or not context.get("selected"):
            return original_apply(self, *args, **kwargs)
        events = [_event() for _ in range(4)]
        _TLS.stage = ({
            "evidence": "MEASURED_TRUE_DEEPEP_SAMPLED_DECODE_STAGE",
            "wave": context["wave"], "condition": context["condition"],
            "decode_step": context["decode_step"], "batch": context["batch"],
            "layer": int(getattr(_TLS, "layer", -1)),
            "request_ids": [row["request_id"] for row in context["states"]],
            "per_expert_weight_bytes": 9437184,
            "ep_rank": int(get_ep_group().rank_in_group),
            "tp_rank": int(get_tp_group().rank_in_group),
            "dp_rank": int(get_dp_group().rank_in_group),
        }, events)
        try:
            result = original_apply(self, *args, **kwargs)
            _PENDING.append(_TLS.stage)
            return result
        finally:
            _TLS.stage = None

    def prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        stage = getattr(_TLS, "stage", None)
        if stage: stage[1][0].record()
        result = original_prepare(self, *args, **kwargs)
        if stage: stage[1][1].record()
        return result

    def experts(self: Any, *args: Any, **kwargs: Any) -> Any:
        stage = getattr(_TLS, "stage", None)
        result = original_experts(self, *args, **kwargs)
        if stage:
            names = ("in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                     "topk_ids", "activation", "global_num_experts", "local_num_experts",
                     "expert_map", "apply_router_weight_on_input", "expert_tokens_meta")
            values = dict(zip(names, args, strict=False)); values.update(kwargs)
            meta = values.get("expert_tokens_meta")
            histogram = [] if meta is None or meta.expert_num_tokens_cpu is None else [
                int(value) for value in meta.expert_num_tokens_cpu.tolist()]
            active = sum(value > 0 for value in histogram)
            stage[0].update({
                "token_expert_pairs": sum(histogram), "distinct_active_experts": active,
                "active_expert_ids_local": [i for i, value in enumerate(histogram) if value > 0],
                "expert_histogram": histogram,
                "max_tokens_per_expert": max(histogram, default=0),
                "mean_tokens_per_active_expert": sum(histogram) / max(active, 1),
                "active_expert_weight_bytes": active * 9437184,
                "routed_hidden_bytes": sum(histogram) * 2048 * 2,
                "combine_output_bytes": sum(histogram) * 2048 * 2,
                "prepare_finalize_backend": type(self.prepare_finalize).__name__,
                "expert_backend": type(self.fused_experts).__name__,
                "topology": {"tp": get_tp_group().world_size,
                             "dp": get_dp_group().world_size,
                             "ep": get_ep_group().world_size},
            })
            stage[1][2].record()
        return result

    def finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        stage = getattr(_TLS, "stage", None)
        result = original_finalize(self, *args, **kwargs)
        if stage: stage[1][3].record()
        return result

    GPUModelRunner.execute_model = execute
    GPUModelRunner._model_forward = model_forward
    Qwen3MoeDecoderLayer.__init__ = decoder_init
    Qwen3MoeDecoderLayer.forward = decoder_forward
    FusedMoEKernelModularImpl.apply = apply
    FusedMoEKernelModularImpl._prepare = prepare
    FusedMoEKernelModularImpl._fused_experts = experts
    FusedMoEKernelModularImpl._finalize = finalize
