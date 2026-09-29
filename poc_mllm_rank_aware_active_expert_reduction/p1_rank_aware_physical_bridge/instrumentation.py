"""Route intervention plus deferred CUDA-event timing for TRUE EP4.

Each measured wave uses identical prompts and forced tokens.  The same-pair
collapse transforms the current native route only within each owner rank, so
it preserves every destination-rank assignment and path count within that
invocation while changing the number of distinct experts touched.
"""
from __future__ import annotations

import atexit
import functools
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

_INSTALLED = False
_FLUSHED = False
_TLS = threading.local()
_PENDING: list[tuple[dict[str, Any], list[torch.cuda.Event]]] = []


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_P1_BRIDGE_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _renormalize(weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    output = torch.where(ids.ge(0), weights, torch.zeros_like(weights))
    return output / output.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(output.dtype).tiny
    )


def _tail_k6(ids: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    changed_ids = ids.clone()
    changed_ids[:, 6:] = -1
    return changed_ids, _renormalize(weights, changed_ids)


def _rank_k6(ids: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove two paths/row, prioritizing closure on crowded owner ranks.

    This is a non-deployable physical-headroom policy.  It uses the current
    frozen wave's complete route tensor, but no output/damage information.
    """
    flat = ids.reshape(-1)
    freq = torch.bincount(flat, minlength=128)
    active = freq.gt(0).reshape(4, 32).sum(dim=1)
    owners = torch.div(ids, 32, rounding_mode="floor")
    # Prefer rare experts on the currently most populated owner rank; native
    # weight is only a deterministic tie-break.  Selecting two per row keeps
    # path reduction exactly matched to K6 native tail.
    score = (
        active[owners].to(torch.float32) * 1_000_000.0
        - freq[ids].to(torch.float32) * 1_000.0
        - weights.to(torch.float32)
    )
    removed = torch.topk(score, k=2, dim=-1, largest=True, sorted=False).indices
    changed_ids = ids.clone()
    changed_ids.scatter_(1, removed, -1)
    return changed_ids, _renormalize(weights, changed_ids)


def _collapse_same_pairs(
    ids: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce active experts ~25% without changing any owner-rank or path count."""
    flat = ids.reshape(-1)
    freq = torch.bincount(flat, minlength=128)
    changed = ids.clone()
    for rank in range(4):
        start = rank * 32
        local = freq[start : start + 32]
        active_ids = torch.nonzero(local.gt(0), as_tuple=False).flatten()
        active_count = int(active_ids.numel())
        if active_count <= 1:
            continue
        keep_count = max(1, int(math.ceil(0.75 * active_count)))
        order = torch.topk(local, k=keep_count, largest=True, sorted=True).indices
        keep = torch.zeros(32, device=ids.device, dtype=torch.bool)
        keep[order] = True
        anchor = start + int(order[0])
        mask = torch.div(changed, 32, rounding_mode="floor").eq(rank)
        local_ids = changed - start
        mask &= ~keep[local_ids.clamp(0, 31)]
        changed[mask] = anchor
    # IDs move only within their original owner rank; weights and the number
    # of paths are intentionally identical.
    return changed, weights.clone()


def _variants(ids: torch.Tensor, weights: torch.Tensor) -> dict[str, torch.Tensor]:
    tail_ids, tail_weights = _tail_k6(ids, weights)
    rank_ids, rank_weights = _rank_k6(ids, weights)
    collapse_ids, collapse_weights = _collapse_same_pairs(ids, weights)
    return {
        "baseline_ids": ids.detach().clone(),
        "baseline_weights": weights.detach().clone(),
        "tail_ids": tail_ids.detach(),
        "tail_weights": tail_weights.detach(),
        "rank_ids": rank_ids.detach(),
        "rank_weights": rank_weights.detach(),
        "collapse_ids": collapse_ids.detach(),
        "collapse_weights": collapse_weights.detach(),
    }


def _drain() -> None:
    """Materialize completed deferred events without relying on worker atexit.

    vLLM's worker shutdown does not guarantee that Python ``atexit`` handlers
    run in the process that owns the CUDA events.  Draining once per sampled
    48-layer decode step keeps synchronization out of the per-layer hot path
    while making the evidence durable before worker teardown.
    """
    if not _PENDING:
        return
    torch.cuda.synchronize()
    root = Path(os.environ["MLLM_P1_BRIDGE_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"stage.pid{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record, events in _PENDING:
            record.update({
                "dispatch_ms": float(events[0].elapsed_time(events[1])),
                "expert_execution_ms": float(events[1].elapsed_time(events[2])),
                "combine_ms": float(events[2].elapsed_time(events[3])),
                "decode_moe_ms": float(events[0].elapsed_time(events[3])),
            })
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    _PENDING.clear()


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    _drain()


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(_flush)

    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_execute = GPUModelRunner.execute_model
    original_forward = GPUModelRunner._model_forward
    original_init = Qwen3MoeDecoderLayer.__init__
    original_layer = Qwen3MoeDecoderLayer.forward
    original_select = BaseRouter.select_experts
    original_apply = FusedMoEKernelModularImpl.apply
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    @functools.wraps(original_execute)
    def execute(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        prior = getattr(_TLS, "context", None)
        _TLS.context = {
            "scheduled": dict(scheduler_output.num_scheduled_tokens),
            "phase": "unknown",
            "selected": False,
        }
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
                states.append((request_id, computed, prompt, int(scheduled)))
            control = _control()
            decode = bool(states) and all(row[1] >= row[2] for row in states)
            step = min((row[1] - row[2] + 1 for row in states), default=-1)
            context.update({
                "phase": "decode" if decode else "prefill",
                "decode_step": step,
                "states": states,
                "condition": str(control.get("condition", "UNKNOWN")),
                "pair_key": str(control.get("pair_key", "seed")),
                "replicate": int(control.get("replicate", -1)),
                "selected": bool(control.get("capture"))
                and decode
                and step in set(map(int, control.get("steps", []))),
            })
        return original_forward(self, *args, **kwargs)

    def decoder_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
        self._p1_layer = int(match.group(1)) if match else -1

    def decoder_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        prior = getattr(_TLS, "layer", -1)
        _TLS.layer = int(getattr(self, "_p1_layer", -1))
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            _TLS.layer = prior

    def select(self: Any, hidden_states: torch.Tensor, router_logits: torch.Tensor, *, input_ids=None):
        weights, ids = original_select(self, hidden_states, router_logits, input_ids=input_ids)
        context = getattr(_TLS, "context", None)
        layer = int(getattr(_TLS, "layer", -1))
        if not context or context.get("phase") != "decode" or layer < 0:
            return weights, ids
        condition = context["condition"]
        if condition == "SEED":
            return weights, ids
        # Each condition begins from the same prompt and forced tokens.  Apply
        # the transformation to the current native router result.  This avoids
        # reusing a tensor whose row count can differ when the external-DP
        # scheduler pads an otherwise identical decode wave differently.
        frozen = _variants(ids, weights)
        prefix = {
            "NATIVE_K8": "baseline",
            "NATIVE_TAIL_K6": "tail",
            "RANK_AWARE_K6": "rank",
            "SAME_PAIR_COLLAPSE25": "collapse",
        }.get(condition)
        if prefix is None:
            raise RuntimeError(f"unknown P1 condition {condition}")
        return frozen[f"{prefix}_weights"], frozen[f"{prefix}_ids"]

    def apply(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = getattr(_TLS, "context", None)
        if not context or not context.get("selected"):
            return original_apply(self, *args, **kwargs)
        events = [_event() for _ in range(4)]
        _TLS.stage = ({
            "evidence": "MEASURED_TRUE_EP4_FROZEN_INPUT_ROUTE_INTERVENTION",
            "condition": context["condition"],
            "replicate": context["replicate"],
            "decode_step": context["decode_step"],
            "layer": int(getattr(_TLS, "layer", -1)),
            "request_ids": [row[0] for row in context["states"]],
            "batch": len(context["states"]),
            "ep_rank": int(get_ep_group().rank_in_group),
            "tp_rank": int(get_tp_group().rank_in_group),
            "dp_rank": int(get_dp_group().rank_in_group),
        }, events)
        try:
            result = original_apply(self, *args, **kwargs)
            _PENDING.append(_TLS.stage)
            # One selected decode step contributes 48 layer records in each
            # worker.  Persist at that boundary rather than trusting atexit.
            if len(_PENDING) >= 48:
                _drain()
            return result
        finally:
            _TLS.stage = None

    def prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        stage = getattr(_TLS, "stage", None)
        if stage:
            stage[1][0].record()
        result = original_prepare(self, *args, **kwargs)
        if stage:
            stage[1][1].record()
        return result

    def experts(self: Any, *args: Any, **kwargs: Any) -> Any:
        stage = getattr(_TLS, "stage", None)
        result = original_experts(self, *args, **kwargs)
        if stage:
            names = (
                "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                "topk_ids", "activation", "global_num_experts", "local_num_experts",
                "expert_map", "apply_router_weight_on_input", "expert_tokens_meta",
            )
            values = dict(zip(names, args, strict=False))
            values.update(kwargs)
            meta = values.get("expert_tokens_meta")
            histogram = [] if meta is None or meta.expert_num_tokens_cpu is None else [
                int(value) for value in meta.expert_num_tokens_cpu.tolist()
            ]
            active = sum(value > 0 for value in histogram)
            stage[0].update({
                "token_expert_pairs": sum(histogram),
                "distinct_active_experts": active,
                "expert_histogram": histogram,
                "max_tokens_per_expert": max(histogram, default=0),
                "active_weight_bytes": active * 9437184,
                "topology": {"tp": get_tp_group().world_size,
                             "dp": get_dp_group().world_size,
                             "ep": get_ep_group().world_size},
            })
            stage[1][2].record()
        return result

    def finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        stage = getattr(_TLS, "stage", None)
        result = original_finalize(self, *args, **kwargs)
        if stage:
            stage[1][3].record()
        return result

    GPUModelRunner.execute_model = execute
    GPUModelRunner._model_forward = model_forward
    Qwen3MoeDecoderLayer.__init__ = decoder_init
    Qwen3MoeDecoderLayer.forward = decoder_forward
    BaseRouter.select_experts = select
    FusedMoEKernelModularImpl.apply = apply
    FusedMoEKernelModularImpl._prepare = prepare
    FusedMoEKernelModularImpl._fused_experts = experts
    FusedMoEKernelModularImpl._finalize = finalize
