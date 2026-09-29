"""Low-overhead DeepEP timing and exact modality-route accounting.

CUDA events surround the installed DeepEP prepare, sparse expert kernel, and
finalize calls.  Route-isolated prefill replays retain the seed run's expert
IDs/weights and merely mask rows by the processor-provided multimodal mask.
They are diagnostic physical replays, not semantic model variants.
"""

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

from .contract import DTYPE_BYTES, EXPERTS_PER_RANK, HIDDEN_SIZE, NUM_EXPERTS

_INSTALLED = False
_TLS = threading.local()
_PENDING: list[tuple[dict[str, Any], list[torch.cuda.Event]]] = []
_ROUTES: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
_ITERATION = -1
_WAVE = None


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_COMM_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _shard(mask: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    world = int(get_tensor_model_parallel_world_size())
    rank = int(get_tensor_model_parallel_rank())
    remainder = int(mask.shape[0]) % world
    if remainder:
        padded = torch.zeros(
            mask.shape[0] + world - remainder, dtype=mask.dtype, device=mask.device
        )
        padded[: mask.shape[0]] = mask
        mask = padded
    width = int(mask.shape[0]) // world
    return mask.narrow(0, rank * width, width)


def _drain() -> None:
    if not _PENDING:
        return
    torch.cuda.synchronize()
    root = Path(os.environ["MLLM_COMM_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"stage.pid{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record, events in _PENDING:
            record.update(
                {
                    "dispatch_ms": float(events[0].elapsed_time(events[1])),
                    "expert_execution_ms": float(events[1].elapsed_time(events[2])),
                    "combine_ms": float(events[2].elapsed_time(events[3])),
                    "moe_ms": float(events[0].elapsed_time(events[3])),
                }
            )
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    _PENDING.clear()


def _route_stats(
    ids: torch.Tensor,
    visual: torch.Tensor,
    text: torch.Tensor,
    injected: torch.Tensor | None,
) -> dict[str, Any]:
    ids_cpu = ids.detach().to("cpu", non_blocking=False).to(torch.int64)
    masks = {
        "visual": visual.detach().to("cpu", non_blocking=False).bool(),
        "text": text.detach().to("cpu", non_blocking=False).bool(),
        "special_other": (~visual & ~text)
        .detach()
        .to("cpu", non_blocking=False)
        .bool(),
    }
    if injected is not None:
        masks["injected"] = injected.detach().to("cpu", non_blocking=False).bool()
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        selected = ids_cpu[mask]
        valid = selected[selected.ge(0)]
        hist = torch.bincount(valid, minlength=NUM_EXPERTS).tolist()
        pairs_rank = [
            sum(hist[r * EXPERTS_PER_RANK : (r + 1) * EXPERTS_PER_RANK])
            for r in range(4)
        ]
        destination_rows = [0, 0, 0, 0]
        for row in selected:
            owners = {
                int(expert) // EXPERTS_PER_RANK
                for expert in row.tolist()
                if expert >= 0
            }
            for owner in owners:
                destination_rows[owner] += 1
        bytes_rank = [rows * HIDDEN_SIZE * DTYPE_BYTES for rows in destination_rows]
        result[name] = {
            "tokens": int(mask.sum()),
            "pairs": int(valid.numel()),
            "expert_histogram": hist,
            "pairs_per_rank": pairs_rank,
            "unique_token_destination_rows_per_rank": destination_rows,
            "dispatch_payload_bytes_per_rank": bytes_rank,
            "combine_payload_bytes_per_rank": bytes_rank,
            "dispatch_payload_bytes": sum(bytes_rank),
            "combine_payload_bytes": sum(bytes_rank),
        }
    return result


def _fit_mask(
    mask: torch.Tensor | None, rows: int, device: torch.device
) -> torch.Tensor:
    if mask is None:
        return torch.zeros(rows, dtype=torch.bool, device=device)
    if mask.shape[0] != rows:
        mask = _shard(mask)
    if mask.shape[0] != rows:
        raise RuntimeError(f"modality mask {tuple(mask.shape)} != route rows {rows}")
    return mask


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(_drain)

    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeDecoderLayer,
        Qwen3MoeSparseMoeBlock,
    )
    from vllm.model_executor.models.qwen3_vl_moe import (
        Qwen3MoeLLMModel,
        Qwen3VLMoeForConditionalGeneration,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_execute = GPUModelRunner.execute_model
    original_forward = GPUModelRunner._model_forward
    original_embed = Qwen3VLMoeForConditionalGeneration.embed_input_ids
    original_model = Qwen3MoeLLMModel.forward
    original_block_init = Qwen3MoeSparseMoeBlock.__init__
    original_block = Qwen3MoeSparseMoeBlock.forward
    original_decoder_init = Qwen3MoeDecoderLayer.__init__
    original_decoder = Qwen3MoeDecoderLayer.forward
    original_select = BaseRouter.select_experts
    original_apply = FusedMoEKernelModularImpl.apply
    original_prepare = FusedMoEKernelModularImpl._prepare
    original_experts = FusedMoEKernelModularImpl._fused_experts
    original_finalize = FusedMoEKernelModularImpl._finalize

    @functools.wraps(original_execute)
    def execute(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
        global _ITERATION, _WAVE
        control = _control()
        prior_bypass = getattr(_TLS, "bypass", False)
        _TLS.bypass = bool(control.get("instrumentation_bypass", False))
        if _TLS.bypass:
            try:
                return original_execute(self, scheduler_output, *args, **kwargs)
            finally:
                _TLS.bypass = prior_bypass
        wave = control.get("wave")
        if wave != _WAVE:
            _WAVE, _ITERATION = wave, -1
        _ITERATION += 1
        scheduled = dict(scheduler_output.num_scheduled_tokens)
        prior = getattr(_TLS, "context", None)
        _TLS.context = {
            "scheduled": scheduled,
            "control": control,
            "iteration": _ITERATION,
            "phase": str(control.get("expected_phase", "unknown")),
            "states": [],
        }
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            _TLS.context = prior
            _TLS.bypass = prior_bypass

    @functools.wraps(original_forward)
    def runner_forward(self: Any, *args: Any, **kwargs: Any):
        context = getattr(_TLS, "context", None)
        if context is not None:
            states = []
            ordered = []
            for request_id, scheduled in context["scheduled"].items():
                index = self.input_batch.req_id_to_index.get(request_id)
                if index is None:
                    continue
                computed = int(self.input_batch.num_computed_tokens_cpu[index])
                prompt = int(self.input_batch.num_prompt_tokens[index])
                row = {
                    "request_id": request_id,
                    "scheduled": int(scheduled),
                    "computed": computed,
                    "prompt": prompt,
                    "index": int(index),
                }
                states.append(row)
                ordered.append(row)
            ordered.sort(key=lambda row: row["index"])
            phase = (
                "decode"
                if states and all(r["computed"] >= r["prompt"] for r in states)
                else (
                    "prefill"
                    if states and all(r["computed"] < r["prompt"] for r in states)
                    else "mixed"
                )
            )
            context.update(
                {
                    "states": states,
                    "ordered_states": ordered,
                    "phase": phase,
                    "decode_step": min(
                        (
                            r["computed"] - r["prompt"] + 1
                            for r in states
                            if r["computed"] >= r["prompt"]
                        ),
                        default=-1,
                    ),
                }
            )
        return original_forward(self, *args, **kwargs)

    def embed(
        self: Any,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ):
        if getattr(_TLS, "bypass", False):
            return original_embed(
                self, input_ids, multimodal_embeddings, is_multimodal=is_multimodal
            )
        _TLS.pending_visual = (
            None if is_multimodal is None else is_multimodal.reshape(-1)
        )
        _TLS.pending_ids = input_ids.reshape(-1)
        return original_embed(
            self, input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )

    def model(self: Any, input_ids: torch.Tensor | None, *args: Any, **kwargs: Any):
        if getattr(_TLS, "bypass", False):
            return original_model(self, input_ids, *args, **kwargs)
        inputs = kwargs.get("inputs_embeds")
        rows = (
            int(input_ids.numel())
            if input_ids is not None
            else (0 if inputs is None else int(inputs.shape[0]))
        )
        # vLLM may retain input_ids on CPU while passing GPU inputs_embeds.
        # Modality masks must follow the hidden-state device.
        visual = getattr(_TLS, "pending_visual", None)
        device = (
            visual.device
            if visual is not None
            else (
                inputs.device
                if inputs is not None
                else (input_ids.device if input_ids is not None else None)
            )
        )
        ids = (
            input_ids.reshape(-1)
            if input_ids is not None
            else getattr(_TLS, "pending_ids", None)
        )
        visual = _fit_mask(visual, rows, device)
        device = visual.device
        active_context = getattr(_TLS, "context", None) or {}
        special_ids = set(
            map(int, active_context.get("control", {}).get("special_token_ids", []))
        )
        special = torch.zeros(rows, dtype=torch.bool, device=device)
        if ids is not None and special_ids:
            ids = ids.reshape(-1)
            if ids.shape[0] != rows:
                ids = _shard(ids)
            ids = ids.to(device=device, non_blocking=True)
            for value in special_ids:
                special |= ids.eq(value)
        text = ~visual & ~special
        injected = torch.zeros(rows, dtype=torch.bool, device=device)
        context = getattr(_TLS, "context", None)
        if context is not None and rows:
            cursor = 0
            raw = torch.zeros(
                sum(r["scheduled"] for r in context.get("ordered_states", [])),
                dtype=torch.bool,
                device=device,
            )
            for state in context.get("ordered_states", []):
                count = state["scheduled"]
                injection_ids = {
                    str(value)
                    for value in context["control"].get("injection_request_ids", [])
                }
                if str(state["request_id"]) in injection_ids:
                    raw[cursor : cursor + count] = True
                cursor += count
            if raw.numel():
                injected = _fit_mask(raw, rows, device)
        prior = (
            getattr(_TLS, "visual", None),
            getattr(_TLS, "text", None),
            getattr(_TLS, "injected", None),
        )
        _TLS.visual, _TLS.text, _TLS.injected = visual, text, injected
        try:
            return original_model(self, input_ids, *args, **kwargs)
        finally:
            _TLS.visual, _TLS.text, _TLS.injected = prior
            _TLS.pending_visual = _TLS.pending_ids = None

    def block_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_block_init(self, *args, **kwargs)
        match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", str(kwargs.get("prefix", "")))
        self._comm_layer = int(match.group(1)) if match else -1

    def block(self: Any, hidden_states: torch.Tensor):
        if getattr(_TLS, "bypass", False):
            return original_block(self, hidden_states)
        rows = int(hidden_states.shape[0])
        device = hidden_states.device
        prior = (
            getattr(_TLS, "router_visual", None),
            getattr(_TLS, "router_text", None),
            getattr(_TLS, "router_injected", None),
            getattr(_TLS, "layer", -1),
        )
        _TLS.router_visual = _fit_mask(getattr(_TLS, "visual", None), rows, device)
        _TLS.router_text = _fit_mask(getattr(_TLS, "text", None), rows, device)
        _TLS.router_injected = _fit_mask(getattr(_TLS, "injected", None), rows, device)
        _TLS.layer = int(getattr(self, "_comm_layer", -1))
        try:
            return original_block(self, hidden_states)
        finally:
            (_TLS.router_visual, _TLS.router_text, _TLS.router_injected, _TLS.layer) = (
                prior
            )

    def decoder_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_decoder_init(self, *args, **kwargs)
        match = re.search(
            r"(?:layers|h)\.(\d+)(?:\.|$)",
            str(kwargs.get("prefix", args[1] if len(args) > 1 else "")),
        )
        self._comm_decoder_layer = int(match.group(1)) if match else -1

    def decoder(self: Any, *args: Any, **kwargs: Any):
        if getattr(_TLS, "bypass", False):
            return original_decoder(self, *args, **kwargs)
        prior = getattr(_TLS, "layer", -1)
        _TLS.layer = int(getattr(self, "_comm_decoder_layer", -1))
        try:
            return original_decoder(self, *args, **kwargs)
        finally:
            _TLS.layer = prior

    def select(
        self: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids=None,
    ):
        if getattr(_TLS, "bypass", False):
            return original_select(
                self, hidden_states, router_logits, input_ids=input_ids
            )
        weights, ids = original_select(
            self, hidden_states, router_logits, input_ids=input_ids
        )
        context = getattr(_TLS, "context", None)
        layer = int(getattr(_TLS, "layer", -1))
        if context is None or layer < 0:
            return weights, ids
        control = context["control"]
        condition = str(control.get("condition", "NATIVE"))
        case_key = str(control.get("case_key", "default"))
        key = (case_key, layer)
        if condition == "FULL_SEED":
            _ROUTES[key] = (weights.detach().clone(), ids.detach().clone())
        elif condition in {"FULL_REPLAY", "VISUAL_ONLY", "TEXT_ONLY"}:
            if key not in _ROUTES:
                raise RuntimeError(f"missing frozen route {key}")
            weights, ids = (tensor.clone() for tensor in _ROUTES[key])
        visual = _fit_mask(
            getattr(_TLS, "router_visual", None), ids.shape[0], ids.device
        )
        text = _fit_mask(getattr(_TLS, "router_text", None), ids.shape[0], ids.device)
        injected = _fit_mask(
            getattr(_TLS, "router_injected", None), ids.shape[0], ids.device
        )
        if condition == "VISUAL_ONLY":
            ids = torch.where(
                visual.to(ids.device)[:, None], ids, torch.full_like(ids, -1)
            )
            weights = torch.where(
                visual.to(weights.device)[:, None], weights, torch.zeros_like(weights)
            )
        elif condition == "TEXT_ONLY":
            ids = torch.where(
                text.to(ids.device)[:, None], ids, torch.full_like(ids, -1)
            )
            weights = torch.where(
                text.to(weights.device)[:, None], weights, torch.zeros_like(weights)
            )
        selected_iterations = set(map(int, control.get("capture_iterations", [])))
        if bool(control.get("route_stats", False)) and (
            not selected_iterations or context["iteration"] in selected_iterations
        ):
            _TLS.route_record = _route_stats(ids, visual, text, injected)
        return weights, ids

    def apply(self: Any, *args: Any, **kwargs: Any):
        if getattr(_TLS, "bypass", False):
            return original_apply(self, *args, **kwargs)
        context = getattr(_TLS, "context", None)
        if context is None:
            # External-DP ranks with no local request participate through
            # execute_dummy_batch, which bypasses GPUModelRunner.execute_model.
            # Time that rank as part of the physical EP4 critical path.
            dummy_control = _control()
            if not bool(dummy_control.get("capture_stage", False)):
                return original_apply(self, *args, **kwargs)
            context = {
                "control": dummy_control,
                "iteration": 0,
                "phase": "external_dp_dummy",
                "states": [],
                "decode_step": -1,
            }
        control = context["control"]
        selected_iterations = set(map(int, control.get("capture_iterations", [])))
        selected = bool(control.get("capture_stage", False)) and (
            not selected_iterations or context["iteration"] in selected_iterations
        )
        if not selected:
            return original_apply(self, *args, **kwargs)
        events = [_event() for _ in range(4)]
        record = {
            "evidence": "MEASURED_TRUE_EP4_DEEPEP_CUDA_EVENTS",
            "wave": control.get("wave"),
            "condition": control.get("condition"),
            "case_key": control.get("case_key"),
            "replicate": control.get("replicate"),
            "iteration": context["iteration"],
            "phase": context["phase"],
            "decode_step": context.get("decode_step", -1),
            "layer": int(getattr(_TLS, "layer", -1)),
            "ep_rank": int(get_ep_group().rank_in_group),
            "tp_rank": int(get_tp_group().rank_in_group),
            "dp_rank": int(get_dp_group().rank_in_group),
            "request_ids": [row["request_id"] for row in context.get("states", [])],
            "scheduled_tokens": sum(
                row["scheduled"] for row in context.get("states", [])
            ),
        }
        route = getattr(_TLS, "route_record", None)
        if route is not None:
            record["route_stats"] = route
        _TLS.stage = (record, events)
        try:
            result = original_apply(self, *args, **kwargs)
            _PENDING.append(_TLS.stage)
            if len(_PENDING) >= 48:
                _drain()
            return result
        finally:
            _TLS.stage = None
            _TLS.route_record = None

    def prepare(self: Any, *args: Any, **kwargs: Any):
        stage = getattr(_TLS, "stage", None)
        if stage:
            stage[1][0].record()
        result = original_prepare(self, *args, **kwargs)
        if stage:
            stage[1][1].record()
        return result

    def experts(self: Any, *args: Any, **kwargs: Any):
        stage = getattr(_TLS, "stage", None)
        result = original_experts(self, *args, **kwargs)
        if stage:
            names = (
                "in_dtype",
                "a1q",
                "a1q_scale",
                "w1",
                "w2",
                "topk_weights",
                "topk_ids",
                "activation",
                "global_num_experts",
                "local_num_experts",
                "expert_map",
                "apply_router_weight_on_input",
                "expert_tokens_meta",
            )
            values = dict(zip(names, args, strict=False))
            values.update(kwargs)
            meta = values.get("expert_tokens_meta")
            histogram = (
                []
                if meta is None or meta.expert_num_tokens_cpu is None
                else [int(value) for value in meta.expert_num_tokens_cpu.tolist()]
            )
            stage[0].update(
                {
                    "local_token_expert_pairs": sum(histogram),
                    "local_distinct_active_experts": sum(
                        value > 0 for value in histogram
                    ),
                    "local_expert_histogram": histogram,
                    "prepare_finalize_backend": type(self.prepare_finalize).__name__,
                    "expert_backend": type(self.fused_experts).__name__,
                    "topology": {
                        "tp": get_tp_group().world_size,
                        "dp": get_dp_group().world_size,
                        "ep": get_ep_group().world_size,
                    },
                }
            )
            stage[1][2].record()
        return result

    def finalize(self: Any, *args: Any, **kwargs: Any):
        stage = getattr(_TLS, "stage", None)
        result = original_finalize(self, *args, **kwargs)
        if stage:
            stage[1][3].record()
        return result

    GPUModelRunner.execute_model = execute
    GPUModelRunner._model_forward = runner_forward
    Qwen3VLMoeForConditionalGeneration.embed_input_ids = embed
    Qwen3MoeLLMModel.forward = model
    Qwen3MoeSparseMoeBlock.__init__ = block_init
    Qwen3MoeSparseMoeBlock.forward = block
    Qwen3MoeDecoderLayer.__init__ = decoder_init
    Qwen3MoeDecoderLayer.forward = decoder
    BaseRouter.select_experts = select
    FusedMoEKernelModularImpl.apply = apply
    FusedMoEKernelModularImpl._prepare = prepare
    FusedMoEKernelModularImpl._fused_experts = experts
    FusedMoEKernelModularImpl._finalize = finalize
