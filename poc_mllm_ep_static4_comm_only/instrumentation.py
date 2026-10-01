"""Physical Static-4 composition over the installed DeepEP modular MoE path.

The hook runs the full router once, partitions source token rows into exactly
four byte-balanced dispatches, executes a resident decode MoE path after S2,
then concatenates received prefill rows for one expert invocation.  Each
dispatch handle is retained and used for a per-slice combine, so this bounded
prototype is explicitly ``PROTO_PER_SLICE_COMBINE``.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import torch

from poc_mllm_ep_collective_granularity import instrumentation as prior
from poc_mllm_ep_communication_motivation import instrumentation as base
from poc_mllm_ep_static4_comm_only.contract import (
    DTYPE_BYTES,
    EXPERTS_PER_RANK,
    HIDDEN_SIZE,
)

_INSTALLED = False
_PENDING: list[tuple[dict[str, Any], dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]]] = []
_LOCK = threading.Lock()
_COMPLETED_WAVES: set[int] = set()


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _pair_begin() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    begin, end = _event(), _event()
    begin.record()
    return begin, end


def _pair_end(pair: tuple[torch.cuda.Event, torch.cuda.Event]) -> None:
    pair[1].record()


def _drain_static4() -> None:
    if not _PENDING:
        return
    torch.cuda.synchronize()
    root = Path(os.environ["MLLM_STATIC4_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"static4.pid{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record, events in _PENDING:
            record["cuda_ms"] = {
                name: float(begin.elapsed_time(end))
                for name, (begin, end) in events.items()
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    _PENDING.clear()


def _hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu", non_blocking=False).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _partition_rows(
    topk_ids: torch.Tensor,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Greedily byte-balance rows while keeping every row's top-k together."""
    ids = topk_ids.detach().to("cpu", non_blocking=False).to(torch.int64)
    assignments: list[list[int]] = [[] for _ in range(4)]
    byte_loads = [0, 0, 0, 0]
    per_rank = [[0, 0, 0, 0] for _ in range(4)]
    row_data: list[tuple[int, int, tuple[int, ...]]] = []
    row_bytes = HIDDEN_SIZE * DTYPE_BYTES
    for row_index, row in enumerate(ids):
        owners = tuple(
            sorted(
                {
                    int(expert) // EXPERTS_PER_RANK
                    for expert in row.tolist()
                    if int(expert) >= 0
                }
            )
        )
        row_data.append((len(owners) * row_bytes, row_index, owners))
    for cost, row_index, owners in sorted(row_data, key=lambda x: (-x[0], x[1])):
        target = min(range(4), key=lambda s: (byte_loads[s], len(assignments[s]), s))
        assignments[target].append(row_index)
        byte_loads[target] += cost
        for owner in owners:
            per_rank[target][owner] += row_bytes
    for rows in assignments:
        rows.sort()
    mean = statistics.fmean(byte_loads) if byte_loads else 0.0
    cv = statistics.pstdev(byte_loads) / mean if mean else 0.0
    positive = [value for value in byte_loads if value]
    ratio = max(positive) / min(positive) if positive else 1.0
    stats: list[dict[str, Any]] = []
    for slice_index, rows in enumerate(assignments):
        selected = ids[rows] if rows else ids[:0]
        valid = selected[selected.ge(0)]
        stats.append(
            {
                "slice": slice_index + 1,
                "source_token_rows": len(rows),
                "logical_dispatch_bytes": byte_loads[slice_index],
                "logical_dispatch_bytes_per_destination_rank": per_rank[slice_index],
                "source_active_experts": int(torch.unique(valid).numel()),
                "global_slice_byte_cv": cv,
                "global_slice_max_min_ratio": ratio,
            }
        )
    return assignments, stats


def _arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    names = (
        "hidden_states",
        "w1",
        "w2",
        "topk_ids",
        "topk_weights",
        "activation",
        "global_num_experts",
        "expert_map",
        "apply_router_weight_on_input",
        "shared_experts_input",
    )
    result = dict(zip(names, args, strict=False))
    result.update(kwargs)
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    result.setdefault("activation", MoEActivation.SILU)
    result.setdefault("global_num_experts", -1)
    result.setdefault("expert_map", None)
    result.setdefault("apply_router_weight_on_input", False)
    result.setdefault("shared_experts_input", None)
    return result


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    prior.install()
    atexit.register(_drain_static4)

    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        ExpertTokensMetadata,
        FusedMoEKernelModularImpl,
    )
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    native_apply = FusedMoEKernelModularImpl.apply

    @functools.wraps(native_apply)
    def apply(self: Any, *args: Any, **kwargs: Any):
        control = base._control()
        if not bool(control.get("static4_enabled", False)):
            return native_apply(self, *args, **kwargs)
        layer = int(getattr(base._TLS, "layer", -1))
        selected_layers = set(map(int, control.get("static4_layers", range(48))))
        if layer not in selected_layers:
            return native_apply(self, *args, **kwargs)
        context = getattr(base._TLS, "context", None) or {
            "iteration": int(control.get("outer_iteration", -1)),
            "phase": "external_dp_dummy",
            "decode_step": int(control.get("decode_progress", -1)),
            "states": [],
        }
        values = _arguments(args, kwargs)
        hidden = values["hidden_states"]
        topk_ids = values["topk_ids"]
        topk_weights = values["topk_weights"]
        iteration = int(context.get("iteration", -1))
        wave = int(control.get("wave", -1))
        # Once the one mixed iteration for a wave has been processed, later
        # decode-only forwards must not enter another cross-DP detector.  One
        # DP replica can become idle earlier than the other after the injected
        # request completes, so another all-reduce would be invalid there.
        if wave in _COMPLETED_WAVES:
            return native_apply(self, *args, **kwargs)
        search_start = int(control.get("static4_search_start", -1))
        search_end = int(control.get("static4_search_end", -2))
        if not search_start <= iteration <= search_end:
            return native_apply(self, *args, **kwargs)
        rows = int(hidden.shape[0])
        injected = base._fit_mask(
            getattr(base._TLS, "router_injected", None), rows, hidden.device
        ).to(device=hidden.device, non_blocking=True)
        # The serving scheduler knows the mixed iteration, but this hook runs
        # inside four independent worker processes.  A single layer-0 flag
        # reduction is an instrumentation-only way to make that scheduler
        # fact visible to every EP rank.  It is outside all reported per-layer
        # Static4 CUDA intervals and is reported as prototype wall overhead.
        if "_static4_active" not in context:
            import torch.distributed as dist

            active = torch.tensor(
                [1 if bool(injected.any().item()) else 0],
                dtype=torch.int32,
                device=hidden.device,
            )
            dist.all_reduce(active, group=get_ep_group().device_group)
            context["_static4_active"] = bool(active.item())
        if not bool(context["_static4_active"]):
            return native_apply(self, *args, **kwargs)
        if self.shared_experts is not None or values["shared_experts_input"] is not None:
            raise RuntimeError("Static4 PoC does not support shared experts")
        if bool(values["apply_router_weight_on_input"]):
            raise RuntimeError("Static4 PoC requires post-expert router weighting")

        # Correctness waves execute the native call once as a shadow reference.
        reference = None
        if bool(control.get("static4_correctness", False)):
            reference = native_apply(self, *args, **kwargs).detach().clone()

        prefill_rows = torch.nonzero(injected, as_tuple=False).flatten()
        decode_rows = torch.nonzero(~injected, as_tuple=False).flatten()
        build_begin = time.perf_counter_ns()
        local_assignments, slice_stats = _partition_rows(
            topk_ids.index_select(0, prefill_rows)
        )
        slice_rows = [
            prefill_rows.index_select(
                0, torch.tensor(indices, device=hidden.device, dtype=torch.long)
            )
            if indices
            else prefill_rows[:0]
            for indices in local_assignments
        ]
        slice_build_ms = (time.perf_counter_ns() - build_begin) / 1e6
        barrier_begin = time.perf_counter_ns()
        import torch.distributed as dist

        dist.barrier(group=get_ep_group().device_group)
        slice_ready_barrier_host_ms = (time.perf_counter_ns() - barrier_begin) / 1e6

        global_num_experts = int(values["global_num_experts"])
        if global_num_experts == -1:
            global_num_experts = int(values["w1"].shape[0])
        handle_index = int(dbo_current_ubatch_id())
        events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        prepared: list[dict[str, Any]] = []

        def dispatch_rows(name: str, indices: torch.Tensor) -> None:
            pair = _pair_begin()
            selected_hidden = hidden.index_select(0, indices)
            selected_weights = topk_weights.index_select(0, indices)
            selected_ids = topk_ids.index_select(0, indices)
            a1q, scale, meta, ids, weights = self._prepare(
                selected_hidden,
                selected_weights,
                selected_ids,
                global_num_experts,
                values["expert_map"],
                False,
            )
            _pair_end(pair)
            events[name] = pair
            if scale is not None:
                raise RuntimeError("BF16 Static4 unexpectedly produced activation scales")
            prepared.append(
                {
                    "source_rows": indices,
                    "source_hidden": selected_hidden,
                    "a1q": a1q,
                    "meta": meta,
                    "ids": ids,
                    "weights": weights,
                    "handle": self.prepare_finalize.handles[handle_index],
                }
            )

        method_begin_ns = time.perf_counter_ns()
        dispatch_rows("prefill_dispatch_s1", slice_rows[0])
        dispatch_rows("prefill_dispatch_s2", slice_rows[1])

        # Fixed decode priority window after S2: Dispatch -> Expert -> Combine.
        pair = _pair_begin()
        decode_hidden = hidden.index_select(0, decode_rows)
        decode_ids_input = topk_ids.index_select(0, decode_rows)
        decode_weights_input = topk_weights.index_select(0, decode_rows)
        d_a1q, d_scale, d_meta, d_ids, d_weights = self._prepare(
            decode_hidden,
            decode_weights_input,
            decode_ids_input,
            global_num_experts,
            values["expert_map"],
            False,
        )
        _pair_end(pair)
        events["decode_dispatch"] = pair
        if d_scale is not None:
            raise RuntimeError("BF16 decode unexpectedly produced activation scales")
        decode_handle = self.prepare_finalize.handles[handle_index]
        pair = _pair_begin()
        d_fused = self._fused_experts(
            in_dtype=hidden.dtype,
            a1q=d_a1q,
            a1q_scale=None,
            w1=values["w1"],
            w2=values["w2"],
            topk_weights=d_weights,
            topk_ids=d_ids,
            activation=values["activation"],
            global_num_experts=global_num_experts,
            local_num_experts=int(values["w1"].shape[0]),
            expert_map=values["expert_map"],
            apply_router_weight_on_input=False,
            expert_tokens_meta=d_meta,
        )
        _pair_end(pair)
        events["decode_expert"] = pair
        pair = _pair_begin()
        self.prepare_finalize.handles[handle_index] = decode_handle
        decode_output = torch.empty_like(decode_hidden)
        self._finalize(
            decode_output,
            d_fused,
            decode_hidden,
            d_weights,
            d_ids,
            False,
            shared_experts_input=None,
        )
        _pair_end(pair)
        events["decode_combine"] = pair

        dispatch_rows("prefill_dispatch_s3", slice_rows[2])
        dispatch_rows("prefill_dispatch_s4", slice_rows[3])

        pair = _pair_begin()
        received_rows = [int(item["a1q"].shape[0]) for item in prepared]
        full_a1q = torch.cat([item["a1q"] for item in prepared], dim=0)
        full_ids = torch.cat([item["ids"] for item in prepared], dim=0)
        full_weights = torch.cat([item["weights"] for item in prepared], dim=0)
        counts = [0] * int(values["w1"].shape[0])
        for item in prepared:
            meta = item["meta"]
            if meta is None or meta.expert_num_tokens_cpu is None:
                raise RuntimeError("Static4 requires per-expert token metadata")
            for index, count in enumerate(meta.expert_num_tokens_cpu.tolist()):
                counts[index] += int(count)
        full_meta = ExpertTokensMetadata.make_from_list(counts, device=hidden.device)
        _pair_end(pair)
        events["prefill_repack"] = pair

        pair = _pair_begin()
        prefill_fused = self._fused_experts(
            in_dtype=hidden.dtype,
            a1q=full_a1q,
            a1q_scale=None,
            w1=values["w1"],
            w2=values["w2"],
            topk_weights=full_weights,
            topk_ids=full_ids,
            activation=values["activation"],
            global_num_experts=global_num_experts,
            local_num_experts=int(values["w1"].shape[0]),
            expert_map=values["expert_map"],
            apply_router_weight_on_input=False,
            expert_tokens_meta=full_meta,
        )
        _pair_end(pair)
        events["prefill_expert_once"] = pair

        static_output = torch.empty_like(hidden)
        if decode_rows.numel():
            static_output.index_copy_(0, decode_rows, decode_output)
        cursor = 0
        for index, item in enumerate(prepared):
            width = received_rows[index]
            fused_slice = prefill_fused.narrow(0, cursor, width)
            cursor += width
            pair = _pair_begin()
            self.prepare_finalize.handles[handle_index] = item["handle"]
            slice_output = torch.empty_like(item["source_hidden"])
            self._finalize(
                slice_output,
                fused_slice,
                item["source_hidden"],
                item["weights"],
                item["ids"],
                False,
                shared_experts_input=None,
            )
            _pair_end(pair)
            events[f"prefill_combine_s{index + 1}"] = pair
            pair = _pair_begin()
            if item["source_rows"].numel():
                static_output.index_copy_(0, item["source_rows"], slice_output)
            _pair_end(pair)
            events[f"prefill_scatter_s{index + 1}"] = pair
        method_host_ms = (time.perf_counter_ns() - method_begin_ns) / 1e6

        record: dict[str, Any] = {
            "evidence": "MEASURED_TRUE_EP4_STATIC4_DEEPEP",
            "implementation": "STOCK_DEEPEP_PYTORCH_COMPOSITION",
            "combine_mode": "PROTO_PER_SLICE_COMBINE",
            "scope": "NATIVE_SERVING_PROTOTYPE_MOE_PATH",
            "wave": control.get("wave"),
            "condition": control.get("condition"),
            "replicate": control.get("replicate"),
            "iteration": int(context.get("iteration", -1)),
            "layer": layer,
            "ep_rank": int(get_ep_group().rank_in_group),
            "tp_rank": int(get_tp_group().rank_in_group),
            "dp_rank": int(get_dp_group().rank_in_group),
            "input_rows": rows,
            "prefill_source_rows": int(prefill_rows.numel()),
            "decode_source_rows": int(decode_rows.numel()),
            "slice_stats": slice_stats,
            "slice_build_host_ms": slice_build_ms,
            "slice_ready_barrier_host_ms": slice_ready_barrier_host_ms,
            "method_host_ms": method_host_ms,
            "received_rows_per_slice": received_rows,
            "prefill_local_expert_token_counts": counts,
            "prefill_local_distinct_active_experts": sum(
                1 for count in counts if count > 0
            ),
            "prefill_local_total_expert_paths": sum(counts),
            "prefill_expert_invocations": 1,
            "decode_expert_invocations": 1,
            "router_ids_sha256": _hash(topk_ids)
            if bool(control.get("static4_correctness", False))
            else None,
            "router_weights_sha256": _hash(topk_weights)
            if bool(control.get("static4_correctness", False))
            else None,
        }
        if reference is not None:
            difference = (static_output.float() - reference.float()).abs()
            denominator = reference.float().abs().clamp_min(1e-8)
            flat_static = static_output.float().flatten()
            flat_reference = reference.float().flatten()
            record["correctness"] = {
                "max_abs_error": float(difference.max().item()),
                "max_relative_error": float((difference / denominator).max().item()),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        flat_static, flat_reference, dim=0
                    ).item()
                ),
                "all_finite": bool(torch.isfinite(static_output).all().item()),
            }
        with _LOCK:
            _PENDING.append((record, events))
        if layer == max(selected_layers):
            _COMPLETED_WAVES.add(wave)
        if len(_PENDING) >= 48:
            _drain_static4()
        return static_output

    FusedMoEKernelModularImpl.apply = apply


__all__ = ["install", "_drain_static4"]
