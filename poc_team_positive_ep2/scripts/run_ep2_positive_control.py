#!/usr/bin/env python3
"""Semantics-preserving EP1/EP2/EP4 expert-parallel SDAR/TEAM diagnostic.

The released SDAR model stores all experts in a Python ``ModuleList``.  This
runner leaves the released decoder untouched and replaces only the physical
execution of each sparse MoE block after checkpoint load:

* rank 0 computes the released router and TEAM live/cold masks;
* token/expert rows are sent to the rank owning that expert with NCCL A2A;
* each rank invokes only its resident expert shard;
* weighted expert outputs return to rank 0 with NCCL A2A and are combined;
* the exact combined tensor is broadcast so the replicated non-MoE path stays
  bitwise aligned between ranks.

This is deliberately a reference EP substrate, not a production backend.
It exists to prove true ownership/remote dispatch/combine while preserving the
official TEAM decisions.  Clean timing and event-heavy structural timing are
separate modes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import socket
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_positive_control import load_official_generator, stop_ids, tokenize


EXPECTED_VISIBLE_BY_WORLD = {1: "0", 2: "0,1", 4: "0,1,2,3"}
EXPECTED_UUIDS_BY_WORLD = {
    1: (
        "f217c8a0-1142-20f4-d84b-af29f3a47a0d",
    ),
    2: (
        "f217c8a0-1142-20f4-d84b-af29f3a47a0d",
        "a77f3471-67d4-20b0-9fab-e502d4de5adb",
    ),
    4: (
        "f217c8a0-1142-20f4-d84b-af29f3a47a0d",
        "a77f3471-67d4-20b0-9fab-e502d4de5adb",
        "24200107-8a7f-de46-1bc8-b81f8d3af13e",
        "17488c15-2d4c-5d9e-d503-29b0d959a8a8",
    ),
}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RemovedExpert(nn.Module):
    """Parameter-free marker installed for experts owned by the other rank."""

    def forward(self, _x):  # pragma: no cover - reaching this is a runtime bug
        raise RuntimeError("attempted to execute a non-owned expert")


class ModuleEventRecorder:
    """Low-intrusion CUDA-event timing for non-MoE module boundaries."""

    def __init__(self):
        self.pending: dict[str, list[tuple[torch.cuda.Event, int]]] = defaultdict(list)
        self.pairs: list[tuple[str, int, torch.cuda.Event, torch.cuda.Event]] = []
        self.handles = []

    def add(self, module: nn.Module, name: str) -> None:
        def before(_module, _args):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            tensor = _args[0] if _args and torch.is_tensor(_args[0]) else None
            rows = int(tensor.numel() // tensor.shape[-1]) if tensor is not None else -1
            self.pending[name].append((event, rows))

        def after(_module, _args, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start, rows = self.pending[name].pop()
            self.pairs.append((name, rows, start, end))

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after))

    def finish(self) -> dict[str, Any]:
        torch.cuda.synchronize()
        values: dict[str, list[float]] = defaultdict(list)
        for name, rows, start, end in self.pairs:
            values[name].append(float(start.elapsed_time(end)))
            values[f"{name}_rows_{rows}"].append(float(start.elapsed_time(end)))
        for handle in self.handles:
            handle.remove()
        return {
            name: {
                "count": len(samples),
                "sum_ms": sum(samples),
                "mean_ms": sum(samples) / len(samples),
                "max_ms": max(samples),
                "samples_ms": samples,
            }
            for name, samples in values.items()
        }

    def reset(self) -> None:
        torch.cuda.synchronize()
        self.pending.clear()
        self.pairs.clear()


class EPRuntime:
    def __init__(self, rank: int, world: int, instrument: bool, local_expert_backend: str,
                 analyze_branch_duplicates: bool, capture_temporal: bool = False,
                 block_length: int = 32, local_draft_k: int = 1):
        self.rank = rank
        self.world = world
        self.instrument = instrument
        self.local_expert_backend = local_expert_backend
        self.analyze_branch_duplicates = analyze_branch_duplicates
        self.capture_temporal = capture_temporal
        self.block_length = block_length
        self.local_draft_k = local_draft_k
        self.execution_mode = "global"
        self.calls = 0
        self.layers: list[dict[str, Any]] = []
        self._events: list[tuple[dict[str, Any], dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]]] = []
        self.request_id: str | None = None
        self.forward_id = -1
        self.block_id = -1
        self.iteration = -1
        self.forward_phase = "unknown"
        self.store_kv = False
        self.masked_positions = -1
        self.accepted_since_previous = -1
        self._previous_refine_masked = -1
        self._history_by_layer: dict[
            tuple[str, int, int], list[dict[str, torch.Tensor]]
        ] = defaultdict(list)
        self.local_draft_rows: list[dict[str, Any]] = []

    def begin_request(self, request_id: str) -> None:
        self.request_id = request_id
        self.forward_id = -1
        self.block_id = -1
        self.iteration = -1
        self.forward_phase = "unknown"
        self.store_kv = False
        self.masked_positions = -1
        self.accepted_since_previous = -1
        self._previous_refine_masked = -1
        self._history_by_layer.clear()

    def begin_forward(self, input_ids: torch.Tensor | None, store_kv: bool) -> None:
        """Attach request/block/refinement identity to every MoE call.

        SDAR calls the complete model once per denoising iteration.  A
        ``store_kv=False`` call is a refinement step; the following
        ``store_kv=True`` call commits the completed block.  Prompt prefill is
        the only commit whose sequence length exceeds the generation block.
        """
        self.forward_id += 1
        self.store_kv = bool(store_kv)
        self.masked_positions = int((input_ids == 151669).sum().item()) if input_ids is not None else -1
        self.accepted_since_previous = -1
        sequence = int(input_ids.shape[-1]) if input_ids is not None else -1
        if store_kv and sequence > self.block_length:
            self.forward_phase = "prefill"
            self.iteration = -1
            return
        if not store_kv:
            if self.forward_phase in {"unknown", "prefill", "commit"}:
                self.block_id += 1
                self.iteration = 0
            else:
                self.iteration += 1
            self.forward_phase = "refine"
            if self._previous_refine_masked >= 0:
                self.accepted_since_previous = max(
                    self._previous_refine_masked - self.masked_positions, 0
                )
            self._previous_refine_masked = self.masked_positions
            return
        self.forward_phase = "commit"
        self._previous_refine_masked = -1
        self._history_by_layer.clear()

    @staticmethod
    def _cosine_mean(current: torch.Tensor, previous: torch.Tensor) -> float:
        current_f = current.float()
        previous_f = previous.float()
        return float(F.cosine_similarity(current_f, previous_f, dim=-1).mean().item())

    @staticmethod
    def _relative_l2_mean(current: torch.Tensor, previous: torch.Tensor) -> float:
        current_f = current.float()
        previous_f = previous.float()
        numerator = torch.linalg.vector_norm(current_f - previous_f, dim=-1)
        denominator = torch.linalg.vector_norm(previous_f, dim=-1).clamp_min(1e-12)
        return float((numerator / denominator).mean().item())

    def temporal_metrics(self, current, previous, block, lag: int) -> dict[str, Any]:
        current_selected = current["selected"]
        previous_selected = previous["selected"]
        ordered_equal = (current_selected == previous_selected).all(dim=-1)
        sorted_equal = (
            current_selected.sort(dim=-1).values
            == previous_selected.sort(dim=-1).values
        ).all(dim=-1)
        per_rank = block.num_experts // self.world
        current_owners = torch.div(current_selected, per_rank, rounding_mode="floor")
        previous_owners = torch.div(previous_selected, per_rank, rounding_mode="floor")
        current_presence = F.one_hot(
            current_owners, num_classes=self.world
        ).bool().any(dim=1)
        previous_presence = F.one_hot(
            previous_owners, num_classes=self.world
        ).bool().any(dim=1)
        owner_equal = (current_presence == previous_presence).all(dim=-1)
        matches = current_selected.unsqueeze(2) == previous_selected.unsqueeze(1)
        cur_index, cur_slot, prev_slot = torch.where(matches)
        if cur_index.numel():
            cur_branch = current["branch_output"][cur_index, cur_slot]
            prev_branch = previous["branch_output"][cur_index, prev_slot]
            branch_cosine = self._cosine_mean(cur_branch, prev_branch)
            branch_rel_l2 = self._relative_l2_mean(cur_branch, prev_branch)
        else:
            branch_cosine = float("nan")
            branch_rel_l2 = float("nan")
        current_load = current["rank_load"]
        previous_load = previous["rank_load"]
        prefix = f"lag{lag}"
        return {
            f"{prefix}_route_order_equal_fraction": float(ordered_equal.float().mean().item()),
            f"{prefix}_route_set_equal_fraction": float(sorted_equal.float().mean().item()),
            f"{prefix}_stable_branch_fraction": float(matches.any(dim=2).float().mean().item()),
            f"{prefix}_owner_set_equal_fraction": float(owner_equal.float().mean().item()),
            f"{prefix}_hidden_cosine_mean": self._cosine_mean(
                current["hidden"], previous["hidden"]
            ),
            f"{prefix}_hidden_rel_l2_mean": self._relative_l2_mean(
                current["hidden"], previous["hidden"]
            ),
            f"{prefix}_stable_branch_output_cosine_mean": branch_cosine,
            f"{prefix}_stable_branch_output_rel_l2_mean": branch_rel_l2,
            f"{prefix}_combined_output_cosine_mean": self._cosine_mean(
                current["combined"], previous["combined"]
            ),
            f"{prefix}_combined_output_rel_l2_mean": self._relative_l2_mean(
                current["combined"], previous["combined"]
            ),
            f"{prefix}_rank_load_cosine": float(F.cosine_similarity(
                current_load, previous_load, dim=0
            ).item()),
            f"{prefix}_same_critical_rank": bool(
                current_load.argmax().item() == previous_load.argmax().item()
            ),
        }

    def event_pair(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def install(self, model: nn.Module) -> dict[str, Any]:
        num_experts = int(model.config.num_experts)
        if num_experts != 128 or num_experts % self.world:
            raise RuntimeError(
                f"expected 128 experts divisible by EP degree {self.world}, got {num_experts}"
            )
        per_rank = num_experts // self.world
        ownership = {}
        for layer_id, layer in enumerate(model.model.layers):
            block = layer.mlp
            if len(block.experts) != num_experts:
                raise RuntimeError(f"layer {layer_id}: unexpected expert count")
            if self.local_expert_backend == "vllm_fused":
                local = list(range(self.rank * per_rank, (self.rank + 1) * per_rank))
                sample = block.experts[local[0]]
                intermediate = int(sample.gate_proj.weight.shape[0])
                hidden = int(sample.gate_proj.weight.shape[1])
                dtype = sample.gate_proj.weight.dtype
                device = sample.gate_proj.weight.device
                offloaded = None
                if self.world == 1 and device.type == "cuda":
                    # The complete checkpoint leaves less free HBM than the
                    # contiguous E=128 fused buffer.  Offload one layer's
                    # source experts before allocating its replacement; this
                    # preserves fast direct checkpoint loading while bounding
                    # peak GPU memory.
                    offloaded = []
                    for expert_id in local:
                        expert = block.experts[expert_id]
                        offloaded.append((
                            expert.gate_proj.weight.detach().cpu(),
                            expert.up_proj.weight.detach().cpu(),
                            expert.down_proj.weight.detach().cpu(),
                        ))
                        block.experts[expert_id] = RemovedExpert()
                    gc.collect()
                    torch.cuda.empty_cache()
                w1 = torch.empty(
                    (len(local), 2 * intermediate, hidden),
                    dtype=dtype,
                    device=device,
                )
                w2 = torch.empty(
                    (len(local), hidden, intermediate),
                    dtype=dtype,
                    device=device,
                )
                for local_id, expert_id in enumerate(local):
                    if offloaded is None:
                        expert = block.experts[expert_id]
                        gate_weight = expert.gate_proj.weight
                        up_weight = expert.up_proj.weight
                        down_weight = expert.down_proj.weight
                    else:
                        gate_weight, up_weight, down_weight = offloaded[local_id]
                    w1[local_id, :intermediate].copy_(gate_weight)
                    w1[local_id, intermediate:].copy_(up_weight)
                    w2[local_id].copy_(down_weight)
                    block.experts[expert_id] = RemovedExpert()
                block.register_buffer("_ep_w1", w1)
                block.register_buffer("_ep_w2", w2)
                del offloaded
                for expert_id in range(num_experts):
                    if not isinstance(block.experts[expert_id], RemovedExpert):
                        block.experts[expert_id] = RemovedExpert()
            else:
                for expert_id in range(num_experts):
                    owner = expert_id // per_rank
                    if owner != self.rank:
                        block.experts[expert_id] = RemovedExpert()
            block.forward = types.MethodType(self._make_forward(layer_id), block)
            ownership[layer_id] = [self.rank * per_rank, (self.rank + 1) * per_rank - 1]
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "num_experts": num_experts,
            "experts_per_rank": per_rank,
            "rank": self.rank,
            "owned_expert_range": [self.rank * per_rank, (self.rank + 1) * per_rank - 1],
            "layer_ownership": ownership,
            "local_expert_backend": self.local_expert_backend,
        }

    def _make_forward(self, layer_id: int):
        runtime = self

        def ep_forward(block, hidden_states, past_hidden_states=None,
                       decoded_index=None, expert_limit_index=None):
            return runtime.execute(
                layer_id, block, hidden_states, past_hidden_states,
                decoded_index, expert_limit_index,
            )

        return ep_forward

    def execute(self, layer_id, block, hidden_states, past_hidden_states,
                decoded_index, expert_limit_index):
        if self.execution_mode == "local_draft":
            return self.execute_local_draft(
                block, hidden_states, past_hidden_states, decoded_index
            )
        rank = self.rank
        device = hidden_states.device
        batch, sequence, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        num_rows = flat.shape[0]
        router_logits = torch.zeros(
            (num_rows, block.num_experts), dtype=hidden_states.dtype, device=device
        )
        final = torch.zeros_like(flat)

        if past_hidden_states is None:
            compute_mask = torch.ones(num_rows, dtype=torch.bool, device=device)
        else:
            if decoded_index is None:
                decoded_index = torch.zeros(
                    batch, sequence, dtype=torch.bool, device=device
                )
            compute_mask = (~decoded_index).reshape(-1)
            final[~compute_mask] = past_hidden_states.reshape(-1, hidden_dim)[~compute_mask]

        stage_events = {}
        if self.instrument:
            total_start, total_end = self.event_pair()
            router_start, router_end = self.event_pair()

        selected = torch.empty((0, block.top_k), dtype=torch.long, device=device)
        weights = torch.empty((0, block.top_k), dtype=hidden_states.dtype, device=device)
        compute_hidden = flat[compute_mask] if rank == 0 else torch.empty(
            (0, hidden_dim), dtype=flat.dtype, device=device
        )
        if rank == 0 and compute_hidden.shape[0]:
            compute_logits = block.gate(compute_hidden)
            probs = F.softmax(compute_logits, dim=1, dtype=torch.float)
            weights, selected = torch.topk(probs, block.top_k, dim=-1)
            if past_hidden_states is not None:
                if expert_limit_index is None:
                    expert_limit_index = torch.zeros(
                        batch, sequence, dtype=torch.bool, device=device
                    )
                limited = expert_limit_index.reshape(-1)[compute_mask]
                if limited.any():
                    unrestricted = selected[~limited]
                    necessary = torch.zeros(block.num_experts, dtype=torch.bool, device=device)
                    if unrestricted.numel():
                        necessary[unrestricted.flatten()] = True
                    probs[:, ~necessary] = 0
                    weights, selected = torch.topk(probs, block.top_k, dim=-1)
            if block.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            weights = weights.to(flat.dtype)
            router_logits[compute_mask] = compute_logits.to(flat.dtype)
        if self.instrument:
            router_end.record()
            prepare_start, prepare_end = self.event_pair()

        # Only rank 0 is the replicated-model source.  Sort branch rows by
        # destination so NCCL A2A receives contiguous per-owner segments.
        if rank == 0 and selected.numel():
            branch_token = torch.arange(
                selected.shape[0], device=device, dtype=torch.long
            ).repeat_interleave(block.top_k)
            expert_ids = selected.reshape(-1)
            branch_weights = weights.reshape(-1)
            exact_duplicate_assignments = 0
            near_duplicate_assignments_1pct = 0
            near_duplicate_assignments_5pct = 0
            route_duplicate_assignments = 0
            if self.analyze_branch_duplicates:
                compute_positions = torch.where(compute_mask)[0]
                original_positions = compute_positions.index_select(0, branch_token)
                branch_period = 32 if sequence % 32 == 0 and sequence >= 128 else sequence
                logical_positions = torch.remainder(original_positions, branch_period)
                position_expert_key = logical_positions * block.num_experts + expert_ids
                _, route_counts = torch.unique(position_expert_key, return_counts=True)
                route_duplicate_assignments = int((route_counts - 1).clamp_min(0).sum().item())
                for key in torch.unique(position_expert_key):
                    members = torch.where(position_expert_key == key)[0]
                    if members.numel() < 2:
                        continue
                    xs = compute_hidden.index_select(0, branch_token[members]).float()
                    reference = xs[0]
                    rel = torch.linalg.vector_norm(xs - reference, dim=-1) / torch.linalg.vector_norm(
                        reference
                    ).clamp_min(1e-12)
                    exact_duplicate_assignments += int((rel[1:] == 0).sum().item())
                    near_duplicate_assignments_1pct += int((rel[1:] <= 0.01).sum().item())
                    near_duplicate_assignments_5pct += int((rel[1:] <= 0.05).sum().item())
            owners = torch.div(
                expert_ids, block.num_experts // self.world, rounding_mode="floor"
            )
            order = torch.argsort(owners, stable=True)
            send_hidden = compute_hidden.index_select(0, branch_token[order]).contiguous()
            send_expert = expert_ids[order].contiguous()
            send_token = branch_token[order].contiguous()
            send_weight = branch_weights[order].contiguous()
            send_counts = torch.bincount(owners, minlength=self.world).to(torch.int64)
        else:
            send_hidden = torch.empty((0, hidden_dim), dtype=flat.dtype, device=device)
            send_expert = torch.empty((0,), dtype=torch.long, device=device)
            send_token = torch.empty((0,), dtype=torch.long, device=device)
            send_weight = torch.empty((0,), dtype=flat.dtype, device=device)
            send_counts = torch.zeros(self.world, dtype=torch.int64, device=device)
            route_duplicate_assignments = 0
            exact_duplicate_assignments = 0
            near_duplicate_assignments_1pct = 0
            near_duplicate_assignments_5pct = 0

        if self.world > 1:
            dist.broadcast(send_counts, src=0)
        counts = [int(x) for x in send_counts.cpu().tolist()]
        zero_splits = [0] * self.world
        input_splits = counts if rank == 0 else zero_splits
        output_splits = list(zero_splits)
        output_splits[0] = counts[rank]
        recv_rows = counts[rank]
        recv_hidden = torch.empty((recv_rows, hidden_dim), dtype=flat.dtype, device=device)
        recv_expert = torch.empty((recv_rows,), dtype=torch.long, device=device)
        recv_token = torch.empty((recv_rows,), dtype=torch.long, device=device)
        recv_weight = torch.empty((recv_rows,), dtype=flat.dtype, device=device)
        if self.instrument:
            prepare_end.record()
            dispatch_start, dispatch_end = self.event_pair()
        if self.world == 1:
            recv_hidden = send_hidden
            recv_expert = send_expert
            recv_token = send_token
            recv_weight = send_weight
        else:
            dist.all_to_all_single(recv_hidden, send_hidden,
                                   output_split_sizes=output_splits,
                                   input_split_sizes=input_splits)
            dist.all_to_all_single(recv_expert, send_expert,
                                   output_split_sizes=output_splits,
                                   input_split_sizes=input_splits)
            dist.all_to_all_single(recv_token, send_token,
                                   output_split_sizes=output_splits,
                                   input_split_sizes=input_splits)
            dist.all_to_all_single(recv_weight, send_weight,
                                   output_split_sizes=output_splits,
                                   input_split_sizes=input_splits)
        if self.instrument:
            dispatch_end.record()
            expert_start, expert_end = self.event_pair()

        local_result = torch.empty_like(recv_hidden)
        local_begin = rank * (block.num_experts // self.world)
        local_end = (rank + 1) * (block.num_experts // self.world)
        if recv_rows and self.local_expert_backend == "vllm_fused":
            from vllm.model_executor.layers.fused_moe import fused_experts
            local_result = fused_experts(
                hidden_states=recv_hidden,
                w1=block._ep_w1,
                w2=block._ep_w2,
                topk_weights=recv_weight.unsqueeze(-1),
                topk_ids=(recv_expert - local_begin).unsqueeze(-1),
                inplace=False,
                activation="silu",
                is_act_and_mul=True,
            )
        else:
            for expert_id in range(local_begin, local_end):
                positions = torch.where(recv_expert == expert_id)[0]
                if positions.numel():
                    values = block.experts[expert_id](recv_hidden.index_select(0, positions))
                    values = values * recv_weight.index_select(0, positions).unsqueeze(-1)
                    local_result.index_copy_(0, positions, values.to(flat.dtype))
        if self.instrument:
            expert_end.record()
            combine_start, combine_end = self.event_pair()

        reverse_input = list(zero_splits)
        reverse_input[0] = recv_rows
        reverse_output = counts if rank == 0 else zero_splits
        if self.world == 1:
            returned = local_result
        else:
            returned = torch.empty(
                (sum(counts) if rank == 0 else 0, hidden_dim),
                dtype=flat.dtype, device=device,
            )
            dist.all_to_all_single(returned, local_result,
                                   output_split_sizes=reverse_output,
                                   input_split_sizes=reverse_input)
        if rank == 0 and returned.numel():
            # Restore the released token-major/top-k branch order before the
            # reduction.  Otherwise EP degree changes the accumulation order
            # (owner-sorted under EP2/4), and tiny FP16 differences can change
            # the adaptive diffusion acceptance trajectory.
            returned_original_order = torch.empty_like(returned)
            returned_original_order.index_copy_(0, order, returned)
            compute_final = torch.zeros_like(compute_hidden)
            compute_final.index_add_(0, branch_token, returned_original_order)
            final[compute_mask] = compute_final
        else:
            returned_original_order = torch.empty(
                (0, hidden_dim), dtype=flat.dtype, device=device
            )
            compute_final = torch.empty(
                (0, hidden_dim), dtype=flat.dtype, device=device
            )
        if self.world > 1:
            dist.broadcast(final, src=0)
        if self.instrument:
            combine_end.record()
            total_end.record()

        row = {
            "call_id": self.calls,
            "request_id": self.request_id,
            "forward_id": self.forward_id,
            "block_id": self.block_id,
            "denoising_iteration": self.iteration,
            "forward_phase": self.forward_phase,
            "store_kv": self.store_kv,
            "masked_positions": self.masked_positions,
            "accepted_since_previous": self.accepted_since_previous,
            "layer": layer_id,
            "rank": rank,
            "physical_rows": int(num_rows),
            "computed_rows": int(compute_mask.sum().item()),
            "assignments": int(sum(counts)),
            "local_assignments": int(counts[rank]),
            "remote_assignments_from_source": int(sum(counts[1:])),
            "destination_rank_fanout": int(sum(count > 0 for count in counts)),
            "remote_destination_rank_fanout": int(sum(count > 0 for count in counts[1:])),
            "rank_assignment_counts": counts,
            "dispatch_bytes_hidden": int(sum(counts) * hidden_dim * flat.element_size()),
            "combine_bytes_hidden": int(sum(counts) * hidden_dim * flat.element_size()),
            "remote_dispatch_bytes_hidden": int(sum(counts[1:]) * hidden_dim * flat.element_size()),
            "remote_combine_bytes_hidden": int(sum(counts[1:]) * hidden_dim * flat.element_size()),
            "active_local_experts": int(torch.unique(recv_expert).numel()),
            "cross_branch_same_position_route_duplicates": route_duplicate_assignments,
            "cross_branch_exact_input_duplicates": exact_duplicate_assignments,
            "cross_branch_near_input_duplicates_1pct": near_duplicate_assignments_1pct,
            "cross_branch_near_input_duplicates_5pct": near_duplicate_assignments_5pct,
        }
        if (
            self.capture_temporal and rank == 0 and selected.numel()
            and self.forward_phase == "refine"
        ):
            key = (str(self.request_id), self.block_id, layer_id)
            current = {
                "selected": selected.detach().clone(),
                "weights": weights.detach().clone(),
                "hidden": compute_hidden.detach().clone(),
                "branch_output": returned_original_order.detach().reshape(
                    selected.shape[0], block.top_k, hidden_dim
                ).clone(),
                "combined": compute_final.detach().clone(),
                "rank_load": torch.tensor(counts, dtype=torch.float32, device=device),
            }
            history = self._history_by_layer[key]
            if self.forward_phase == "refine":
                for lag in (1, 2, 4, 8):
                    if len(history) >= lag:
                        row.update(self.temporal_metrics(current, history[-lag], block, lag))
            row["selected_experts"] = selected.detach().cpu().tolist()
            row["router_weights"] = weights.detach().float().cpu().tolist()
            top16_weights, top16_experts = torch.topk(
                probs, min(16, block.num_experts), dim=-1
            )
            row["router_top16_experts"] = top16_experts.detach().cpu().tolist()
            row["router_top16_weights"] = top16_weights.detach().float().cpu().tolist()
            history.append(current)
            if len(history) > 8:
                del history[:-8]
        self.calls += 1
        if self.instrument:
            stage_events = {
                "router": (router_start, router_end),
                "prepare": (prepare_start, prepare_end),
                "dispatch": (dispatch_start, dispatch_end),
                "expert": (expert_start, expert_end),
                "combine": (combine_start, combine_end),
                "moe": (total_start, total_end),
            }
            self._events.append((row, stage_events))
        else:
            self.layers.append(row)
        return final.reshape(batch, sequence, hidden_dim), router_logits

    def execute_local_draft(self, block, hidden_states, past_hidden_states, decoded_index):
        """Run an approximate rank-local MoE view with no EP communication.

        This path is observer-only: the generated sequence always consumes the
        subsequent exact global-EP result.  It quantifies both proposal quality
        and the real cost of treating resident physical shards as parallel
        diffusion views.
        """
        batch, sequence, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        final = torch.zeros_like(flat)
        if past_hidden_states is None:
            compute_mask = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
        else:
            if decoded_index is None:
                decoded_index = torch.zeros(
                    batch, sequence, dtype=torch.bool, device=flat.device
                )
            compute_mask = (~decoded_index).reshape(-1)
            final[~compute_mask] = past_hidden_states.reshape(-1, hidden_dim)[~compute_mask]
        router_logits = torch.zeros(
            (flat.shape[0], block.num_experts), dtype=flat.dtype, device=flat.device
        )
        compute_hidden = flat[compute_mask]
        if compute_hidden.numel():
            logits = block.gate(compute_hidden)
            local_count = block.num_experts // self.world
            local_begin = self.rank * local_count
            local_logits = logits[:, local_begin:local_begin + local_count]
            probabilities = F.softmax(local_logits, dim=-1, dtype=torch.float)
            draft_k = min(self.local_draft_k, local_count)
            weights, local_ids = torch.topk(probabilities, draft_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            from vllm.model_executor.layers.fused_moe import fused_experts
            values = fused_experts(
                hidden_states=compute_hidden,
                w1=block._ep_w1,
                w2=block._ep_w2,
                topk_weights=weights.to(flat.dtype),
                topk_ids=local_ids,
                inplace=False,
                activation="silu",
                is_act_and_mul=True,
            )
            final[compute_mask] = values.to(flat.dtype)
            router_logits[compute_mask] = logits.to(flat.dtype)
        return final.reshape(batch, sequence, hidden_dim), router_logits

    def finish(self) -> list[dict[str, Any]]:
        torch.cuda.synchronize()
        for row, pairs in self._events:
            for name, (start, end) in pairs.items():
                row[f"{name}_ms"] = float(start.elapsed_time(end))
            self.layers.append(row)
        self._events.clear()
        return self.layers

    def reset(self) -> None:
        torch.cuda.synchronize()
        self.calls = 0
        self.layers.clear()
        self._events.clear()
        self._history_by_layer.clear()
        self.local_draft_rows.clear()


def run_rank(rank: int, port: int, args) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=args.world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wrapper = (
        "huggingface_bd3_original.py" if args.mode == "baseline"
        else "huggingface_bd3_decoded_jump_expert_limit_speculative.py"
    )
    generator = load_official_generator(
        args.team_repo / "evaluation/opencompass/opencompass/models" / wrapper
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    # EP1 leaves the complete 30B checkpoint resident on one device.  Building
    # the contiguous fused-expert buffers after a direct GPU load temporarily
    # duplicates a whole layer's expert weights and exceeds 80 GB.  Convert the
    # expert layout on CPU first, then perform a single model transfer.  EP2/4
    # retain the faster direct-to-rank load path.
    # EP1's fused conversion is peak-memory safe inside EPRuntime.install: one
    # layer of source experts is temporarily offloaded before its contiguous
    # replacement is allocated.  This avoids a repeated 57-GiB CPU checkpoint
    # load while keeping the measured execution weights resident on GPU.
    convert_on_cpu = False
    load_device: str | int = "cpu" if convert_on_cpu else rank
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": load_device},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    validation_hidden = None
    validation_reference = None
    validation_reference_logits = None
    if args.validate_layer and convert_on_cpu:
        raise RuntimeError(
            "EP1 fused conversion cannot run the pre-conversion GPU layer validation; "
            "use output equality against validated EP2/EP4 instead"
        )
    if args.validate_layer:
        validation_hidden = torch.empty(
            (1, args.validation_rows, model.config.hidden_size),
            dtype=torch.float16, device=rank,
        )
        if rank == 0:
            generator_state = torch.Generator(device=rank).manual_seed(args.seed + 991)
            validation_hidden.normal_(mean=0.0, std=0.2, generator=generator_state)
        dist.broadcast(validation_hidden, src=0)
        if rank == 0:
            with torch.inference_mode():
                validation_reference, validation_reference_logits = model.model.layers[0].mlp(
                    validation_hidden
                )
                validation_reference = validation_reference.detach().clone()
                validation_reference_logits = validation_reference_logits.detach().clone()
        dist.barrier()
    runtime = EPRuntime(
        rank, args.world_size, args.instrument, args.local_expert_backend,
        args.analyze_branch_duplicates, args.capture_temporal, args.block_length,
        args.local_draft_k,
    )
    ownership = runtime.install(model)
    if convert_on_cpu:
        model.to(rank)
    original_model_forward = model.forward

    def traced_model_forward(*model_args, **model_kwargs):
        input_ids = model_kwargs.get("input_ids")
        if input_ids is None and model_args and torch.is_tensor(model_args[0]):
            input_ids = model_args[0]
        runtime.begin_forward(input_ids, bool(model_kwargs.get("store_kv", False)))
        store_kv = bool(model_kwargs.get("store_kv", False))
        if not args.capture_local_draft or store_kv:
            return original_model_forward(*model_args, **model_kwargs)

        local_start = torch.cuda.Event(enable_timing=True)
        local_end = torch.cuda.Event(enable_timing=True)
        exact_start = torch.cuda.Event(enable_timing=True)
        exact_end = torch.cuda.Event(enable_timing=True)
        runtime.execution_mode = "local_draft"
        local_start.record()
        local_output = original_model_forward(*model_args, **model_kwargs)
        local_end.record()
        mask = input_ids == 151669
        local_logits = local_output.logits[mask]
        proposal_post_start = torch.cuda.Event(enable_timing=True)
        proposal_post_end = torch.cuda.Event(enable_timing=True)
        proposal_post_start.record()
        local_probabilities = F.softmax(local_logits.float(), dim=-1)
        local_confidence, local_token = local_probabilities.max(dim=-1)
        local_top5 = torch.topk(local_logits, 5, dim=-1).indices
        proposal_post_end.record()
        runtime.execution_mode = "global"
        exact_start.record()
        exact_output = original_model_forward(*model_args, **model_kwargs)
        exact_end.record()

        exact_logits = exact_output.logits[mask]
        exact_probabilities = F.softmax(exact_logits.float(), dim=-1)
        exact_confidence, exact_token = exact_probabilities.max(dim=-1)
        exact_top5 = torch.topk(exact_logits, 5, dim=-1).indices
        gathered_tokens = [torch.empty_like(local_token) for _ in range(args.world_size)]
        gathered_confidence = [
            torch.empty_like(local_confidence) for _ in range(args.world_size)
        ]
        exchange_start = torch.cuda.Event(enable_timing=True)
        exchange_end = torch.cuda.Event(enable_timing=True)
        exchange_start.record()
        dist.all_gather(gathered_tokens, local_token)
        dist.all_gather(gathered_confidence, local_confidence)
        exchange_end.record()
        torch.cuda.synchronize(rank)
        timing = torch.tensor([
            local_start.elapsed_time(local_end),
            proposal_post_start.elapsed_time(proposal_post_end),
            exact_start.elapsed_time(exact_end),
            exchange_start.elapsed_time(exchange_end),
        ], dtype=torch.float32, device=rank)
        gathered_timing = [torch.empty_like(timing) for _ in range(args.world_size)]
        dist.all_gather(gathered_timing, timing)
        if rank == 0:
            proposal_tokens = torch.stack(gathered_tokens).detach().cpu().numpy()
            proposal_confidence = torch.stack(gathered_confidence).detach().cpu().numpy()
            exact_tokens = exact_token.detach().cpu().numpy()
            exact_conf = exact_confidence.detach().cpu().numpy()
            modal_tokens = []
            modal_counts = []
            for position in range(proposal_tokens.shape[1]):
                values, counts = np.unique(
                    proposal_tokens[:, position], return_counts=True
                )
                winner = int(counts.argmax())
                modal_tokens.append(int(values[winner]))
                modal_counts.append(int(counts[winner]))
            modal_tokens_array = np.asarray(modal_tokens)
            modal_counts_array = np.asarray(modal_counts)
            union_hits = np.asarray([
                exact_tokens[position] in proposal_tokens[:, position]
                for position in range(proposal_tokens.shape[1])
            ])
            all_agree = modal_counts_array == args.world_size
            majority_threshold = max(2, int(np.ceil(0.75 * args.world_size)))
            majority = modal_counts_array >= majority_threshold
            timing_by_rank = torch.stack(gathered_timing).detach().cpu().numpy()
            runtime.local_draft_rows.append({
                "request_id": runtime.request_id,
                "forward_id": runtime.forward_id,
                "block_id": runtime.block_id,
                "denoising_iteration": runtime.iteration,
                "masked_positions": runtime.masked_positions,
                "local_draft_k": args.local_draft_k,
                "local_draft_ms": float(local_start.elapsed_time(local_end)),
                "local_proposal_postprocess_ms": float(
                    proposal_post_start.elapsed_time(proposal_post_end)
                ),
                "global_exact_ms": float(exact_start.elapsed_time(exact_end)),
                "proposal_exchange_ms": float(exchange_start.elapsed_time(exchange_end)),
                "critical_local_draft_ms": float(timing_by_rank[:, 0].max()),
                "critical_local_proposal_postprocess_ms": float(timing_by_rank[:, 1].max()),
                "critical_global_exact_ms": float(timing_by_rank[:, 2].max()),
                "critical_proposal_exchange_ms": float(timing_by_rank[:, 3].max()),
                "timing_by_rank_ms": timing_by_rank.tolist(),
                "local_global_logits_cosine_mean_rank0": float(
                    F.cosine_similarity(local_logits.float(), exact_logits.float(), dim=-1)
                    .mean().item()
                ),
                "local_global_top5_overlap_mean_rank0": float(
                    (
                        local_top5.unsqueeze(-1) == exact_top5.unsqueeze(-2)
                    ).any(dim=-1).float().mean().item()
                ),
                "local_global_confidence_correlation_rank0": float(
                    torch.corrcoef(torch.stack((local_confidence, exact_confidence)))[0, 1].item()
                ) if local_confidence.numel() > 1 else None,
                "proposal_union_hit_fraction": float(union_hits.mean()),
                "all_rank_agreement_coverage": float(all_agree.mean()),
                "all_rank_agreement_precision": float(
                    (modal_tokens_array[all_agree] == exact_tokens[all_agree]).mean()
                ) if all_agree.any() else None,
                "three_quarter_agreement_coverage": float(majority.mean()),
                "three_quarter_agreement_precision": float(
                    (modal_tokens_array[majority] == exact_tokens[majority]).mean()
                ) if majority.any() else None,
                "per_rank_top1_match_fraction": [
                    float((proposal_tokens[proposal_rank] == exact_tokens).mean())
                    for proposal_rank in range(args.world_size)
                ],
                "exact_best_confidence_position": int(exact_conf.argmax()),
                "per_rank_best_confidence_position": [
                    int(proposal_confidence[proposal_rank].argmax())
                    for proposal_rank in range(args.world_size)
                ],
                "proposal_tokens": proposal_tokens.tolist(),
                "proposal_confidence": proposal_confidence.tolist(),
                "exact_tokens": exact_tokens.tolist(),
                "exact_confidence": exact_conf.tolist(),
            })
        return exact_output

    model.forward = traced_model_forward
    layer_validation = None
    if args.validate_layer:
        with torch.inference_mode():
            validation_actual, validation_actual_logits = model.model.layers[0].mlp(
                validation_hidden
            )
        torch.cuda.synchronize(rank)
        if rank == 0:
            ref = validation_reference.float().reshape(-1)
            actual = validation_actual.float().reshape(-1)
            layer_validation = {
                "layer": 0,
                "rows": args.validation_rows,
                "output_cosine": float(F.cosine_similarity(ref, actual, dim=0).item()),
                "output_rel_l2": float(
                    torch.linalg.vector_norm(actual - ref).div(
                        torch.linalg.vector_norm(ref).clamp_min(1e-12)
                    ).item()
                ),
                "output_max_abs": float((validation_actual.float() - validation_reference.float()).abs().max().item()),
                "router_logits_max_abs": float(
                    (validation_actual_logits.float() - validation_reference_logits.float()).abs().max().item()
                ),
            }
        runtime.reset()
    module_events = None
    if args.instrument:
        module_events = ModuleEventRecorder()
        module_events.add(model, "model_forward")
        for layer_id, layer in enumerate(model.model.layers):
            module_events.add(layer.self_attn, "attention")
            module_events.add(layer, "decoder_layer")
        module_events.add(model.lm_head, "lm_head")
    memory_after_shard = int(torch.cuda.memory_allocated(rank))

    if args.microbenchmark_only:
        if not args.instrument:
            raise RuntimeError("--microbenchmark-only requires --instrument")
        benchmark_rows = []
        layer = model.model.layers[args.microbenchmark_layer].mlp
        for rows in (int(value) for value in args.microbenchmark_rows.split(",")):
            generator_state = torch.Generator(device=rank).manual_seed(args.seed + rows)
            hidden = torch.empty(
                (1, rows, model.config.hidden_size), dtype=torch.float16, device=rank
            )
            if rank == 0:
                hidden.normal_(mean=0.0, std=0.2, generator=generator_state)
            dist.broadcast(hidden, src=0)
            runtime.begin_request(f"micro_m{rows}")
            for _ in range(args.microbenchmark_warmup):
                runtime.begin_forward(None, False)
                layer(hidden)
            torch.cuda.synchronize(rank)
            runtime.reset()
            runtime.begin_request(f"micro_m{rows}")
            for _ in range(args.microbenchmark_repeats):
                runtime.begin_forward(None, False)
                layer(hidden)
            for row in runtime.finish():
                row["microbenchmark_m"] = rows
                row["microbenchmark_layer"] = args.microbenchmark_layer
                benchmark_rows.append(row)
            runtime.reset()
        payload = {
            "mode": args.mode,
            "rank": rank,
            "physical_gpu": rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "backend": "torch.distributed.nccl_all_to_all_single_reference",
            "local_expert_backend": args.local_expert_backend,
            "topology": {"tp": 1, "dp": 1, "ep": args.world_size},
            "ownership": ownership,
            "memory_after_shard_bytes": memory_after_shard,
            "microbenchmark": True,
            "records": [],
            "stage_rows": benchmark_rows,
        }
        out = args.output.with_name(args.output.stem + f"_rank{rank}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.validation_only:
        payload = {
            "mode": args.mode,
            "rank": rank,
            "physical_gpu": rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "backend": "torch.distributed.nccl_all_to_all_single_reference",
            "local_expert_backend": args.local_expert_backend,
            "topology": {"tp": 1, "dp": 1, "ep": args.world_size},
            "ownership": ownership,
            "memory_after_shard_bytes": memory_after_shard,
            "layer_validation": layer_validation,
            "records": [],
            "stage_rows": [],
        }
        out = args.output.with_name(args.output.stem + f"_rank{rank}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        dist.barrier()
        dist.destroy_process_group()
        return

    prompts = [json.loads(x) for x in args.prompts.read_text().splitlines() if x.strip()]
    wanted = set(args.ids.split(",")) if args.ids else None
    if wanted is not None:
        prompts = [x for x in prompts if x.get("id") in wanted]
    if args.max_samples is not None:
        prompts = prompts[:args.max_samples]

    def generate_once(item):
        runtime.begin_request(str(item.get("id", "unknown")))
        tokens = tokenize(tokenizer, item["prompt"], model.device)
        output = generator(
            model=model,
            tokenizer=tokenizer,
            prompt=tokens,
            mask_id=151669,
            gen_length=args.gen_length,
            block_length=args.block_length,
            denoising_steps=args.denoising_steps,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            remasking="low_confidence",
            threshold=args.threshold,
            stopping_criteria_idx=(
                [] if args.disable_early_stop else stop_ids(tokenizer)
            ),
        )
        return output, int(tokens["input_ids"].shape[1])

    for i in range(args.warmup):
        generate_once(prompts[i % len(prompts)])
        torch.cuda.synchronize(rank)
        dist.barrier()
    runtime.reset()
    if module_events is not None:
        module_events.reset()

    records = []
    dist.barrier()
    request_start = time.perf_counter()
    for item in prompts:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        dist.barrier()
        start = time.perf_counter()
        output, prompt_length = generate_once(item)
        torch.cuda.synchronize(rank)
        elapsed = time.perf_counter() - start
        if rank == 0:
            decoded = tokenizer.batch_decode(
                output[:, prompt_length:], skip_special_tokens=False
            )[0].replace("<|MASK|>", "")
            records.append({
                "id": item.get("id"), "task": item.get("task"),
                "reference": item.get("reference"), "elapsed_s": elapsed,
                "output": decoded,
            })
    request_wall = time.perf_counter() - request_start
    layers = runtime.finish()
    module_summary = module_events.finish() if module_events is not None else None

    payload = {
        "mode": args.mode,
        "rank": rank,
        "physical_gpu": rank,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "backend": "torch.distributed.nccl_all_to_all_single_reference",
        "local_expert_backend": args.local_expert_backend,
        "topology": {"tp": 1, "dp": 1, "ep": args.world_size},
        "ownership": ownership,
        "memory_after_shard_bytes": memory_after_shard,
        "wrapper": wrapper,
        "generation": {
            "gen_length": args.gen_length, "block_length": args.block_length,
            "denoising_steps": args.denoising_steps, "threshold": args.threshold,
            "disable_early_stop": args.disable_early_stop,
        },
        "instrument": args.instrument,
        "request_wall_s": request_wall,
        "records": records if rank == 0 else None,
        "stage_rows": layers,
        "local_draft_rows": runtime.local_draft_rows if rank == 0 else None,
        "module_event_summary": module_summary,
        "layer_validation": layer_validation,
    }
    out = args.output.with_name(args.output.stem + f"_rank{rank}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "team"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--team-repo", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument(
        "--local-expert-backend", choices=["python", "vllm_fused"],
        default="vllm_fused",
    )
    parser.add_argument("--analyze-branch-duplicates", action="store_true")
    parser.add_argument("--capture-temporal", action="store_true")
    parser.add_argument("--capture-local-draft", action="store_true")
    parser.add_argument("--local-draft-k", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--validate-layer", action="store_true")
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--microbenchmark-only", action="store_true")
    parser.add_argument("--microbenchmark-rows", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--microbenchmark-warmup", type=int, default=5)
    parser.add_argument("--microbenchmark-repeats", type=int, default=30)
    parser.add_argument("--microbenchmark-layer", type=int, default=24)
    parser.add_argument("--world-size", type=int, choices=[1, 2, 4], default=2)
    args = parser.parse_args()
    if args.validation_only and not args.validate_layer:
        raise SystemExit("--validation-only requires --validate-layer")
    if args.capture_local_draft and args.world_size < 2:
        raise SystemExit("--capture-local-draft requires EP2 or EP4")
    expected_visible = EXPECTED_VISIBLE_BY_WORLD[args.world_size]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly {expected_visible}")
    if torch.cuda.device_count() != args.world_size:
        raise SystemExit(
            f"expected {args.world_size} visible GPUs, got {torch.cuda.device_count()}"
        )
    actual_uuids = tuple(
        str(torch.cuda.get_device_properties(i).uuid) for i in range(args.world_size)
    )
    if actual_uuids != EXPECTED_UUIDS_BY_WORLD[args.world_size]:
        raise SystemExit(f"unexpected physical GPU mapping: {actual_uuids}")
    mp.spawn(run_rank, args=(free_port(), args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
