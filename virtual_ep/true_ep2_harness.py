"""Minimal TP1/EP2 LLaDA2 routed-MoE isolation harness.

This deliberately bypasses SGLang's TP-coupled model-parallel wiring while
using its DeepEP dispatcher.  Dense, attention, router, and shared-expert
weights retain TP1 semantics.  Only routed experts are sharded.

The replicated-state bridge all-gather is a separate category.  It is never
included in EP dispatch/combine or in simulated EP-stage time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
import json

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class EPIsolationTiming:
    request_id: int
    rank: int
    layer: int
    invocation: int
    physical_rows_global: int
    source_rows_local: int
    source_assignments_to_rank0: int
    source_assignments_to_rank1: int
    source_unique_activations_to_rank0: int
    source_unique_activations_to_rank1: int
    source_remote_assignments: int
    source_remote_unique_activations: int
    recv_rows: int
    local_active_experts: int
    local_expert_assignments: int
    dispatch_ms: float
    routed_expert_compute_ms: float
    combine_ms: float
    replicated_state_bridge_allgather_ms: float
    ep_stage_ms_excluding_bridge: float
    ep_isolation_harness_moe_wall_ms: float


class _ManualForwardBatch:
    def __init__(self, rows: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.forward_mode = ForwardMode.EXTEND
        self.is_extend_in_batch = True
        self.num_token_non_padded = rows


def _elapsed(events, begin: int, end: int) -> float:
    return float(events[begin].elapsed_time(events[end]))


class TrueEP2Harness:
    """Install an exact two-rank routed-MoE forward into the official model."""

    def __init__(self, model, output_dir: Path, trace_timing: bool = True):
        if not dist.is_initialized() or dist.get_world_size() != 2:
            raise RuntimeError("EP-isolation harness requires a real world_size=2")
        self.model = model
        self.rank = dist.get_rank()
        self.world_size = 2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_timing = trace_timing
        self.request_id = -1
        self.invocations = [0 for _ in model.model.layers]
        self._prepare_layers()

    def _prepare_layers(self) -> None:
        from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
        from sglang.srt.layers.moe.utils import DeepEPMode

        config = self.model.config
        if int(config.num_experts) != 256 or int(config.num_experts_per_tok) != 8:
            raise ValueError("harness is intentionally bounded to LLaDA2.0-mini")
        experts_per_rank = int(config.num_experts) // self.world_size
        local_begin = self.rank * experts_per_rank
        local_end = local_begin + experts_per_rank
        ownership = {
            "topology": "DP1_TP1_SP1_EP2",
            "harness": "EP-isolation harness",
            "rank": self.rank,
            "world_size": self.world_size,
            "local_expert_begin": local_begin,
            "local_expert_end_exclusive": local_end,
            "local_expert_count": experts_per_rank,
            "source_partition": "contiguous_equal_rows_inside_routed_moe",
            "replicated_state_bridge": "all_gather; separate non-EP cost category",
        }
        (self.output_dir / f"ownership_rank{self.rank}.json").write_text(
            json.dumps(ownership, indent=2) + "\n"
        )
        for layer_id, layer in enumerate(self.model.model.layers):
            block = layer.mlp
            if not hasattr(block, "experts"):
                continue
            for expert_id in range(int(config.num_experts)):
                if not local_begin <= expert_id < local_end:
                    block.experts[expert_id] = nn.Identity()
            block._ep_local_begin = local_begin
            block._ep_local_end = local_end
            block._ep_layer_id = layer_id
            block._ep_dispatcher = DeepEPDispatcher(
                group=dist.group.WORLD,
                router_topk=int(config.num_experts_per_tok),
                permute_fusion=False,
                num_experts=int(config.num_experts),
                num_local_experts=experts_per_rank,
                hidden_size=int(config.hidden_size),
                params_dtype=torch.bfloat16,
                deepep_mode=DeepEPMode.NORMAL,
                async_finish=True,
                return_recv_hook=True,
            )
            block.forward = MethodType(self._make_forward(layer_id), block)
        torch.cuda.empty_cache()
        dist.barrier()

    def _make_forward(self, layer_id: int):
        harness = self

        def ep_forward(block, hidden_states):
            identity = hidden_states
            batch, sequence, hidden = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden)
            if flat.shape[0] % harness.world_size:
                raise RuntimeError(
                    f"routed rows {flat.shape[0]} not divisible by EP2 source partition"
                )
            rows_per_rank = flat.shape[0] // harness.world_size
            local_hidden = flat.narrow(0, harness.rank * rows_per_rank, rows_per_rank)

            # Router/shared paths retain unsharded TP1 semantics on each rank.
            topk_ids, topk_weights, router_logits = block.gate(hidden_states)
            local_ids = topk_ids.reshape(-1, topk_ids.shape[-1]).narrow(
                0, harness.rank * rows_per_rank, rows_per_rank
            )
            local_weights = topk_weights.reshape(-1, topk_weights.shape[-1]).narrow(
                0, harness.rank * rows_per_rank, rows_per_rank
            )
            local_owners = torch.div(local_ids, block._ep_local_end - block._ep_local_begin,
                                     rounding_mode="floor")
            assignment_by_destination = torch.bincount(
                local_owners.reshape(-1), minlength=harness.world_size
            )
            unique_by_destination = torch.zeros(
                harness.world_size, dtype=torch.int64, device=local_ids.device
            )
            for destination in range(harness.world_size):
                unique_by_destination[destination] = (
                    local_owners == destination
                ).any(dim=1).sum()
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            events[0].record()
            recv_hidden, recv_ids, recv_weights, _ = block._ep_dispatcher.dispatch(
                hidden_states=local_hidden,
                input_global_scale=None,
                topk_idx=local_ids,
                topk_weights=local_weights,
                forward_batch=_ManualForwardBatch(rows_per_rank),
            )
            events[1].record()

            # Match the official HF path within each destination: place every
            # branch back in its original top-k slot, weight/sum in FP32, then
            # cast the per-destination partial result to BF16 for DeepEP.
            recv_branches = torch.zeros(
                (recv_hidden.shape[0], recv_ids.shape[1], recv_hidden.shape[1]),
                dtype=recv_hidden.dtype,
                device=recv_hidden.device,
            )
            local_assignments = 0
            local_active = 0
            for local_id, global_id in enumerate(
                range(block._ep_local_begin, block._ep_local_end)
            ):
                matches = recv_ids == local_id
                if not matches.any():
                    continue
                local_active += 1
                row_index, slot_index = matches.nonzero(as_tuple=True)
                local_assignments += int(row_index.numel())
                expert_output = block.experts[global_id](recv_hidden[row_index])
                recv_branches[row_index, slot_index] = expert_output
            recv_output = (
                recv_branches.to(recv_weights.dtype)
                .mul_(recv_weights.unsqueeze(-1))
                .sum(dim=1)
                .to(recv_hidden.dtype)
            )
            events[2].record()

            # BF16 manual expert execution already reduces contributions per
            # received token, so use DeepEP's exact combine core as dInfer does.
            normal = block._ep_dispatcher._normal_dispatcher
            local_routed, combine_event = normal._combine_core(
                recv_output, previous_event=None
            )
            combine_event.current_stream_wait()
            normal.handle = None
            block._ep_dispatcher._stage = block._ep_dispatcher._stage.__class__.INITIAL
            events[3].record()

            full_routed = torch.empty_like(flat)
            dist.all_gather_into_tensor(
                full_routed, local_routed.contiguous(), group=dist.group.WORLD
            )
            events[4].record()
            events[4].synchronize()
            if harness.trace_timing:
                dispatch_ms = _elapsed(events, 0, 1)
                expert_ms = _elapsed(events, 1, 2)
                combine_ms = _elapsed(events, 2, 3)
                bridge_ms = _elapsed(events, 3, 4)
                timing = EPIsolationTiming(
                    request_id=harness.request_id,
                    rank=harness.rank,
                    layer=layer_id,
                    invocation=harness.invocations[layer_id],
                    physical_rows_global=int(flat.shape[0]),
                    source_rows_local=int(rows_per_rank),
                    source_assignments_to_rank0=int(assignment_by_destination[0]),
                    source_assignments_to_rank1=int(assignment_by_destination[1]),
                    source_unique_activations_to_rank0=int(unique_by_destination[0]),
                    source_unique_activations_to_rank1=int(unique_by_destination[1]),
                    source_remote_assignments=int(
                        assignment_by_destination[1 - harness.rank]
                    ),
                    source_remote_unique_activations=int(
                        unique_by_destination[1 - harness.rank]
                    ),
                    recv_rows=int(recv_hidden.shape[0]),
                    local_active_experts=local_active,
                    local_expert_assignments=local_assignments,
                    dispatch_ms=dispatch_ms,
                    routed_expert_compute_ms=expert_ms,
                    combine_ms=combine_ms,
                    replicated_state_bridge_allgather_ms=bridge_ms,
                    ep_stage_ms_excluding_bridge=dispatch_ms + expert_ms + combine_ms,
                    ep_isolation_harness_moe_wall_ms=dispatch_ms + expert_ms + combine_ms + bridge_ms,
                )
                with (harness.output_dir / f"timing_rank{harness.rank}.jsonl").open("a") as stream:
                    stream.write(json.dumps(asdict(timing)) + "\n")
            harness.invocations[layer_id] += 1
            routed = full_routed.view(batch, sequence, hidden)
            if block.config.num_shared_experts is not None:
                routed = routed + block.shared_experts(identity)
            return routed, (
                router_logits.view(batch, sequence, -1),
                topk_ids.view(batch, sequence, -1),
            )

        return ep_forward
