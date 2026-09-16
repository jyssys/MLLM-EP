#!/usr/bin/env python3
"""Replay measured routes through true EP2 DeepEP + BF16 grouped expert MLP.

This is the held-out validation substrate for the routed-MoE critical path.
It measures exactly three categories: EP dispatch, routed expert compute, and
EP combine.  No replicated-state bridge is present.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from virtual_ep.mapping import SourcePartition
from virtual_ep.schema import TraceBundle, invocation_slices


NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 2048
INTERMEDIATE = 512
LOCAL_EXPERTS = 128


class _ForwardBatch:
    def __init__(self, rows: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.forward_mode = ForwardMode.EXTEND
        self.is_extend_in_batch = True
        self.num_token_non_padded = rows


def _grouped_mlp(x, counts, gate_up_weights, down_weights):
    offsets = counts.cumsum(0, dtype=torch.int32)
    gate_up = torch._grouped_mm(x, gate_up_weights, offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    return torch._grouped_mm(activated, down_weights, offs=offsets)


def _cases(trace: TraceBundle, maximum: int) -> list[dict]:
    candidates = []
    for key, selected in invocation_slices(trace.rows):
        source_keys = trace.rows["source_partition_key"][selected]
        ids = trace.rows["expert_ids"][selected]
        candidates.append(
            {
                "key": key,
                "source_keys": source_keys,
                "expert_ids": ids,
                "rows": len(ids),
            }
        )
    if len(candidates) <= maximum:
        return candidates

    def select(group: list[dict], count: int) -> list[dict]:
        selected: list[dict] = []
        selected_ids: set[int] = set()

        def add(case: dict) -> None:
            if id(case) not in selected_ids and len(selected) < count:
                selected_ids.add(id(case))
                selected.append(case)

        for ordering in (
            sorted(group, key=lambda case: case["rows"]),
            sorted(group, key=lambda case: (case["key"][3], case["key"][4])),
        ):
            for quantile in np.linspace(0.0, 1.0, max(2, count // 2)):
                add(ordering[round(quantile * (len(ordering) - 1))])
        for case in group:
            add(case)
        return selected

    train = [case for case in candidates if case["key"][0] % 5 != 0]
    heldout = [case for case in candidates if case["key"][0] % 5 == 0]
    train_count = min(len(train), maximum * 2 // 3)
    heldout_count = min(len(heldout), maximum - train_count)
    return select(train, train_count) + select(heldout, heldout_count)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p10_ms": float(np.percentile(values, 10)),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if os.environ.get("SGL_ENABLE_JIT_DEEPGEMM") != "0":
        raise RuntimeError("BF16 replay requires SGL_ENABLE_JIT_DEEPGEMM=0")
    if int(os.environ.get("WORLD_SIZE", "-1")) != 2:
        raise RuntimeError("true EP2 replay requires world_size=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()

    from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
    from sglang.srt.layers.moe.utils import DeepEPMode

    trace = TraceBundle.load(args.trace)
    cases = _cases(trace, args.maximum_cases)
    dispatcher = DeepEPDispatcher(
        group=dist.group.WORLD,
        router_topk=TOP_K,
        permute_fusion=False,
        num_experts=NUM_EXPERTS,
        num_local_experts=LOCAL_EXPERTS,
        hidden_size=HIDDEN,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.NORMAL,
        async_finish=True,
        return_recv_hook=True,
    )
    generator = torch.Generator(device=device).manual_seed(20260916 + rank)
    gate_up_weights = torch.randn(
        LOCAL_EXPERTS, HIDDEN, 2 * INTERMEDIATE,
        dtype=torch.bfloat16, device=device, generator=generator,
    ) * 0.01
    down_weights = torch.randn(
        LOCAL_EXPERTS, INTERMEDIATE, HIDDEN,
        dtype=torch.bfloat16, device=device, generator=generator,
    ) * 0.01
    records = []
    partition = SourcePartition(2)
    for case_index, case in enumerate(cases):
        sources = partition.ranks(case["source_keys"])
        selected = sources == rank
        local_ids = torch.as_tensor(case["expert_ids"][selected], device=device).long()
        rows = int(selected.sum())
        hidden = torch.randn(
            rows, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator
        ) * 0.02
        route_weights = torch.full(
            (rows, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device
        )
        dispatch_samples = []
        expert_samples = []
        combine_samples = []
        recv_rows = local_assignments = active_experts = 0
        for repeat in range(args.warmup + args.repeats):
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
            recv_hidden, recv_ids, recv_weights, _counts = dispatcher.dispatch(
                hidden_states=hidden,
                input_global_scale=None,
                topk_idx=local_ids,
                topk_weights=route_weights,
                forward_batch=_ForwardBatch(rows),
            )
            events[1].record()
            row_index, slot_index = torch.where(recv_ids >= 0)
            expert_index = recv_ids[row_index, slot_index].long()
            order = torch.argsort(expert_index, stable=True)
            ordered_experts = expert_index[order]
            ordered_rows = row_index[order]
            counts = torch.bincount(ordered_experts, minlength=LOCAL_EXPERTS).to(torch.int32)
            branch_output = _grouped_mlp(
                recv_hidden[ordered_rows], counts, gate_up_weights, down_weights
            )
            branches = torch.zeros(
                recv_hidden.shape[0], TOP_K, HIDDEN,
                dtype=torch.bfloat16, device=device,
            )
            branches[row_index[order], slot_index[order]] = branch_output
            recv_output = (
                branches.float().mul_(recv_weights.unsqueeze(-1)).sum(dim=1).bfloat16()
            )
            events[2].record()
            normal = dispatcher._normal_dispatcher
            _combined, combine_event = normal._combine_core(
                recv_output, previous_event=None
            )
            combine_event.current_stream_wait()
            normal.handle = None
            normal.src2dst = None
            dispatcher._stage = dispatcher._stage.__class__.INITIAL
            events[3].record()
            events[3].synchronize()
            recv_rows = int(recv_hidden.shape[0])
            local_assignments = int(row_index.numel())
            active_experts = int((counts > 0).sum())
            if repeat >= args.warmup:
                dispatch_samples.append(events[0].elapsed_time(events[1]))
                expert_samples.append(events[1].elapsed_time(events[2]))
                combine_samples.append(events[2].elapsed_time(events[3]))
        request_id, block_id, iteration_id, nfe, layer_id = case["key"]
        records.append(
            {
                "case_index": case_index,
                "rank": rank,
                "request_id": request_id,
                "block_id": block_id,
                "iteration_id": iteration_id,
                "nfe": nfe,
                "layer_id": layer_id,
                "source_rows": rows,
                "recv_rows": recv_rows,
                "local_expert_assignments": local_assignments,
                "local_active_experts": active_experts,
                "dispatch_samples_ms": dispatch_samples,
                "routed_expert_compute_samples_ms": expert_samples,
                "combine_samples_ms": combine_samples,
                "split": "heldout" if request_id % 5 == 0 else "train",
                "replicated_state_bridge_allgather": "not_present",
            }
        )
        del hidden, local_ids, route_weights
    gathered = [None, None]
    dist.all_gather_object(gathered, records)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        for source_rank, rank_records in enumerate(gathered):
            (args.output / f"rank{source_rank}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rank_records)
            )
        aggregate = []
        for case_index in range(len(cases)):
            endpoints = [records[case_index] for records in gathered]
            dispatch = [
                max(endpoint["dispatch_samples_ms"][repeat] for endpoint in endpoints)
                for repeat in range(args.repeats)
            ]
            expert = [
                max(
                    endpoint["routed_expert_compute_samples_ms"][repeat]
                    for endpoint in endpoints
                )
                for repeat in range(args.repeats)
            ]
            combine = [
                max(endpoint["combine_samples_ms"][repeat] for endpoint in endpoints)
                for repeat in range(args.repeats)
            ]
            first = endpoints[0]
            aggregate.append(
                {
                    "case_index": case_index,
                    "request_id": first["request_id"],
                    "block_id": first["block_id"],
                    "iteration_id": first["iteration_id"],
                    "nfe": first["nfe"],
                    "layer_id": first["layer_id"],
                    "split": first["split"],
                    "EP_dispatch": _summary(dispatch),
                    "routed_expert_compute": _summary(expert),
                    "EP_combine": _summary(combine),
                    "EP_stage_excluding_bridge": _summary(
                        [dispatch[i] + expert[i] + combine[i] for i in range(args.repeats)]
                    ),
                    "replicated_state_bridge_allgather": "not_present",
                }
            )
        (args.output / "aggregate.json").write_text(
            json.dumps(
                {
                    "label": "TRUE-EP2 trace replay",
                    "topology": "DP1_TP1_SP1_EP2",
                    "backend": "DeepEP normal + torch._grouped_mm BF16",
                    "cost_categories": [
                        "EP dispatch", "routed expert compute", "EP combine"
                    ],
                    "explicitly_excluded": [
                        "replicated-state bridge all-gather",
                        "production serving latency",
                    ],
                    "cases": aggregate,
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
