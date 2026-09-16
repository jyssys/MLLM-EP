#!/usr/bin/env python3
"""Selected-state true-EP2 replay for H1/H4 discovery validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 2048
INTERMEDIATE = 512
LOCAL_EXPERTS = 128


class _ForwardBatch:
    def __init__(self, rows):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        self.forward_mode = ForwardMode.EXTEND
        self.is_extend_in_batch = True
        self.num_token_non_padded = rows


def grouped_mlp(x, counts, gate_up, down):
    offsets = counts.cumsum(0, dtype=torch.int32)
    mixed = torch._grouped_mm(x, gate_up, offs=offsets)
    gate, up = mixed.chunk(2, dim=-1)
    return torch._grouped_mm(torch.nn.functional.silu(gate) * up, down, offs=offsets)


def summary(values):
    return {"p10_ms": float(np.percentile(values, 10)),
            "median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1" or int(os.environ.get("WORLD_SIZE", -1)) != 2:
        raise RuntimeError("selected replay requires exactly physical GPUs 0,1 / world size 2")
    if os.environ.get("SGL_ENABLE_JIT_DEEPGEMM") != "0":
        raise RuntimeError("BF16 grouped replay requires SGL_ENABLE_JIT_DEEPGEMM=0")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
    from sglang.srt.layers.moe.utils import DeepEPMode

    with np.load(args.cases, allow_pickle=False) as source:
        routes = source["expert_ids"]
        offsets = source["offsets"]
        labels = json.loads(str(source["labels_json"]))
    dispatcher = DeepEPDispatcher(
        group=dist.group.WORLD, router_topk=TOP_K, permute_fusion=False,
        num_experts=NUM_EXPERTS, num_local_experts=LOCAL_EXPERTS,
        hidden_size=HIDDEN, params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.NORMAL, async_finish=True, return_recv_hook=True,
    )
    generator = torch.Generator(device=device).manual_seed(20260917 + rank)
    gate_up = torch.randn(LOCAL_EXPERTS, HIDDEN, 2 * INTERMEDIATE,
                          dtype=torch.bfloat16, device=device, generator=generator) * .01
    down = torch.randn(LOCAL_EXPERTS, INTERMEDIATE, HIDDEN,
                       dtype=torch.bfloat16, device=device, generator=generator) * .01
    # Physical GPU service rates are not assumed identical.  Measure an
    # identical balanced local workload on both ranks, then use the ratio only
    # for the structural critical-rank check.  Reported wall times remain raw.
    calibration_rows = 2048
    calibration_x = torch.randn(
        calibration_rows, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator
    ) * .02
    calibration_counts = torch.full(
        (LOCAL_EXPERTS,), calibration_rows // LOCAL_EXPERTS,
        dtype=torch.int32, device=device,
    )
    calibration_samples = []
    for repeat in range(args.warmup + args.repeats):
        begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        begin.record(); calibration_output = grouped_mlp(
            calibration_x, calibration_counts, gate_up, down
        ); end.record(); end.synchronize()
        if repeat >= args.warmup:
            calibration_samples.append(begin.elapsed_time(end))
    calibration_median = float(np.median(calibration_samples))
    del calibration_x, calibration_counts, calibration_output
    records = []
    for case_id, label in enumerate(labels):
        case = routes[offsets[case_id]:offsets[case_id + 1]]
        sizes = np.full(2, len(case) // 2); sizes[:len(case) % 2] += 1
        sources = np.repeat(np.arange(2), sizes)
        local_routes = torch.as_tensor(case[sources == rank], device=device).long()
        rows = len(local_routes)
        hidden = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator) * .02
        weights = torch.full((rows, TOP_K), 1 / TOP_K, dtype=torch.float32, device=device)
        samples = {"dispatch": [], "expert": [], "combine": []}
        assignments = 0
        for repeat in range(args.warmup + args.repeats):
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
            recv, recv_ids, recv_weights, _ = dispatcher.dispatch(
                hidden_states=hidden, input_global_scale=None, topk_idx=local_routes,
                topk_weights=weights, forward_batch=_ForwardBatch(rows),
            )
            events[1].record()
            row_index, slot = torch.where(recv_ids >= 0)
            expert = recv_ids[row_index, slot].long()
            order = torch.argsort(expert, stable=True)
            counts = torch.bincount(expert[order], minlength=LOCAL_EXPERTS).to(torch.int32)
            branch = grouped_mlp(recv[row_index[order]], counts, gate_up, down)
            branches = torch.zeros(recv.shape[0], TOP_K, HIDDEN, dtype=torch.bfloat16, device=device)
            branches[row_index[order], slot[order]] = branch
            combined_local = branches.float().mul_(recv_weights.unsqueeze(-1)).sum(1).bfloat16()
            events[2].record()
            normal = dispatcher._normal_dispatcher
            _, event = normal._combine_core(combined_local, previous_event=None)
            event.current_stream_wait(); normal.handle = None; normal.src2dst = None
            dispatcher._stage = dispatcher._stage.__class__.INITIAL
            events[3].record(); events[3].synchronize()
            assignments = int(row_index.numel())
            if repeat >= args.warmup:
                samples["dispatch"].append(events[0].elapsed_time(events[1]))
                samples["expert"].append(events[1].elapsed_time(events[2]))
                samples["combine"].append(events[2].elapsed_time(events[3]))
        records.append({"case_id": case_id, "rank": rank, "label": label,
                        "source_rows": rows, "local_assignments": assignments,
                        "balanced_device_calibration_ms": calibration_median,
                        "EP_dispatch": summary(samples["dispatch"]),
                        "routed_expert_compute": summary(samples["expert"]),
                        "EP_combine": summary(samples["combine"]),
                        "replicated_state_bridge_allgather": "not_present"})
    gathered = [None, None]; dist.all_gather_object(gathered, records)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        combined = []
        for case_id, label in enumerate(labels):
            endpoints = [gathered[r][case_id] for r in range(2)]
            raw_times = [row["routed_expert_compute"]["median_ms"] for row in endpoints]
            normalized_times = [
                row["routed_expert_compute"]["median_ms"] /
                row["balanced_device_calibration_ms"] for row in endpoints
            ]
            raw_slow = int(np.argmax(raw_times))
            slow = int(np.argmax(normalized_times))
            combined.append({"case_id": case_id, "label": label, "ranks": endpoints,
                             "measured_slow_rank": slow,
                             "raw_physical_slow_rank": raw_slow,
                             "device_normalized_expert_time": normalized_times,
                             "structural_rank_match": slow == label["structural_critical_rank_ep2"],
                             "critical_expert_ms": max(row["routed_expert_compute"]["median_ms"] for row in endpoints),
                             "critical_dispatch_ms": max(row["EP_dispatch"]["median_ms"] for row in endpoints),
                             "critical_combine_ms": max(row["EP_combine"]["median_ms"] for row in endpoints)})
        (args.output / "summary.json").write_text(json.dumps({
            "label": "selected-state TRUE-EP2 replay", "topology": "DP1_TP1_SP1_EP2",
            "cost_categories": ["EP dispatch", "routed expert compute", "EP combine"],
            "excluded": ["replicated-state bridge all-gather", "production serving latency"],
            "structural_rank_match_rate": float(np.mean([row["structural_rank_match"] for row in combined])),
            "cases": combined,
        }, indent=2) + "\n")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
