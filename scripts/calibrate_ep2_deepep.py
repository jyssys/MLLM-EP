#!/usr/bin/env python3
"""Calibrate true EP2 DeepEP dispatch/combine on physical GPUs 0 and 1.

This is an EP-only microbenchmark.  It contains no replicated-state bridge and
therefore cannot be reported as model-serving or EP-isolation-harness E2E
latency.  Logical assignment traffic and deduplicated activation traffic are
both retained; measured timings come from the actual DeepEP normal path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist


HIDDEN_SIZE = 2048
NUM_EXPERTS = 256
TOP_K = 8


class _ForwardBatch:
    def __init__(self, rows: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.forward_mode = ForwardMode.EXTEND
        self.is_extend_in_batch = True
        self.num_token_non_padded = rows


def _routes(rows: int, rank: int, pattern: str, device: torch.device) -> torch.Tensor:
    row = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
    slot = torch.arange(TOP_K, device=device, dtype=torch.int64).unsqueeze(0)
    if pattern == "local":
        begin = rank * (NUM_EXPERTS // 2)
        return begin + (row * TOP_K + slot) % (NUM_EXPERTS // 2)
    if pattern == "bidirectional_remote":
        begin = (1 - rank) * (NUM_EXPERTS // 2)
        return begin + (row * TOP_K + slot) % (NUM_EXPERTS // 2)
    if pattern == "rank0_to_rank1":
        begin = NUM_EXPERTS // 2
        return begin + (row * TOP_K + slot) % (NUM_EXPERTS // 2)
    if pattern == "rank1_to_rank0":
        begin = 0
        return begin + (row * TOP_K + slot) % (NUM_EXPERTS // 2)
    if pattern == "uniform":
        return (row * TOP_K + slot + rank * 37) % NUM_EXPERTS
    if pattern == "expert_hot":
        owner = (row + rank) % 2
        local_id = (slot + (row % 4)) % 16
        return owner * (NUM_EXPERTS // 2) + local_id
    raise ValueError(pattern)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "p10_ms": _percentile(values, 0.10),
        "median_ms": median(values),
        "p90_ms": _percentile(values, 0.90),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--rows-per-source-rank", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--patterns", nargs="+",
        default=[
            "local", "uniform", "expert_hot", "rank0_to_rank1",
            "rank1_to_rank0", "bidirectional_remote",
        ],
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if os.environ.get("SGL_ENABLE_JIT_DEEPGEMM") != "0":
        raise RuntimeError("BF16 DeepEP calibration requires SGL_ENABLE_JIT_DEEPGEMM=0")
    if int(os.environ.get("WORLD_SIZE", "-1")) != 2:
        raise RuntimeError("calibration requires true world_size=2")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()

    from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
    from sglang.srt.layers.moe.utils import DeepEPMode

    dispatcher = DeepEPDispatcher(
        group=dist.group.WORLD,
        router_topk=TOP_K,
        permute_fusion=False,
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS // 2,
        hidden_size=HIDDEN_SIZE,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.NORMAL,
        async_finish=True,
        return_recv_hook=True,
    )
    generator = torch.Generator(device=device).manual_seed(20260916 + rank)
    rank_records: list[dict] = []
    for rows in args.rows_per_source_rank:
        hidden = torch.randn(
            rows, HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=generator
        )
        weights = torch.full(
            (rows, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device
        )
        forward_batch = _ForwardBatch(rows)
        for pattern in args.patterns:
            ids = _routes(rows, rank, pattern, device)
            owners = torch.div(ids, NUM_EXPERTS // 2, rounding_mode="floor")
            assignment_by_destination = torch.bincount(
                owners.reshape(-1), minlength=2
            ).cpu().tolist()
            unique_by_destination = [
                int((owners == destination).any(dim=1).sum())
                for destination in range(2)
            ]
            dispatch_samples: list[float] = []
            combine_samples: list[float] = []
            for repeat in range(args.warmup + args.repeats):
                begin_dispatch = torch.cuda.Event(enable_timing=True)
                end_dispatch = torch.cuda.Event(enable_timing=True)
                end_combine = torch.cuda.Event(enable_timing=True)
                begin_dispatch.record()
                recv_hidden, _recv_ids, _recv_weights, _counts = dispatcher.dispatch(
                    hidden_states=hidden,
                    input_global_scale=None,
                    topk_idx=ids,
                    topk_weights=weights,
                    forward_batch=forward_batch,
                )
                end_dispatch.record()
                normal = dispatcher._normal_dispatcher
                combined, event = normal._combine_core(
                    torch.zeros_like(recv_hidden), previous_event=None
                )
                event.current_stream_wait()
                normal.handle = None
                normal.src2dst = None
                dispatcher._stage = dispatcher._stage.__class__.INITIAL
                end_combine.record()
                end_combine.synchronize()
                if repeat >= args.warmup:
                    dispatch_samples.append(begin_dispatch.elapsed_time(end_dispatch))
                    combine_samples.append(end_dispatch.elapsed_time(end_combine))
                del recv_hidden, combined
            rank_records.append(
                {
                    "rank": rank,
                    "rows_per_source_rank": rows,
                    "global_physical_rows": rows * 2,
                    "pattern": pattern,
                    "source_assignments_to_rank0": assignment_by_destination[0],
                    "source_assignments_to_rank1": assignment_by_destination[1],
                    "source_unique_activations_to_rank0": unique_by_destination[0],
                    "source_unique_activations_to_rank1": unique_by_destination[1],
                    "source_remote_assignments": assignment_by_destination[1 - rank],
                    "source_remote_unique_activations": unique_by_destination[1 - rank],
                    "logical_remote_activation_bytes": (
                        unique_by_destination[1 - rank] * HIDDEN_SIZE * 2
                    ),
                    "dispatch_samples_ms": dispatch_samples,
                    "combine_samples_ms": combine_samples,
                }
            )
    gathered = [None, None]
    dist.all_gather_object(gathered, rank_records)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        for source_rank, records in enumerate(gathered):
            (args.output / f"rank{source_rank}_samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )
        aggregate = []
        by_key = {
            (record["rows_per_source_rank"], record["pattern"]): []
            for records in gathered for record in records
        }
        for records in gathered:
            for record in records:
                by_key[(record["rows_per_source_rank"], record["pattern"])].append(record)
        for (rows, pattern), records in sorted(by_key.items()):
            # A collective completes at the slower endpoint. Pair corresponding
            # repeats rather than pooling rank samples as if independent.
            dispatch_critical = [
                max(record["dispatch_samples_ms"][repeat] for record in records)
                for repeat in range(args.repeats)
            ]
            combine_critical = [
                max(record["combine_samples_ms"][repeat] for record in records)
                for repeat in range(args.repeats)
            ]
            aggregate.append(
                {
                    "rows_per_source_rank": rows,
                    "global_physical_rows": rows * 2,
                    "pattern": pattern,
                    "remote_assignments": sum(
                        record["source_remote_assignments"] for record in records
                    ),
                    "remote_unique_activations": sum(
                        record["source_remote_unique_activations"] for record in records
                    ),
                    "logical_remote_activation_bytes": sum(
                        record["logical_remote_activation_bytes"] for record in records
                    ),
                    "max_endpoint_remote_activation_bytes": max(
                        record["logical_remote_activation_bytes"] for record in records
                    ),
                    "dispatch": _summarize(dispatch_critical),
                    "combine": _summarize(combine_critical),
                    "replicated_state_bridge_allgather": "not_present",
                }
            )
        (args.output / "aggregate.json").write_text(
            json.dumps(
                {
                    "label": "TRUE-EP2 DeepEP communication calibration",
                    "topology": "DP1_TP1_SP1_EP2",
                    "hidden_size": HIDDEN_SIZE,
                    "num_routed_experts": NUM_EXPERTS,
                    "top_k": TOP_K,
                    "dtype": "bfloat16",
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "cost_categories": ["EP dispatch", "EP combine"],
                    "explicitly_excluded": [
                        "routed expert compute",
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
