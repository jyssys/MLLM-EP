#!/usr/bin/env python3
"""Bounded EP4 cost-surface calibration on physical GPUs 0--3.

This is a reference NCCL all-to-all transport with the production fused expert
kernel and the exact LLaDA-MoE expert dimensions (H=2048, I=1024).  It is not
presented as DeepEP throughput.  Its purpose is to identify whether bytes,
fanout, critical-rank work, and expert fragmentation explain physical stage
latency on the execution substrate used by the characterization runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


VISIBLE = "0,1,2,3"
UUIDS = (
    "f217c8a0-1142-20f4-d84b-af29f3a47a0d",
    "a77f3471-67d4-20b0-9fab-e502d4de5adb",
    "24200107-8a7f-de46-1bc8-b81f8d3af13e",
    "17488c15-2d4c-5d9e-d503-29b0d959a8a8",
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def allocation(total: int, fanout: int, remote_fraction: float) -> list[int]:
    if fanout == 1:
        return [total, 0, 0, 0]
    remote = min(total - (fanout - 1), max(fanout - 1, round(total * remote_fraction)))
    local = total - remote
    base, remainder = divmod(remote, fanout - 1)
    return [local] + [base + (i < remainder) for i in range(fanout - 1)] + [0] * (4 - fanout)


def worker(rank: int, port: int, args) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0,1,2,3")
    torch.cuda.set_device(rank)
    if str(torch.cuda.get_device_properties(rank).uuid) != UUIDS[rank]:
        raise RuntimeError(f"physical UUID mismatch at logical rank {rank}")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4")
    dist.init_process_group("nccl", rank=rank, world_size=4)
    from vllm.model_executor.layers.fused_moe import fused_experts

    torch.manual_seed(1200 + rank)
    device = torch.device(f"cuda:{rank}")
    # 16 resident experts per rank, matching LLaDA-MoE EP4.
    w1 = torch.empty((16, 2 * args.intermediate, args.hidden), device=device,
                     dtype=torch.bfloat16).normal_(0, .01)
    w2 = torch.empty((16, args.hidden, args.intermediate), device=device,
                     dtype=torch.bfloat16).normal_(0, .01)

    configs = []
    for assignments in args.assignments:
        configs.append((assignments, 1, 0.0, 8))
        for fanout in (2, 3, 4):
            for remote_fraction in (.25, .50, .75):
                for active_experts in (2, 8, 16):
                    configs.append((assignments, fanout, remote_fraction, active_experts))
    random.Random(20260912).shuffle(configs)

    records = []
    for config_id, (total, fanout, remote_fraction, active_experts) in enumerate(configs):
        counts = allocation(total, fanout, remote_fraction)
        recv_rows = counts[rank]
        if rank == 0:
            hidden_chunks, expert_chunks, weight_chunks = [], [], []
            for owner, count in enumerate(counts):
                hidden_chunks.append(torch.randn((count, args.hidden), device=device,
                                                 dtype=torch.bfloat16) * .02)
                local_ids = torch.arange(count, device=device, dtype=torch.long) % active_experts
                expert_chunks.append(local_ids + owner * 16)
                weight_chunks.append(torch.ones(count, device=device, dtype=torch.bfloat16))
            send_hidden = torch.cat(hidden_chunks)
            send_expert = torch.cat(expert_chunks)
            send_weight = torch.cat(weight_chunks)
        else:
            send_hidden = torch.empty((0, args.hidden), device=device, dtype=torch.bfloat16)
            send_expert = torch.empty(0, device=device, dtype=torch.long)
            send_weight = torch.empty(0, device=device, dtype=torch.bfloat16)

        def once(measure: bool):
            recv_hidden = torch.empty((recv_rows, args.hidden), device=device, dtype=torch.bfloat16)
            recv_expert = torch.empty(recv_rows, device=device, dtype=torch.long)
            recv_weight = torch.empty(recv_rows, device=device, dtype=torch.bfloat16)
            events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for name in ("dispatch", "expert", "combine", "total")}
            events["total"][0].record(); events["dispatch"][0].record()
            input_splits = counts if rank == 0 else [0, 0, 0, 0]
            output_splits = [recv_rows, 0, 0, 0]
            for output, source in ((recv_hidden, send_hidden), (recv_expert, send_expert),
                                   (recv_weight, send_weight)):
                dist.all_to_all_single(output, source, output_split_sizes=output_splits,
                                       input_split_sizes=input_splits)
            events["dispatch"][1].record(); events["expert"][0].record()
            if recv_rows:
                local_output = fused_experts(
                    hidden_states=recv_hidden, w1=w1, w2=w2,
                    topk_weights=recv_weight.unsqueeze(-1),
                    topk_ids=(recv_expert - rank * 16).unsqueeze(-1),
                    inplace=False, activation="silu", is_act_and_mul=True)
            else:
                local_output = torch.empty_like(recv_hidden)
            events["expert"][1].record(); events["combine"][0].record()
            returned = torch.empty((total if rank == 0 else 0, args.hidden),
                                   device=device, dtype=torch.bfloat16)
            dist.all_to_all_single(returned, local_output,
                                   output_split_sizes=counts if rank == 0 else [0, 0, 0, 0],
                                   input_split_sizes=[recv_rows, 0, 0, 0])
            events["combine"][1].record(); events["total"][1].record()
            torch.cuda.synchronize(rank)
            if returned.numel() and not torch.isfinite(returned).all():
                raise RuntimeError("non-finite expert output")
            if measure:
                return {name: start.elapsed_time(end) for name, (start, end) in events.items()}
            return None

        for _ in range(args.warmup):
            once(False)
        for repetition in range(args.repetitions):
            values = once(True)
            records.append({
                "config_id": config_id, "repetition": repetition, "rank": rank,
                "assignments": total, "fanout": fanout,
                "remote_fraction_target": remote_fraction,
                "remote_fraction_actual": sum(counts[1:]) / total,
                "active_experts_per_active_rank": active_experts,
                "rank0_count": counts[0], "rank1_count": counts[1],
                "rank2_count": counts[2], "rank3_count": counts[3],
                "critical_rank_assignments": max(counts),
                "remote_assignments": sum(counts[1:]),
                "remote_dispatch_bytes": sum(counts[1:]) * (args.hidden * 2 + 8 + 2),
                "combine_bytes": total * args.hidden * 2,
                **{f"{key}_ms": float(value) for key, value in values.items()},
            })

    gathered = [None] * 4 if rank == 0 else None
    dist.gather_object(records, gathered, dst=0)
    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        flat = [row for rank_rows in gathered for row in rank_rows]
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(flat[0]), lineterminator="\n"
            )
            writer.writeheader(); writer.writerows(flat)
        audit = {
            "cuda_visible_devices": VISIBLE,
            "physical_uuids": UUIDS,
            "world_size": 4,
            "backend": "NCCL all_to_all_single + vLLM fused_experts",
            "evidence_boundary": "reference transport cost surface, not DeepEP production throughput",
            "shape": {"hidden": args.hidden, "intermediate": args.intermediate,
                      "local_experts": 16, "dtype": "bfloat16"},
            "warmup": args.warmup, "repetitions": args.repetitions,
            "num_configs": len(configs),
        }
        output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    dist.barrier(); dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--assignments", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE or torch.cuda.device_count() != 4:
        raise SystemExit("requires exactly CUDA_VISIBLE_DEVICES=0,1,2,3")
    mp.spawn(worker, args=(free_port(), args), nprocs=4, join=True)


if __name__ == "__main__":
    main()
