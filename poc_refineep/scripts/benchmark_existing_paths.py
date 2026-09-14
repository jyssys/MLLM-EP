#!/usr/bin/env python3
"""Replay real post-compaction LLaDA2 EP4 routes on existing DeepEP paths.

This benchmark measures communication semantics only.  Routed expert GEMM is
not replaced by an identity operation in the final oracle; the oracle joins
these communication measurements with separately measured fused-expert timing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "4,5,6,7"
WORLD = 4


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def load_cases(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def make_inputs(case, rank):
    ids = case["source_topk_ids"][rank]
    topk_idx = torch.tensor(ids, dtype=torch.int64, device=rank)
    if topk_idx.numel() == 0:
        topk_idx = torch.empty((0, case["topk"]), dtype=torch.int64, device=rank)
    generator = torch.Generator(device=f"cuda:{rank}")
    generator.manual_seed(17000 + rank * 97 + case["wave"] * 31 + case["layer"])
    hidden = torch.randn(
        (len(ids), case["hidden"]), dtype=torch.bfloat16, device=rank, generator=generator
    )
    weights = torch.full(
        (len(ids), case["topk"]), 1.0 / case["topk"], dtype=torch.float32, device=rank
    )
    return hidden, topk_idx, weights


def event_triplet():
    return [torch.cuda.Event(enable_timing=True) for _ in range(5)]


def normal_full(buffer, hidden, topk_idx, weights, experts, dispatch_config, combine_config):
    ev = event_triplet()
    ev[0].record()
    layout = buffer.get_dispatch_layout(topk_idx, experts)
    ev[1].record()
    recv_x, recv_ids, recv_weights, _, handle, _ = buffer.dispatch(
        hidden,
        topk_idx=topk_idx,
        topk_weights=weights,
        num_tokens_per_rank=layout[0],
        num_tokens_per_rdma_rank=layout[1],
        is_token_in_rank=layout[3],
        num_tokens_per_expert=layout[2],
        async_finish=False,
        expert_alignment=1,
        config=dispatch_config,
    )
    ev[2].record()
    local_weight = torch.where(recv_ids >= 0, recv_weights, 0).sum(dim=1)
    recv_output = recv_x * local_weight[:, None].to(recv_x.dtype)
    ev[3].record()
    combined, _, _ = buffer.combine(recv_output, handle, async_finish=False, config=combine_config)
    ev[4].record()
    info = {"received_assignments": int((recv_ids >= 0).sum().item())}
    return combined, ev, handle, (recv_x, recv_ids, recv_weights), info


def normal_cached(buffer, hidden, handle, recv_template, dispatch_config, combine_config):
    recv_ids, recv_weights = recv_template[1], recv_template[2]
    ev = event_triplet()
    ev[0].record(); ev[1].record()
    recv_x, _, _, _, _, _ = buffer.dispatch(
        hidden, handle=handle, async_finish=False, expert_alignment=1, config=dispatch_config
    )
    ev[2].record()
    local_weight = torch.where(recv_ids >= 0, recv_weights, 0).sum(dim=1)
    recv_output = recv_x * local_weight[:, None].to(recv_x.dtype)
    ev[3].record()
    combined, _, _ = buffer.combine(recv_output, handle, async_finish=False, config=combine_config)
    ev[4].record()
    return combined, ev, {"received_assignments": int((recv_ids >= 0).sum().item())}


def low_latency(ll_buffer, hidden, topk_idx, weights, max_tokens, experts):
    ev = event_triplet()
    ev[0].record(); ev[1].record()
    recv_x, recv_count, handle, _, _ = ll_buffer.low_latency_dispatch(
        hidden,
        topk_idx,
        max_tokens,
        experts,
        use_fp8=False,
        async_finish=False,
        return_recv_hook=False,
    )
    ev[2].record(); ev[3].record()
    out = torch.empty_like(hidden)
    combined, _, _ = ll_buffer.low_latency_combine(
        recv_x,
        topk_idx,
        weights,
        handle,
        async_finish=False,
        return_recv_hook=False,
        out=out,
    )
    ev[4].record()
    return combined, ev, {"received_assignments": int(recv_count.sum().item())}


def timing(ev):
    return {
        "layout_ms": ev[0].elapsed_time(ev[1]),
        "dispatch_ms": ev[1].elapsed_time(ev[2]),
        "weight_prep_ms": ev[2].elapsed_time(ev[3]),
        "combine_ms": ev[3].elapsed_time(ev[4]),
        "comm_semantics_ms": ev[0].elapsed_time(ev[4]),
    }


def worker(rank, port, args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {EXPECTED_VISIBLE}")
    torch.cuda.set_device(rank)
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
        LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD)
    )
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    from deep_ep import Buffer

    cases = load_cases(args.replay)
    if args.case_index >= 0:
        cases = [cases[args.case_index]]
    elif args.start_case:
        cases = cases[args.start_case :]
    if args.max_cases:
        cases = cases[: args.max_cases]
    max_tokens = max(len(case["source_topk_ids"][rank]) for case in cases)
    # The API requires one shared maximum across ranks.
    maximum = torch.tensor(max_tokens, dtype=torch.int64, device=rank)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    max_tokens = int(maximum.item())
    hidden_size, experts = cases[0]["hidden"], cases[0]["experts"]

    dispatch_config = Buffer.get_dispatch_config(WORLD)
    combine_config = Buffer.get_combine_config(WORLD)
    hidden_bytes = hidden_size * 2
    nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, WORLD),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, WORLD),
    )
    normal_buffer = Buffer(
        dist.group.WORLD, nvl_bytes, 0, low_latency_mode=False,
        num_qps_per_rank=Buffer.num_sms, allow_mnnvl=True,
    )
    ll_bytes = Buffer.get_low_latency_rdma_size_hint(max_tokens, hidden_size, WORLD, experts)
    ll_buffer = Buffer(
        dist.group.WORLD,
        num_rdma_bytes=ll_bytes,
        low_latency_mode=True,
        num_qps_per_rank=experts // WORLD,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
        allow_mnnvl=True,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank{rank}.jsonl"
    with path.open("w") as stream:
        for case_index, case in enumerate(cases):
            hidden, topk_idx, weights = make_inputs(case, rank)
            dist.barrier()
            reference, _, cached_handle, recv_template, _ = normal_full(
                normal_buffer, hidden, topk_idx, weights, experts, dispatch_config, combine_config
            )
            torch.cuda.synchronize(rank)
            policies = ["normal_fresh", "normal_cached", "low_latency"]
            for repeat in range(args.warmup + args.repeats):
                order = policies[:]
                random.Random(args.seed + case_index * 1009 + repeat).shuffle(order)
                for policy in order:
                    dist.barrier()
                    if policy == "normal_fresh":
                        combined, ev, _, _, info = normal_full(
                            normal_buffer, hidden, topk_idx, weights, experts,
                            dispatch_config, combine_config
                        )
                    elif policy == "normal_cached":
                        combined, ev, info = normal_cached(
                            normal_buffer, hidden, cached_handle, recv_template,
                            dispatch_config, combine_config
                        )
                    else:
                        combined, ev, info = low_latency(
                            ll_buffer, hidden, topk_idx, weights, max_tokens, experts
                        )
                    torch.cuda.synchronize(rank)
                    if repeat >= args.warmup:
                        denom = max(float(reference.float().norm().item()), 1e-12)
                        error = float((combined.float() - reference.float()).norm().item()) / denom
                        identity_denom = max(float(hidden.float().norm().item()), 1e-12)
                        identity_error = float((combined.float() - hidden.float()).norm().item()) / identity_denom
                        record = {
                            "rank": rank,
                            "physical_gpu": int(EXPECTED_VISIBLE.split(",")[rank]),
                            "uuid": str(torch.cuda.get_device_properties(rank).uuid),
                            "case_id": case["case_id"],
                            "dataset": case["dataset"],
                            "wave": case["wave"],
                            "layer": case["layer"],
                            "phase": case["phase"],
                            "shape_class": case["shape_class"],
                            "global_fresh_m": case["fresh_m"],
                            "local_fresh_m": int(hidden.shape[0]),
                            "max_tokens_contract": max_tokens,
                            "policy": policy,
                            "repeat": repeat - args.warmup,
                            "relative_l2_vs_normal": error,
                            "relative_l2_vs_identity": identity_error,
                            "source_assignments": int((topk_idx >= 0).sum().item()),
                            "received_assignments": info["received_assignments"],
                            **timing(ev),
                        }
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
    dist.barrier()
    ll_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--case-index", type=int, default=-1)
    args = parser.parse_args()
    mp.spawn(worker, args=(free_port(), args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
