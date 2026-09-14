#!/usr/bin/env python3
"""Measure DeepEP Normal and low-latency on synthetic EP4 route cases.

The LL allocation capacity and per-call capacity contract are intentionally
separate.  DeepEP's capacity is per source rank, not global M.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "4,5,6,7"
WORLD = 4


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def load_cases(path: str):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def make_inputs(case, rank):
    ids = case["source_topk_ids"][rank]
    topk_idx = torch.tensor(ids, dtype=torch.int64, device=rank)
    if not ids:
        topk_idx = torch.empty((0, case["topk"]), dtype=torch.int64, device=rank)
    gen = torch.Generator(device=f"cuda:{rank}")
    gen.manual_seed(31013 + rank * 101 + case["global_m"] * 7)
    hidden = torch.randn((len(ids), case["hidden"]), dtype=torch.bfloat16,
                         device=rank, generator=gen)
    weights = torch.full((len(ids), case["topk"]), 1.0 / case["topk"],
                         dtype=torch.float32, device=rank)
    return hidden, topk_idx, weights


def events(n=5):
    return [torch.cuda.Event(enable_timing=True) for _ in range(n)]


def normal_call(buffer, hidden, topk_idx, weights, experts, dc, cc):
    ev = events()
    ev[0].record()
    layout = buffer.get_dispatch_layout(topk_idx, experts)
    ev[1].record()
    recv_x, recv_ids, recv_weights, _, handle, _ = buffer.dispatch(
        hidden, topk_idx=topk_idx, topk_weights=weights,
        num_tokens_per_rank=layout[0], num_tokens_per_rdma_rank=layout[1],
        is_token_in_rank=layout[3], num_tokens_per_expert=layout[2],
        async_finish=False, expert_alignment=1, config=dc)
    ev[2].record()
    local_weight = torch.where(recv_ids >= 0, recv_weights, 0).sum(dim=1)
    recv_output = recv_x * local_weight[:, None].to(recv_x.dtype)
    ev[3].record()
    combined, _, _ = buffer.combine(recv_output, handle, async_finish=False, config=cc)
    ev[4].record()
    return combined, ev, int((recv_ids >= 0).sum().item())


def ll_call(buffer, hidden, topk_idx, weights, contract, experts):
    ev = events()
    ev[0].record(); ev[1].record()
    recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
        hidden, topk_idx, contract, experts, use_fp8=False,
        async_finish=False, return_recv_hook=False)
    ev[2].record(); ev[3].record()
    out = torch.empty_like(hidden)
    combined, _, _ = buffer.low_latency_combine(
        recv_x, topk_idx, weights, handle, async_finish=False,
        return_recv_hook=False, out=out)
    ev[4].record()
    return combined, ev, int(recv_count.sum().item())


def ms(ev, a, b):
    return ev[a].elapsed_time(ev[b])


def worker(rank, port, args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {EXPECTED_VISIBLE}")
    torch.cuda.set_device(rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD))
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    from deep_ep import Buffer

    cases = load_cases(args.cases)
    if args.min_global_m:
        cases = [c for c in cases if c["global_m"] >= args.min_global_m]
    if args.max_global_m:
        cases = [c for c in cases if c["global_m"] <= args.max_global_m]
    hidden_size, experts = cases[0]["hidden"], cases[0]["experts"]
    dc, cc = Buffer.get_dispatch_config(WORLD), Buffer.get_combine_config(WORLD)
    hidden_bytes = hidden_size * 2
    nvl_bytes = max(dc.get_nvl_buffer_size_hint(hidden_bytes, WORLD),
                    cc.get_nvl_buffer_size_hint(hidden_bytes, WORLD))
    normal = Buffer(dist.group.WORLD, nvl_bytes, 0, low_latency_mode=False,
                    num_qps_per_rank=Buffer.num_sms, allow_mnnvl=True)

    allocation_capacity = args.allocation_capacity
    if not allocation_capacity:
        allocation_capacity = max(max(len(c["source_topk_ids"][r]) for r in range(WORLD))
                                  for c in cases)
    ll_bytes = Buffer.get_low_latency_rdma_size_hint(
        allocation_capacity, hidden_size, WORLD, experts)
    dist.barrier()
    t0 = time.perf_counter()
    ll = Buffer(dist.group.WORLD, num_rdma_bytes=ll_bytes, low_latency_mode=True,
                num_qps_per_rank=experts // WORLD,
                allow_nvlink_for_low_latency_mode=True, explicitly_destroy=True,
                allow_mnnvl=True)
    dist.barrier()
    init_s = time.perf_counter() - t0

    capacities = ([int(x) for x in args.capacities.split(",") if x]
                  if args.capacities else [])
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"rank{rank}.jsonl").open("w") as stream:
        for ci, case in enumerate(cases):
            hidden, topk_idx, weights = make_inputs(case, rank)
            local_m = hidden.shape[0]
            max_local = torch.tensor(local_m, dtype=torch.int64, device=rank)
            dist.all_reduce(max_local, op=dist.ReduceOp.MAX)
            max_local = int(max_local.item())
            contracts = capacities or [max(1, max_local)]
            contracts = [c for c in contracts if c >= max_local and c <= allocation_capacity]
            for contract in contracts:
                dist.barrier()
                ref, _, _ = normal_call(normal, hidden, topk_idx, weights, experts, dc, cc)
                torch.cuda.synchronize(rank)
                policies = ["normal", "low_latency"]
                for repeat in range(args.warmup + args.repeats):
                    order = policies[:]
                    random.Random(args.seed + ci * 1009 + contract * 17 + repeat).shuffle(order)
                    for policy in order:
                        dist.barrier()
                        if policy == "normal":
                            combined, ev, received = normal_call(
                                normal, hidden, topk_idx, weights, experts, dc, cc)
                        else:
                            combined, ev, received = ll_call(
                                ll, hidden, topk_idx, weights, contract, experts)
                        torch.cuda.synchronize(rank)
                        if repeat < args.warmup:
                            continue
                        denom = max(float(ref.float().norm().item()), 1e-12)
                        rel = float((combined.float() - ref.float()).norm().item()) / denom
                        rec = {
                            "rank": rank,
                            "physical_gpu": int(EXPECTED_VISIBLE.split(",")[rank]),
                            "uuid": str(torch.cuda.get_device_properties(rank).uuid),
                            "case_id": case["case_id"], "routing": case["routing"],
                            "global_m": case["global_m"], "local_m": int(local_m),
                            "max_local_m": max_local, "policy": policy,
                            "allocation_capacity_per_rank": allocation_capacity,
                            "contract_capacity_per_rank": contract,
                            "capacity_to_actual_ratio": contract / max(1, max_local),
                            "ll_heap_bytes_per_rank": ll_bytes,
                            "ll_double_recv_bytes_per_rank": 2 * (experts // WORLD) * WORLD * contract * hidden_size * 2,
                            "ll_init_s": init_s, "repeat": repeat - args.warmup,
                            "layout_ms": ms(ev, 0, 1), "dispatch_ms": ms(ev, 1, 2),
                            "expert_proxy_ms": ms(ev, 2, 3), "combine_ms": ms(ev, 3, 4),
                            "transaction_ms": ms(ev, 0, 4),
                            "relative_l2_vs_normal": rel,
                            "source_assignments": int((topk_idx >= 0).sum().item()),
                            "received_assignments": received,
                            "rank_load_cv": case["rank_load_cv"],
                            "mean_fanout": case["mean_fanout"],
                            "active_experts": case["active_experts"],
                        }
                        stream.write(json.dumps(rec) + "\n"); stream.flush()
    dist.barrier(); ll.destroy(); dist.barrier(); dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-global-m", type=int, default=0)
    parser.add_argument("--min-global-m", type=int, default=0)
    parser.add_argument("--allocation-capacity", type=int, default=0)
    parser.add_argument("--capacities", default="")
    args = parser.parse_args()
    mp.spawn(worker, args=(free_port(), args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
