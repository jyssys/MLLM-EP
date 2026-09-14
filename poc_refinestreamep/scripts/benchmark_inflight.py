#!/usr/bin/env python3
"""Safe one-slot/two-slot DeepEP serving microbenchmark.

No low-latency receive buffer is retained across more than two dispatches.  Q>2
is therefore serviced in legal chunks of two; completion timestamps include the
resulting queue/serialization wait.
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


VISIBLE = "4,5,6,7"
WORLD = 4


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def load(path):
    with Path(path).open() as f: return [json.loads(x) for x in f]


def tensors(case, rank, serial):
    ids = case["source_topk_ids"][rank]
    idx = torch.tensor(ids, dtype=torch.int64, device=rank)
    if not ids: idx = torch.empty((0, case["topk"]), dtype=torch.int64, device=rank)
    gen = torch.Generator(device=f"cuda:{rank}"); gen.manual_seed(50021 + rank * 997 + serial)
    x = torch.randn((len(ids), case["hidden"]), dtype=torch.bfloat16, device=rank, generator=gen)
    w = torch.full((len(ids), case["topk"]), 1 / case["topk"], dtype=torch.float32, device=rank)
    return x, idx, w


def ll_one(buffer, job, capacity, experts):
    x, idx, w = job
    recv, _, handle, _, _ = buffer.low_latency_dispatch(
        x, idx, capacity, experts, use_fp8=False, async_finish=False, return_recv_hook=False)
    out = torch.empty_like(x)
    y, _, _ = buffer.low_latency_combine(
        recv, idx, w, handle, async_finish=False, return_recv_hook=False, out=out)
    return y


def ll_chunk2(buffer, jobs, capacity, experts):
    dispatched = []
    for x, idx, w in jobs:
        recv, _, handle, event, _ = buffer.low_latency_dispatch(
            x, idx, capacity, experts, use_fp8=False, async_finish=True, return_recv_hook=False)
        dispatched.append((x, idx, w, recv, handle, event))
    combined = []
    for x, idx, w, recv, handle, event in dispatched:
        event.current_stream_wait()
        out = torch.empty_like(x)
        y, event2, _ = buffer.low_latency_combine(
            recv, idx, w, handle, async_finish=True, return_recv_hook=False, out=out)
        combined.append((y, event2))
    for _, event in combined: event.current_stream_wait()
    return [x for x, _ in combined]


def normal_one(buffer, job, experts, dc, cc):
    x, idx, w = job
    layout = buffer.get_dispatch_layout(idx, experts)
    recv, recv_idx, recv_w, _, handle, _ = buffer.dispatch(
        x, topk_idx=idx, topk_weights=w, num_tokens_per_rank=layout[0],
        num_tokens_per_rdma_rank=layout[1], is_token_in_rank=layout[3],
        num_tokens_per_expert=layout[2], async_finish=False, expert_alignment=1, config=dc)
    local_w = torch.where(recv_idx >= 0, recv_w, 0).sum(1)
    y, _, _ = buffer.combine(recv * local_w[:, None].to(recv.dtype), handle,
                             async_finish=False, config=cc)
    return y


def execute(policy, buffer, normal, jobs, capacity, experts, dc, cc):
    done = []; completion = []
    t0 = time.perf_counter()
    if policy == "ll_serial":
        for job in jobs:
            done.append(ll_one(buffer, job, capacity, experts)); torch.cuda.synchronize()
            completion.append(time.perf_counter() - t0)
    elif policy == "ll_two_slot":
        for begin in range(0, len(jobs), 2):
            done.extend(ll_chunk2(buffer, jobs[begin:begin + 2], capacity, experts))
            torch.cuda.synchronize()
            now = time.perf_counter() - t0
            completion.extend([now] * len(jobs[begin:begin + 2]))
    elif policy == "normal_serial":
        for job in jobs:
            done.append(normal_one(normal, job, experts, dc, cc)); torch.cuda.synchronize()
            completion.append(time.perf_counter() - t0)
    else:
        raise ValueError(policy)
    return done, completion, time.perf_counter() - t0


def worker(rank, p, args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE: raise RuntimeError("bad visible devices")
    torch.cuda.set_device(rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(p), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD))
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    from deep_ep import Buffer
    cases = load(args.cases); experts, hidden = cases[0]["experts"], cases[0]["hidden"]
    dc, cc = Buffer.get_dispatch_config(WORLD), Buffer.get_combine_config(WORLD)
    hb = hidden * 2
    nvl = max(dc.get_nvl_buffer_size_hint(hb, WORLD), cc.get_nvl_buffer_size_hint(hb, WORLD))
    normal = Buffer(dist.group.WORLD, nvl, 0, low_latency_mode=False,
                    num_qps_per_rank=Buffer.num_sms, allow_mnnvl=True)
    cap = args.capacity
    llbytes = Buffer.get_low_latency_rdma_size_hint(cap, hidden, WORLD, experts)
    ll = Buffer(dist.group.WORLD, num_rdma_bytes=llbytes, low_latency_mode=True,
                num_qps_per_rank=experts // WORLD, allow_nvlink_for_low_latency_mode=True,
                explicitly_destroy=True, allow_mnnvl=True)
    is_real_stream = any("provenance" in c for c in cases)
    if is_real_stream:
        ordered = sorted(cases, key=lambda c: (c["dataset"], c["wave"]))
        small = min(ordered, key=lambda c: abs(c["global_m"] - args.homogeneous_m))
        large = min(ordered, key=lambda c: abs(c["global_m"] - args.large_m))
        # Fixed age offsets cover early, middle, and late real refinement routes.
        offsets = [0, 7, 19, 31, 43, 57, 73, 101]
        mixed = [ordered[i % len(ordered)] for i in offsets]
        patterns = {"homogeneous_small": [small], "homogeneous_large": [large],
                    "real_age_offset": mixed}
    else:
        by_m = {c["global_m"]: c for c in cases if c["routing"] == "real_like"}
        patterns = {
            "homogeneous_small": [by_m[args.homogeneous_m]],
            "homogeneous_large": [by_m[args.large_m]],
            "heterogeneous": [by_m[m] for m in [8, 512, 32, 1024, 16, 256, 64, 128]],
        }
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with (out / f"rank{rank}.jsonl").open("w") as f:
        for workload, seq in patterns.items():
            for q in [1, 2, 4, 8, 16]:
                selected = [seq[i % len(seq)] for i in range(q)]
                jobs = [tensors(c, rank, 1000 * q + i) for i, c in enumerate(selected)]
                for repeat in range(args.warmup + args.repeats):
                    policies = ["ll_serial", "ll_two_slot", "normal_serial"]
                    random.Random(args.seed + q * 101 + repeat).shuffle(policies)
                    for policy in policies:
                        dist.barrier(); torch.cuda.synchronize()
                        done, completion, wall = execute(
                            policy, ll, normal, jobs, cap, experts, dc, cc)
                        torch.cuda.synchronize(); dist.barrier()
                        if repeat < args.warmup: continue
                        errors = []
                        for y, (x, _, _) in zip(done, jobs):
                            denom = max(float(x.float().norm().item()), 1e-12)
                            errors.append(float((y.float() - x.float()).norm().item()) / denom)
                        rec = {"rank": rank, "physical_gpu": int(VISIBLE.split(",")[rank]),
                               "uuid": str(torch.cuda.get_device_properties(rank).uuid),
                               "workload": workload, "q": q, "policy": policy,
                               "repeat": repeat - args.warmup, "wall_ms": wall * 1000,
                               "waves_per_s": q / wall, "rows_per_s": sum(c["global_m"] for c in selected) / wall,
                               "completion_ms": [v * 1000 for v in completion],
                               "p50_completion_ms": float(torch.tensor(completion).median().item() * 1000),
                               "p95_completion_ms": sorted(completion)[max(0, int(0.95 * len(completion)) - 1)] * 1000,
                               "p99_completion_ms": sorted(completion)[max(0, int(0.99 * len(completion)) - 1)] * 1000,
                               "max_relative_l2": max(errors, default=0.0),
                               "capacity_per_rank": cap, "ll_heap_bytes_per_rank": llbytes}
                        f.write(json.dumps(rec) + "\n"); f.flush()
    dist.barrier(); ll.destroy(); dist.barrier(); dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--homogeneous-m", type=int, default=32); ap.add_argument("--large-m", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=4); ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260914); a = ap.parse_args()
    mp.spawn(worker, args=(port(), a), nprocs=WORLD, join=True)


if __name__ == "__main__": main()
