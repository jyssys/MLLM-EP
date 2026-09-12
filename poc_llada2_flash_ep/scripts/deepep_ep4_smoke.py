#!/usr/bin/env python3
"""Four-H100 DeepEP BF16 dispatch/combine ownership and correctness smoke."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "0,1,2,3"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank: int, world: int, port: int, args) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError("GPU visibility must be exactly 0,1,2,3")
    torch.cuda.set_device(rank)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from deep_ep import Buffer

    hidden_bytes = args.hidden * 2
    dispatch_config = Buffer.get_dispatch_config(world)
    combine_config = Buffer.get_combine_config(world)
    nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, world),
    )
    buffer = Buffer(
        dist.group.WORLD,
        nvl_bytes,
        0,
        low_latency_mode=False,
        num_qps_per_rank=Buffer.num_sms,
        allow_mnnvl=True,
    )

    generator = torch.Generator(device=f"cuda:{rank}")
    generator.manual_seed(1234 + rank)
    hidden = torch.randn(
        args.tokens, args.hidden, dtype=torch.bfloat16, device=rank, generator=generator
    )
    # Every source rank sends useful branches to all four owners.
    base = torch.arange(args.tokens, device=rank)[:, None]
    lane = torch.arange(args.topk, device=rank)[None, :]
    topk_idx = ((base * 13 + lane * 37 + rank * 11) % args.experts).to(torch.int64)
    weights = torch.rand(
        args.tokens, args.topk, device=rank, generator=generator, dtype=torch.float32
    )
    weights /= weights.sum(dim=1, keepdim=True)

    for repeat_index in range(args.warmup + args.repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        combine_end = torch.cuda.Event(enable_timing=True)
        start.record()
        layout = buffer.get_dispatch_layout(topk_idx, args.experts)
        recv_x, recv_ids, recv_weights, recv_counts, handle, event = buffer.dispatch(
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
        dispatch_end.record()
        local_weight = torch.where(recv_ids >= 0, recv_weights, 0).sum(dim=1)
        recv_output = recv_x * local_weight[:, None].to(recv_x.dtype)
        combined, _, event = buffer.combine(
            recv_output,
            handle,
            async_finish=False,
            config=combine_config,
        )
        combine_end.record()
        torch.cuda.synchronize(rank)
        if repeat_index >= args.warmup:
            error = (combined.float() - hidden.float()).norm() / hidden.float().norm()
            row = {
                "rank": rank,
                "physical_gpu": rank,
                "uuid": str(torch.cuda.get_device_properties(rank).uuid),
                "experts_owned": [
                    rank * (args.experts // world),
                    (rank + 1) * (args.experts // world) - 1,
                ],
                "local_experts": args.experts // world,
                "source_tokens": args.tokens,
                "recv_tokens": int(recv_x.shape[0]),
                "recv_expert_id_min": int(recv_ids[recv_ids >= 0].min().item()),
                "recv_expert_id_max": int(recv_ids.max().item()),
                "recv_expert_ids": sorted(
                    {int(value) for value in recv_ids[recv_ids >= 0].unique().tolist()}
                ),
                "nonlocal_destination_fraction": float(
                    (topk_idx // (args.experts // world) != rank).float().mean().item()
                ),
                "dispatch_ms": start.elapsed_time(dispatch_end),
                "combine_ms": dispatch_end.elapsed_time(combine_end),
                "relative_l2": float(error.item()),
            }
            path = Path(args.output)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"rank{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    mp.spawn(worker, args=(4, free_port(), args), nprocs=4, join=True)


if __name__ == "__main__":
    main()
