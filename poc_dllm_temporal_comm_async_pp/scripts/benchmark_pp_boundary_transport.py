#!/usr/bin/env python3
"""Measure exact BF16 activation transport across allowed adjacent GPUs."""

from __future__ import annotations

import argparse
import csv
import os
import socket
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "4,5,6,7"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank: int, port: int, args) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError("physical visibility must be exactly 4,5,6,7")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    rows = []
    for count in map(int, args.rows.split(",")):
        tensor = torch.randn(count, args.hidden, device=rank, dtype=torch.bfloat16)
        for source, destination in ((0, 1), (1, 2), (2, 3)):
            samples = []
            for iteration in range(args.warmup + args.repeats):
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                if rank == source:
                    dist.send(tensor, dst=destination)
                elif rank == destination:
                    dist.recv(tensor, src=source)
                stop.record()
                stop.synchronize()
                if rank in (source, destination) and iteration >= args.warmup:
                    samples.append(start.elapsed_time(stop))
            gathered = [None] * 4
            dist.all_gather_object(gathered, samples)
            if rank == 0:
                critical = [max(a, b) for a, b in zip(gathered[source], gathered[destination])]
                rows.append(
                    {
                        "source": source,
                        "destination": destination,
                        "rows": count,
                        "hidden": args.hidden,
                        "bytes": count * args.hidden * 2,
                        "p50_ms": float(np.median(critical)),
                        "p90_ms": float(np.quantile(critical, 0.9)),
                        "p99_ms": float(np.quantile(critical, 0.99)),
                    }
                )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="32,64,128,256,512,1024")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    mp.spawn(worker, args=(free_port(), args), nprocs=4, join=True)


if __name__ == "__main__":
    main()
