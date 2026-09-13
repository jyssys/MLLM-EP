#!/usr/bin/env python3
"""Audit SGLang PP2xEP2/PP4 rank groups without loading model weights."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "4,5,6,7"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank: int, port: int, topology: str, output: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError("GPU visibility must be physical 4,5,6,7")
    torch.cuda.set_device(rank)
    from sglang.srt import distributed

    distributed.init_distributed_environment(4, rank, f"tcp://127.0.0.1:{port}", rank, "nccl")
    if topology == "pp2_ep2":
        tp_size, ep_size, pp_size = 2, 2, 2
    elif topology == "pp4":
        tp_size, ep_size, pp_size = 1, 1, 4
    else:
        raise ValueError(topology)
    distributed.initialize_model_parallel(tp_size, ep_size, pp_size, backend="nccl")
    tp = distributed.get_tp_group()
    pp = distributed.get_pp_group()
    ep = distributed.get_moe_ep_group()
    row = {
        "topology": topology,
        "global_rank": rank,
        "physical_gpu": rank + 4,
        "tp_size": tp_size,
        "ep_size": ep_size,
        "pp_size": pp_size,
        "tp_ranks": list(tp.ranks),
        "pp_ranks": list(pp.ranks),
        "ep_ranks": list(ep.ranks),
        "tp_rank": tp.rank_in_group,
        "pp_rank": pp.rank_in_group,
        "ep_rank": ep.rank_in_group,
    }
    gathered = [None] * 4
    dist.all_gather_object(gathered, row)
    if rank == 0:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(gathered, indent=2) + "\n")
    distributed.destroy_model_parallel()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("pp2_ep2", "pp4"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mp.spawn(worker, args=(free_port(), args.topology, args.output), nprocs=4, join=True)


if __name__ == "__main__":
    main()
