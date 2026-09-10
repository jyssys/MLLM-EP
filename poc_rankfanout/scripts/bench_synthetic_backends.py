#!/usr/bin/env python3
"""Exact-route Qwen MoE operator benchmark for AGRS and DeepEP HT.

The four-rank benchmark loads one Qwen3-VL expert slice per rank.  AGRS uses
NCCL all-gather for hidden/top-k tensors, local TritonExperts, and NCCL
reduce-scatter.  DeepEP uses its stock HT layout/dispatch/combine calls plus
the same local weights and TritonExperts.  This is a bounded forward-only
operator diagnostic, not a claim that the standalone process reproduces all
TP2/DP2 scheduler behavior; clean full-model runs provide that evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_rankfanout.rankfanout.routing import (  # noqa: E402
    balanced_fanout_route,
    linear_expert_to_rank,
    summarize_route,
    validate_control_family,
)


def load_weight(model: Path, index: dict[str, str], key: str, lo=None, hi=None) -> torch.Tensor:
    with safe_open(str(model / index[key]), framework="pt", device="cpu") as handle:
        if lo is None:
            return handle.get_tensor(key).contiguous()
        return handle.get_slice(key)[lo:hi].contiguous()


def load_local_experts(model: Path, layer: int, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    expert = f"model.language_model.layers.{layer}.mlp.experts."
    packed_key = expert + "gate_up_proj"
    if packed_key not in index:
        expert = f"model.layers.{layer}.mlp.experts."
        packed_key = expert + "gate_up_proj"
    first = rank * 32
    if packed_key in index:
        w1 = load_weight(model, index, packed_key, first, first + 32).transpose(1, 2).contiguous().cuda()
        w2 = load_weight(model, index, expert + "down_proj", first, first + 32).transpose(1, 2).contiguous().cuda()
    else:
        gate_up, down = [], []
        for expert_id in range(first, first + 32):
            gate = load_weight(model, index, expert + f"{expert_id}.gate_proj.weight")
            up = load_weight(model, index, expert + f"{expert_id}.up_proj.weight")
            gate_up.append(torch.cat((gate, up), dim=0))
            down.append(load_weight(model, index, expert + f"{expert_id}.down_proj.weight"))
        w1 = torch.stack(gate_up).contiguous().cuda()
        w2 = torch.stack(down).contiguous().cuda()
    expert_map = torch.full((128,), -1, dtype=torch.int32, device="cuda")
    expert_map[first:first + 32] = torch.arange(32, dtype=torch.int32, device="cuda")
    return w1, w2, expert_map


def critical(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return [float(x) for x in tensor.cpu().tolist()]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("agrs", "deepep_ht"), required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--fanout", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--buffer-mib", type=int, default=1024)
    parser.add_argument("--precondition-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("GPU safety violation: expected CUDA_VISIBLE_DEVICES=4,5,6,7")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError("synthetic benchmark requires exactly four ranks")

    w1, w2, expert_map = load_local_experts(args.model, args.layer, rank)
    hidden_size = int(w1.shape[2])
    deep_buffer = None
    comm_stream = None
    if args.backend == "deepep_ht":
        import deep_ep

        deep_ep.Buffer.set_num_sms(20)
        deep_buffer = deep_ep.Buffer(
            dist.group.WORLD,
            args.buffer_mib * 1024 * 1024,
            0,
            low_latency_mode=False,
            num_qps_per_rank=1,
            explicitly_destroy=True,
        )
        comm_stream = deep_buffer.get_comm_stream()

    def sync() -> None:
        torch.cuda.synchronize()
        dist.barrier()

    def run_agrs(route: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        weights = torch.full(route.shape, 1.0 / 8, dtype=torch.float32, device=device)
        gathered_h = torch.empty((world * hidden.shape[0], hidden.shape[1]), dtype=hidden.dtype, device=device)
        gathered_i = torch.empty((world * route.shape[0], route.shape[1]), dtype=route.dtype, device=device)
        gathered_w = torch.empty((world * weights.shape[0], weights.shape[1]), dtype=weights.dtype, device=device)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        dist.all_gather_into_tensor(gathered_h, hidden)
        dist.all_gather_into_tensor(gathered_i, route)
        dist.all_gather_into_tensor(gathered_w, weights)
        events[1].record()
        local_out = fused_experts(
            gathered_h,
            w1,
            w2,
            gathered_w,
            gathered_i,
            global_num_experts=128,
            expert_map=expert_map,
        )
        events[2].record()
        output = torch.empty_like(hidden)
        dist.reduce_scatter_tensor(output, local_out.contiguous(), op=dist.ReduceOp.SUM)
        events[3].record()
        events[3].synchronize()
        return output, {
            "dispatch_ms": events[0].elapsed_time(events[1]),
            "expert_ms": events[1].elapsed_time(events[2]),
            "combine_ms": events[2].elapsed_time(events[3]),
            "moe_total_ms": events[0].elapsed_time(events[3]),
        }

    def run_deepep(route: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        assert deep_buffer is not None and comm_stream is not None
        import deep_ep

        weights = torch.full(route.shape, 1.0 / 8, dtype=torch.float32, device=device)
        ids = route.to(deep_ep.topk_idx_t).contiguous()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        events[0].record()
        num_rank, num_rdma, num_expert, in_rank, layout_event = deep_buffer.get_dispatch_layout(
            ids, 128, async_finish=True, allocate_on_comm_stream=False
        )
        recv_h, recv_ids, recv_w, _, handle, dispatch_event = deep_buffer.dispatch(
            x=hidden,
            handle=None,
            num_tokens_per_rank=num_rank,
            num_tokens_per_rdma_rank=num_rdma,
            is_token_in_rank=in_rank,
            num_tokens_per_expert=num_expert,
            topk_idx=ids,
            topk_weights=weights,
            expert_alignment=1,
            config=deep_ep.Buffer.get_dispatch_config(world),
            previous_event=layout_event,
            async_finish=True,
            allocate_on_comm_stream=False,
        )
        events[1].record(comm_stream)
        dispatch_event.current_stream_wait()
        events[2].record()
        first = rank * 32
        global_ids = torch.where(
            recv_ids == -1,
            127 if first == 0 else 0,
            recv_ids.to(torch.int64) + first,
        )
        local_out = fused_experts(
            recv_h,
            w1,
            w2,
            recv_w,
            global_ids,
            global_num_experts=128,
            expert_map=expert_map,
        )
        events[3].record()
        output, _, combine_event = deep_buffer.combine(
            x=local_out,
            handle=handle,
            topk_weights=None,
            config=deep_ep.Buffer.get_combine_config(world),
            async_finish=True,
            allocate_on_comm_stream=False,
        )
        events[4].record(comm_stream)
        combine_event.current_stream_wait()
        events[5].record()
        events[5].synchronize()
        return output, {
            "dispatch_ms": events[0].elapsed_time(events[2]),
            "expert_ms": events[2].elapsed_time(events[3]),
            "combine_ms": events[3].elapsed_time(events[5]),
            "moe_total_ms": events[0].elapsed_time(events[5]),
        }

    run = run_agrs if args.backend == "agrs" else run_deepep
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        args.output.write_text("")

    mapping = linear_expert_to_rank()
    rows: list[dict[str, object]] = []
    samples: dict[str, np.ndarray] = {}
    for num_tokens in args.tokens:
        torch.manual_seed(args.seed + num_tokens + rank * 100003)
        hidden = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
        numpy_routes = {
            fanout: balanced_fanout_route(num_tokens, fanout, seed=args.seed + num_tokens)
            for fanout in args.fanout
        }
        validate_control_family(numpy_routes.values())
        routes = {fanout: torch.from_numpy(route).to(device=device, dtype=torch.int64) for fanout, route in numpy_routes.items()}
        summaries = {fanout: summarize_route(route, mapping).to_dict() for fanout, route in numpy_routes.items()}

        warm_schedule = [fanout for _ in range(args.warmup) for fanout in args.fanout]
        random.Random(args.seed + num_tokens * 13).shuffle(warm_schedule)
        for fanout in warm_schedule:
            run(routes[fanout], hidden)
        # Never use an independently timed loop around a collective: ranks can
        # execute different iteration counts and deadlock.  Raise clocks with
        # rank-local GEMMs, then rejoin at a barrier before communication.
        if args.precondition_seconds > 0:
            precondition = torch.randn((4096, 4096), dtype=torch.bfloat16, device=device)
            deadline = time.perf_counter() + args.precondition_seconds
            while time.perf_counter() < deadline:
                torch.mm(precondition, precondition)
            del precondition
            torch.cuda.synchronize()
            dist.barrier()
        for fanout in args.fanout:
            run(routes[fanout], hidden)
        sync()

        schedule = [(fanout, rep) for rep in range(args.reps) for fanout in args.fanout]
        random.Random(args.seed + num_tokens * 17).shuffle(schedule)
        sampled: set[int] = set()
        for order, (fanout, rep) in enumerate(schedule):
            sync()
            start = time.perf_counter()
            output, timing = run(routes[fanout], hidden)
            wall_ms = (time.perf_counter() - start) * 1000.0
            values = critical(
                [timing["dispatch_ms"], timing["expert_ms"], timing["combine_ms"], timing["moe_total_ms"], wall_ms],
                device,
            )
            if fanout not in sampled:
                sample = output[: min(16, num_tokens)].float().cpu().numpy()
                gathered_samples = [None for _ in range(world)] if rank == 0 else None
                dist.gather_object(sample, gathered_samples, dst=0)
                if rank == 0:
                    samples[f"m{num_tokens}_f{fanout}"] = np.stack(gathered_samples)
                sampled.add(fanout)
            else:
                # Keep object collectives balanced across ranks only when a
                # sample is actually requested.
                pass
            if rank == 0:
                row = {
                    "backend": args.backend,
                    "M_per_source_rank": num_tokens,
                    "requested_fanout": fanout,
                    "rep": rep,
                    "order": order,
                    "dispatch_ms": values[0],
                    "expert_ms": values[1],
                    "combine_ms": values[2],
                    "moe_total_ms": values[3],
                    "wall_ms": values[4],
                    "hidden_size": hidden_size,
                    "route_hash": hashlib.sha256(numpy_routes[fanout].tobytes()).hexdigest(),
                    **summaries[fanout],
                }
                rows.append(row)
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    if rank == 0:
        np.savez_compressed(args.output.with_suffix(".samples.npz"), **samples)
        summary = {
            "scope": "QWEN3_VL_EXACT_EXPERT_FORWARD_ONLY_OPERATOR",
            "backend": args.backend,
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "physical_gpus": [4, 5, 6, 7],
            "world_size": world,
            "layer": args.layer,
            "warmup": args.warmup,
            "repetitions": args.reps,
            "precondition_seconds": args.precondition_seconds,
            "rows": len(rows),
            "model": str(args.model.resolve()),
            "torch": torch.__version__,
            "path_semantics": (
                "NCCL all_gather hidden/topk + local vLLM TritonExperts + NCCL reduce_scatter"
                if args.backend == "agrs"
                else "DeepEP HT get_dispatch_layout/dispatch/combine + local vLLM TritonExperts"
            ),
        }
        if args.backend == "deepep_ht":
            import deep_ep

            summary.update(deep_ep=str(Path(deep_ep.__file__).resolve()), deep_ep_sm90=bool(deep_ep.Buffer.is_sm90_compiled()))
        args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sync()
    if deep_buffer is not None:
        deep_buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
