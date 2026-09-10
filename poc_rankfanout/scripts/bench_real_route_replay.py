#!/usr/bin/env python3
"""Replay fresh Qwen3-VL routes identically through AGRS and DeepEP HT.

Full-model backend runs diverge slightly in hidden values and therefore in
later-layer top-k identities.  This forward-only replay is the spec-required
fallback: one measured real route snapshot is held bit-exact across backends,
while actual Qwen expert weights and the production communication primitives
are used.  It supplies a valid per-layer oracle without pretending mismatched
full-model routes are aligned.
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

from poc_rankfanout.rankfanout.routing import linear_expert_to_rank, summarize_route  # noqa: E402


def load_weight(model: Path, index: dict[str, str], key: str, lo=None, hi=None) -> torch.Tensor:
    with safe_open(str(model / index[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous() if lo is None else handle.get_slice(key)[lo:hi].contiguous()


def load_local_experts(model: Path, layer: int, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    packed = prefix + "gate_up_proj"
    first = rank * 32
    w1 = load_weight(model, index, packed, first, first + 32).transpose(1, 2).contiguous().cuda()
    w2 = load_weight(model, index, prefix + "down_proj", first, first + 32).transpose(1, 2).contiguous().cuda()
    expert_map = torch.full((128,), -1, dtype=torch.int32, device="cuda")
    expert_map[first : first + 32] = torch.arange(32, dtype=torch.int32, device="cuda")
    return w1, w2, expert_map


def critical(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--route-run", type=Path, required=True)
    parser.add_argument("--route-run-id", required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("agrs", "deepep_ht"), required=True)
    parser.add_argument("--weight-layer", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--buffer-mib", type=int, default=1024)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("GPU safety violation")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError("route replay requires four ranks")

    direct = args.route_run / "profile" / "direct_routes"
    suffix = f"_it{args.iteration}_ep0_layer0.npy"
    workloads = sorted(
        path.name[len(args.route_run_id) + 1 : -len(suffix)]
        for path in direct.glob(f"{args.route_run_id}_*_it{args.iteration}_ep0_layer0.npy")
    )
    if not workloads:
        raise RuntimeError("no route snapshots found")
    conditions = [(workload, layer) for workload in workloads for layer in range(48)]
    random.Random(args.seed).shuffle(conditions)

    w1, w2, expert_map = load_local_experts(args.model, args.weight_layer, rank)
    hidden_size = int(w1.shape[2])
    deep_buffer = None
    comm_stream = None
    if args.backend == "deepep_ht":
        import deep_ep

        deep_ep.Buffer.set_num_sms(20)
        deep_buffer = deep_ep.Buffer(
            dist.group.WORLD, args.buffer_mib * 1024 * 1024, 0,
            low_latency_mode=False, num_qps_per_rank=1, explicitly_destroy=True,
        )
        comm_stream = deep_buffer.get_comm_stream()

    def sync() -> None:
        torch.cuda.synchronize()
        dist.barrier()

    def run_agrs(route: torch.Tensor, hidden: torch.Tensor):
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
        local = fused_experts(gathered_h, w1, w2, gathered_w, gathered_i, global_num_experts=128, expert_map=expert_map)
        events[2].record()
        output = torch.empty_like(hidden)
        dist.reduce_scatter_tensor(output, local.contiguous(), op=dist.ReduceOp.SUM)
        events[3].record(); events[3].synchronize()
        return output, [events[0].elapsed_time(events[1]), events[1].elapsed_time(events[2]), events[2].elapsed_time(events[3]), events[0].elapsed_time(events[3])]

    def run_deepep(route: torch.Tensor, hidden: torch.Tensor):
        assert deep_buffer is not None and comm_stream is not None
        import deep_ep

        weights = torch.full(route.shape, 1.0 / 8, dtype=torch.float32, device=device)
        ids = route.to(deep_ep.topk_idx_t).contiguous()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        events[0].record()
        num_rank, num_rdma, num_expert, in_rank, layout_event = deep_buffer.get_dispatch_layout(ids, 128, async_finish=True, allocate_on_comm_stream=False)
        recv_h, recv_ids, recv_w, _, handle, dispatch_event = deep_buffer.dispatch(
            x=hidden, handle=None, num_tokens_per_rank=num_rank,
            num_tokens_per_rdma_rank=num_rdma, is_token_in_rank=in_rank,
            num_tokens_per_expert=num_expert, topk_idx=ids, topk_weights=weights,
            expert_alignment=1, config=deep_ep.Buffer.get_dispatch_config(world),
            previous_event=layout_event, async_finish=True, allocate_on_comm_stream=False,
        )
        events[1].record(comm_stream); dispatch_event.current_stream_wait(); events[2].record()
        first = rank * 32
        global_ids = torch.where(recv_ids == -1, 127 if first == 0 else 0, recv_ids.to(torch.int64) + first)
        local = fused_experts(recv_h, w1, w2, recv_w, global_ids, global_num_experts=128, expert_map=expert_map)
        events[3].record()
        output, _, combine_event = deep_buffer.combine(
            x=local, handle=handle, topk_weights=None,
            config=deep_ep.Buffer.get_combine_config(world), async_finish=True,
            allocate_on_comm_stream=False,
        )
        events[4].record(comm_stream); combine_event.current_stream_wait(); events[5].record(); events[5].synchronize()
        return output, [events[0].elapsed_time(events[2]), events[2].elapsed_time(events[3]), events[3].elapsed_time(events[5]), events[0].elapsed_time(events[5])]

    run = run_agrs if args.backend == "agrs" else run_deepep
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        args.output.write_text("")
    hidden_by_workload: dict[str, torch.Tensor] = {}
    route_by_condition: dict[tuple[str, int], torch.Tensor] = {}
    summary_by_condition: dict[tuple[str, int], dict] = {}
    for workload, layer in conditions:
        path = direct / f"{args.route_run_id}_{workload}_it{args.iteration}_ep{rank}_layer{layer}.npy"
        route_np = np.load(path).astype(np.int64)
        route_by_condition[(workload, layer)] = torch.from_numpy(route_np).to(device)
        if workload not in hidden_by_workload:
            generator = torch.Generator(device=device).manual_seed(args.seed + rank * 100003 + route_np.shape[0])
            hidden_by_workload[workload] = torch.randn((route_np.shape[0], hidden_size), generator=generator, dtype=torch.bfloat16, device=device)
        if rank == 0:
            rank_routes = [
                np.load(direct / f"{args.route_run_id}_{workload}_it{args.iteration}_ep{source}_layer{layer}.npy").astype(np.int64)
                for source in range(4)
            ]
            summary = summarize_route(np.concatenate(rank_routes), linear_expert_to_rank()).to_dict()
            summary["route_hash"] = hashlib.sha256(b"".join(item.tobytes() for item in rank_routes)).hexdigest()
            summary_by_condition[(workload, layer)] = summary

    # Global warmup/preconditioning, then per-shape warmup.
    first_condition = conditions[0]
    for _ in range(3):
        run(route_by_condition[first_condition], hidden_by_workload[first_condition[0]])
    clock = torch.randn((4096, 4096), dtype=torch.bfloat16, device=device)
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        torch.mm(clock, clock)
    del clock; sync()

    samples: dict[str, np.ndarray] = {}
    for order, (workload, layer) in enumerate(conditions):
        route = route_by_condition[(workload, layer)]
        hidden = hidden_by_workload[workload]
        for _ in range(args.warmup):
            run(route, hidden)
        sync()
        for rep in range(args.reps):
            sync(); begin = time.perf_counter(); output, timing = run(route, hidden); wall_ms = (time.perf_counter() - begin) * 1000.0
            values = critical([*timing, wall_ms], device)
            if rep == 0:
                sample = output[: min(8, output.shape[0])].float().cpu().numpy()
                gathered = [None] * world if rank == 0 else None
                dist.gather_object(sample, gathered, dst=0)
                if rank == 0:
                    samples[f"{workload}_layer{layer}"] = np.stack(gathered)
            if rank == 0:
                row = {
                    "backend": args.backend, "workload_id": workload,
                    "phase": "prefill", "step_id": args.iteration,
                    "layer_id": layer, "M_per_source_rank": int(route.shape[0]),
                    "rep": rep, "order": order,
                    "dispatch_ms": values[0], "expert_ms": values[1],
                    "combine_ms": values[2], "moe_total_ms": values[3],
                    "wall_ms": values[4], **summary_by_condition[(workload, layer)],
                }
                with args.output.open("a") as handle:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    if rank == 0:
        np.savez_compressed(args.output.with_suffix(".samples.npz"), **samples)
        args.output.with_suffix(".summary.json").write_text(json.dumps({
            "scope": "REAL_QWEN_ROUTE_EXACT_REPLAY", "backend": args.backend,
            "route_source": str(args.route_run), "route_run_id": args.route_run_id,
            "iteration": args.iteration, "conditions": len(conditions),
            "warmup": args.warmup, "repetitions": args.reps,
            "weight_layer": args.weight_layer, "physical_gpus": [4, 5, 6, 7],
        }, indent=2) + "\n")
    sync()
    if deep_buffer is not None:
        deep_buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
