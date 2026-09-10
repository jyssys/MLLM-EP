#!/usr/bin/env python3
"""Exact four-rank DeepEP replay of visual heterogeneous Top-K policies.

This is a captured-layer diagnostic. It preserves real Qwen3-VL hidden states,
expert IDs, router weights, expert weights, and the stock DeepEP HT/Triton path.
Only selected visual expert assignments are masked. It is not a full-model
variable-k implementation and cannot establish benchmark quality by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from poc_flashvep.mllm_moe_transient_branch_compression.analyze_branches import (
    load_layer,
    modality_and_coords,
)
from poc_vision_heterogeneous_topk.analyze_policy_oracle import (
    budget_keep,
    random_keep,
    semantic_keep,
    tail_aware_keep,
)


def load_weight(model: Path, index: dict[str, str], key: str,
                lo: int | None = None, hi: int | None = None) -> torch.Tensor:
    with safe_open(str(model / index[key]), framework="pt", device="cpu") as handle:
        if lo is None:
            return handle.get_tensor(key).contiguous()
        return handle.get_slice(key)[lo:hi].contiguous()


def load_experts(model: Path, layer: int, rank: int, device: torch.device):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    packed = prefix + "gate_up_proj"
    if packed not in index:
        prefix = f"model.layers.{layer}.mlp.experts."
        packed = prefix + "gate_up_proj"
    first = rank * 32
    if packed in index:
        w1 = load_weight(model, index, packed, first, first + 32).transpose(1, 2).contiguous()
        w2 = load_weight(model, index, prefix + "down_proj", first, first + 32).transpose(1, 2).contiguous()
    else:
        gate_up, down = [], []
        for expert in range(first, first + 32):
            gate = load_weight(model, index, prefix + f"{expert}.gate_proj.weight")
            up = load_weight(model, index, prefix + f"{expert}.up_proj.weight")
            gate_up.append(torch.cat((gate, up), dim=0))
            down.append(load_weight(model, index, prefix + f"{expert}.down_proj.weight"))
        w1, w2 = torch.stack(gate_up), torch.stack(down)
    expert_map = torch.full((128,), -1, dtype=torch.int32, device=device)
    expert_map[first:first + 32] = torch.arange(32, dtype=torch.int32, device=device)
    return w1.to(device), w2.to(device), expert_map, first


def parse_policy(name: str, ids: np.ndarray, weights: np.ndarray,
                 modality: np.ndarray) -> np.ndarray:
    if name == "full_k8":
        return np.ones_like(weights, dtype=bool)
    parts = name.split("_")
    if parts[0] == "budget":
        return budget_keep(ids, weights, modality, float(parts[2]), parts[1])
    if parts[0] == "random":
        return random_keep(modality, int(parts[1][1:]), 20260911)
    if parts[0] == "semantic":
        return semantic_keep(weights, modality, int(parts[1][1:]))
    if parts[0] == "tail":
        k = int(parts[1][1:])
        slack = float(parts[2].replace("s", "")) if len(parts) > 2 else .01
        return tail_aware_keep(ids, weights, modality, k, slack)
    raise ValueError(name)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--target-tokens", type=int, default=0,
                        help="Tile the real captured token rows to this M for a scale replay")
    parser.add_argument("--policies", nargs="+", default=[
        "full_k8", "random_k6", "semantic_k6", "tail_k6_s0.01",
        "random_k4", "semantic_k4", "tail_k4_s0.01",
    ])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--renormalize", action="store_true",
                        help="Renormalize retained router weights per token")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError(f"illegal visibility: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"requires exactly four ranks, got {world}")

    manifest = json.loads((args.capture / "manifest.json").read_text())
    sample = next(row for row in manifest["samples"] if row["sample_id"] == args.sample)
    data = load_layer(args.capture, manifest, sample, args.layer)
    modality, _ = modality_and_coords(sample, data["position"])
    source_tokens = len(modality)
    if args.target_tokens:
        copies = (args.target_tokens + source_tokens - 1) // source_tokens
        # Runtime replay needs only the expert input and routing tensors. Keep
        # the observer-heavy per-branch outputs at their captured size.
        for key in ("hidden", "ids", "weights"):
            data[key] = np.concatenate([data[key]] * copies, axis=0)[:args.target_tokens]
        modality = np.concatenate([modality] * copies)[:args.target_tokens]
    hidden = torch.from_numpy(data["hidden"]).to(torch.bfloat16).contiguous().to(device)
    original_ids = data["ids"].astype(np.int64)
    original_weights = data["weights"].astype(np.float32)
    policies = {}
    for name in args.policies:
        keep = parse_policy(name, original_ids, original_weights, modality)
        ids = torch.from_numpy(np.where(keep, original_ids, -1)).to(torch.int64).contiguous().to(device)
        kept_weights = np.where(keep, original_weights, 0)
        if args.renormalize:
            kept_weights = kept_weights / np.maximum(kept_weights.sum(axis=1, keepdims=True), 1e-12)
        weights = torch.from_numpy(kept_weights).contiguous().to(device)
        policies[name] = (ids, weights, keep)

    w1, w2, expert_map, first = load_experts(args.model, args.layer, rank, device)
    import deep_ep
    deep_ep.Buffer.set_num_sms(20)
    buffer = deep_ep.Buffer(dist.group.WORLD, 1024 * 1024 * 1024, 0,
                            low_latency_mode=False, num_qps_per_rank=1,
                            explicitly_destroy=True)

    def sync() -> None:
        torch.cuda.synchronize()
        dist.barrier()

    def run(name: str):
        ids, weights, _ = policies[name]
        ids_deep = ids.to(deep_ep.topk_idx_t).contiguous()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        layout = buffer.get_dispatch_layout(ids_deep, 128, async_finish=True,
                                            allocate_on_comm_stream=False)
        nrank, nrdma, nexpert, inrank, layout_event = layout
        recv_h, recv_ids, recv_w, _, handle, dispatch_event = buffer.dispatch(
            x=hidden, handle=None, num_tokens_per_rank=nrank,
            num_tokens_per_rdma_rank=nrdma, is_token_in_rank=inrank,
            num_tokens_per_expert=nexpert, topk_idx=ids_deep, topk_weights=weights,
            expert_alignment=1, config=deep_ep.Buffer.get_dispatch_config(world),
            previous_event=layout_event, async_finish=True, allocate_on_comm_stream=False)
        dispatch_event.current_stream_wait()
        events[1].record()
        global_ids = torch.where(recv_ids == -1, 127 if first == 0 else 0,
                                 recv_ids.to(torch.int64) + first)
        local = fused_experts(recv_h, w1, w2, recv_w, global_ids,
                              global_num_experts=128, expert_map=expert_map)
        events[2].record()
        combined, _, combine_event = buffer.combine(
            x=local, handle=handle, topk_weights=None,
            config=deep_ep.Buffer.get_combine_config(world), async_finish=True,
            allocate_on_comm_stream=False)
        combine_event.current_stream_wait()
        events[3].record(); events[3].synchronize()
        return combined, {
            "dispatch_ms": events[0].elapsed_time(events[1]),
            "expert_ms": events[1].elapsed_time(events[2]),
            "combine_ms": events[2].elapsed_time(events[3]),
            "moe_ms": events[0].elapsed_time(events[3]),
        }

    for name in args.policies:
        for _ in range(args.warmup):
            run(name)
    sync()
    reference, _ = run("full_k8")
    sync()
    schedule = [(rep, name) for rep in range(args.reps) for name in args.policies]
    random.Random(20260911).shuffle(schedule)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        args.output.write_text("")
    start = time.monotonic()
    for order, (rep, name) in enumerate(schedule):
        sync()
        output, timing = run(name)
        delta = output.float() - reference.float()
        cosine = torch.nn.functional.cosine_similarity(
            output.float().reshape(1, -1), reference.float().reshape(1, -1))[0]
        rel = delta.norm() / reference.float().norm().clamp_min(1e-20)
        _, weights, keep = policies[name]
        dropped = ~keep
        vals = torch.tensor([
            timing["dispatch_ms"], timing["expert_ms"], timing["combine_ms"], timing["moe_ms"],
            -float(cosine), float(rel), float(dropped.mean()),
            float(original_weights[dropped].sum() / max(original_weights.sum(), 1e-12)),
        ], dtype=torch.float64, device=device)
        dist.all_reduce(vals, op=dist.ReduceOp.MAX)
        if rank == 0:
            row = {"sample": args.sample, "layer": args.layer, "policy": name,
                   "rep": rep, "order": order, "dispatch_ms": float(vals[0]),
                   "expert_ms": float(vals[1]), "combine_ms": float(vals[2]),
                   "moe_ms": float(vals[3]), "worst_cosine": -float(vals[4]),
                   "worst_rel_l2": float(vals[5]), "all_assignment_drop": float(vals[6]),
                   "all_router_mass_drop": float(vals[7])}
            rows.append(row)
            with args.output.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
    sync()
    if rank == 0:
        args.output.with_suffix(".summary.json").write_text(json.dumps({
            "scope": "REAL_QWEN3_VL_WEIGHT_DEEPEP_HT_CAPTURED_LAYER_REPLAY",
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "deep_ep": str(Path(deep_ep.__file__).resolve()),
            "sample": args.sample, "layer": args.layer, "tokens": len(data["ids"]),
            "source_tokens": source_tokens, "target_tokens": args.target_tokens,
            "vision_tokens": int((modality == "vision").sum()),
            "policies": args.policies, "warmup": args.warmup, "reps": args.reps,
            "renormalize": args.renormalize,
            "elapsed_s": time.monotonic() - start,
            "evidence_boundary": "captured-layer operator replay; not full-model variable-k",
        }, indent=2) + "\n")
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
