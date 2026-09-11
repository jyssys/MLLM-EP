#!/usr/bin/env python3
"""Captured-layer DeepEP HT replay for semantic-first and EP-refined Top-K.

The replay uses real Qwen3-VL hidden states, routes, router weights and expert
weights.  It measures the exact sparse communication/expert path after omitted
prefix-tail assignments are encoded as expert id -1.  It is an operator
counterfactual, not a production variable-K integration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from poc_flashvep.mllm_moe_transient_branch_compression.analyze_branches import (
    load_layer,
    modality_and_coords,
)
from poc_semantic_first_topk.policy_core import (
    FULL_K,
    branch_risk,
    ep_same_budget_refinement,
    keep_from_k,
    semantic_allocation,
    spatial_schedule_k,
)
from poc_vision_heterogeneous_topk.exact_deepep_replay import load_experts


def build_policies(data: dict[str, np.ndarray], modality: np.ndarray,
                   coords: np.ndarray,
                   fractions: list[float], risk_kinds: list[str],
                   risk_slack: float, schedule_document: dict | None,
                   fixed_ks: list[int], router_thresholds: list[float]) -> dict[str, np.ndarray]:
    policies = {"stock": np.full(len(modality), 8, dtype=np.int16)}
    possible = int(np.count_nonzero(modality == "vision")) * 8
    for kind in risk_kinds:
        risk = branch_risk(data["weights"], data["outputs"], kind)
        for fraction in fractions:
            allocation = semantic_allocation(risk, modality, round(possible * fraction), FULL_K)
            stem = f"{kind}_f{fraction:g}"
            policies[f"semantic_{stem}"] = allocation.k
            policies[f"ep_refined_{stem}_s{risk_slack:g}"] = ep_same_budget_refinement(
                data["ids"], risk, modality, allocation.k, risk_slack=risk_slack).k
    for fixed in fixed_ks:
        k = np.full(len(modality), 8, dtype=np.int16)
        k[modality == "vision"] = int(fixed)
        policies[f"fixed_vision_k{fixed}"] = k
    for threshold in router_thresholds:
        cumulative = np.cumsum(data["weights"], axis=-1)
        chosen = np.sum(cumulative < threshold, axis=-1) + 1
        k = np.full(len(modality), 8, dtype=np.int16)
        k[modality == "vision"] = np.clip(chosen[modality == "vision"], 1, 8)
        policies[f"router_mass_t{threshold:g}"] = k
    if schedule_document is not None:
        for name, record in schedule_document["global_policies"].items():
            base = spatial_schedule_k(modality, coords, record["group_k"])
            policies[f"global_{name}"] = base
            # The deployable semantic schedule is refined using only current
            # router state.  This is the exact same-budget Stage-2 diagnostic.
            policies[f"global_{name}_ep_refined_s{risk_slack:g}"] = (
                ep_same_budget_refinement(data["ids"], data["weights"], modality,
                                          base, risk_slack=risk_slack).k)
    return policies


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--fractions", type=float, nargs="+", default=[.05, .1, .2, .3])
    parser.add_argument("--risk-kinds", nargs="+", default=["router", "contribution"])
    parser.add_argument("--risk-slack", type=float, default=.05)
    parser.add_argument("--schedules", type=Path,
                        help="Optional calibration-aggregate semantic schedules")
    parser.add_argument("--fixed-ks", type=int, nargs="*", default=[])
    parser.add_argument("--router-thresholds", type=float, nargs="*", default=[])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-regex", default=".*",
                        help="Run stock plus policies whose full name matches this regex")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError(f"illegal visibility: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"requires four ranks, got {world}")

    manifest = json.loads((args.capture / "manifest.json").read_text())
    sample = next(row for row in manifest["samples"] if row["sample_id"] == args.sample)
    data = load_layer(args.capture, manifest, sample, args.layer)
    modality, coords = modality_and_coords(sample, data["position"])
    source_tokens = len(modality)
    schedule_document = json.loads(args.schedules.read_text()) if args.schedules else None
    base_policies = build_policies(data, modality, coords, args.fractions, args.risk_kinds,
                                   args.risk_slack, schedule_document,
                                   args.fixed_ks, args.router_thresholds)
    base_policies = {name: value for name, value in base_policies.items()
                     if name == "stock" or re.fullmatch(args.policy_regex, name)}
    copies = max(1, (args.target_tokens + source_tokens - 1) // source_tokens)
    target = args.target_tokens or source_tokens
    original_ids = np.concatenate([data["ids"]] * copies, axis=0)[:target].astype(np.int64)
    original_weights = np.concatenate([data["weights"]] * copies, axis=0)[:target].astype(np.float32)
    hidden_np = np.concatenate([data["hidden"]] * copies, axis=0)[:target]
    modality = np.concatenate([modality] * copies)[:target]
    policies_k = {name: np.concatenate([k] * copies)[:target]
                  for name, k in base_policies.items()}
    hidden = torch.from_numpy(hidden_np).to(torch.bfloat16).contiguous().to(device)
    policies = {}
    policy_meta = {}
    for name, k in policies_k.items():
        keep = keep_from_k(k)
        ids = torch.from_numpy(np.where(keep, original_ids, -1)).to(
            torch.int64).contiguous().to(device)
        weights = torch.from_numpy(np.where(keep, original_weights, 0)).contiguous().to(device)
        policies[name] = (ids, weights, keep)
        ranks = original_ids // 32
        before = np.array([np.count_nonzero(ranks == r) for r in range(4)])
        after = np.array([np.count_nonzero(keep & (ranks == r)) for r in range(4)])
        policy_meta[name] = {
            "vision_assignment_drop": float((~keep[modality == "vision"]).mean()),
            "all_assignment_drop": float((~keep).mean()),
            "max_rank_reduction": 1 - float(after.max()) / max(float(before.max()), 1),
            "rank_load_before": before.tolist(), "rank_load_after": after.tolist(),
        }

    w1, w2, expert_map, first = load_experts(args.model, args.layer, rank, device)
    import deep_ep
    deep_ep.Buffer.set_num_sms(20)
    buffer = deep_ep.Buffer(dist.group.WORLD, 1024 * 1024 * 1024, 0,
                            low_latency_mode=False, num_qps_per_rank=1,
                            explicitly_destroy=True)

    def sync() -> None:
        torch.cuda.synchronize(); dist.barrier()

    def run(name: str):
        ids, weights, _ = policies[name]
        ids_deep = ids.to(deep_ep.topk_idx_t).contiguous()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        nrank, nrdma, nexpert, inrank, layout_event = buffer.get_dispatch_layout(
            ids_deep, 128, async_finish=True, allocate_on_comm_stream=False)
        recv_h, recv_ids, recv_w, _, handle, dispatch_event = buffer.dispatch(
            x=hidden, handle=None, num_tokens_per_rank=nrank,
            num_tokens_per_rdma_rank=nrdma, is_token_in_rank=inrank,
            num_tokens_per_expert=nexpert, topk_idx=ids_deep, topk_weights=weights,
            expert_alignment=1, config=deep_ep.Buffer.get_dispatch_config(world),
            previous_event=layout_event, async_finish=True, allocate_on_comm_stream=False)
        dispatch_event.current_stream_wait(); events[1].record()
        global_ids = torch.where(recv_ids == -1, 127 if first == 0 else 0,
                                 recv_ids.to(torch.int64) + first)
        local = fused_experts(recv_h, w1, w2, recv_w, global_ids,
                              global_num_experts=128, expert_map=expert_map)
        events[2].record()
        combined, _, combine_event = buffer.combine(
            x=local, handle=handle, topk_weights=None,
            config=deep_ep.Buffer.get_combine_config(world), async_finish=True,
            allocate_on_comm_stream=False)
        combine_event.current_stream_wait(); events[3].record(); events[3].synchronize()
        return combined, {"dispatch_ms": events[0].elapsed_time(events[1]),
                          "expert_ms": events[1].elapsed_time(events[2]),
                          "combine_ms": events[2].elapsed_time(events[3]),
                          "moe_ms": events[0].elapsed_time(events[3])}

    names = list(policies)
    for name in names:
        for _ in range(args.warmup):
            run(name)
    sync(); reference, _ = run("stock"); sync()
    schedule = [(rep, name) for rep in range(args.reps) for name in names]
    random.Random(20260911 + args.layer).shuffle(schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        args.output.write_text("")
    started = time.monotonic()
    for order, (rep, name) in enumerate(schedule):
        sync(); output, timing = run(name)
        delta = output.float() - reference.float()
        cosine = torch.nn.functional.cosine_similarity(
            output.float().reshape(1, -1), reference.float().reshape(1, -1))[0]
        rel = delta.norm() / reference.float().norm().clamp_min(1e-20)
        vals = torch.tensor([timing["dispatch_ms"], timing["expert_ms"],
                             timing["combine_ms"], timing["moe_ms"],
                             -float(cosine), float(rel)], dtype=torch.float64, device=device)
        dist.all_reduce(vals, op=dist.ReduceOp.MAX)
        if rank == 0:
            row = {"sample": args.sample, "layer": args.layer, "policy": name,
                   "rep": rep, "order": order, "dispatch_ms": float(vals[0]),
                   "expert_ms": float(vals[1]), "combine_ms": float(vals[2]),
                   "moe_ms": float(vals[3]), "worst_cosine": -float(vals[4]),
                   "worst_rel_l2": float(vals[5]), **policy_meta[name]}
            with args.output.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
    sync()
    if rank == 0:
        args.output.with_suffix(".summary.json").write_text(json.dumps({
            "scope": "REAL_QWEN3_VL_WEIGHT_DEEPEP_HT_CAPTURED_LAYER_REPLAY",
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "deep_ep": str(Path(deep_ep.__file__).resolve()), "sample": args.sample,
            "layer": args.layer, "source_tokens": source_tokens, "target_tokens": target,
            "policies": policy_meta, "warmup": args.warmup, "reps": args.reps,
            "elapsed_seconds": time.monotonic() - started,
            "evidence_boundary": "captured-layer operator replay; no variable-K runtime integration",
        }, indent=2) + "\n")
    buffer.destroy(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
