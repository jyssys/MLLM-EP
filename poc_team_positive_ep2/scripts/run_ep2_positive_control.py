#!/usr/bin/env python3
"""Semantics-preserving two-rank expert-parallel SDAR/TEAM diagnostic.

The released SDAR model stores all experts in a Python ``ModuleList``.  This
runner leaves the released decoder untouched and replaces only the physical
execution of each sparse MoE block after checkpoint load:

* rank 0 computes the released router and TEAM live/cold masks;
* token/expert rows are sent to the rank owning that expert with NCCL A2A;
* each rank invokes only its 64 resident experts;
* weighted expert outputs return to rank 0 with NCCL A2A and are combined;
* the exact combined tensor is broadcast so the replicated non-MoE path stays
  bitwise aligned between ranks.

This is deliberately a reference EP2 substrate, not a production backend.
It exists to prove true ownership/remote dispatch/combine while preserving the
official TEAM decisions.  Clean timing and event-heavy structural timing are
separate modes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import socket
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_positive_control import load_official_generator, stop_ids, tokenize


EXPECTED_VISIBLE = "6,7"
EXPECTED_UUIDS = (
    "GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28",
    "GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b",
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RemovedExpert(nn.Module):
    """Parameter-free marker installed for experts owned by the other rank."""

    def forward(self, _x):  # pragma: no cover - reaching this is a runtime bug
        raise RuntimeError("attempted to execute a non-owned expert")


class ModuleEventRecorder:
    """Low-intrusion CUDA-event timing for non-MoE module boundaries."""

    def __init__(self):
        self.pending: dict[str, list[tuple[torch.cuda.Event, int]]] = defaultdict(list)
        self.pairs: list[tuple[str, int, torch.cuda.Event, torch.cuda.Event]] = []
        self.handles = []

    def add(self, module: nn.Module, name: str) -> None:
        def before(_module, _args):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            tensor = _args[0] if _args and torch.is_tensor(_args[0]) else None
            rows = int(tensor.numel() // tensor.shape[-1]) if tensor is not None else -1
            self.pending[name].append((event, rows))

        def after(_module, _args, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start, rows = self.pending[name].pop()
            self.pairs.append((name, rows, start, end))

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after))

    def finish(self) -> dict[str, Any]:
        torch.cuda.synchronize()
        values: dict[str, list[float]] = defaultdict(list)
        for name, rows, start, end in self.pairs:
            values[name].append(float(start.elapsed_time(end)))
            values[f"{name}_rows_{rows}"].append(float(start.elapsed_time(end)))
        for handle in self.handles:
            handle.remove()
        return {
            name: {
                "count": len(samples),
                "sum_ms": sum(samples),
                "mean_ms": sum(samples) / len(samples),
                "max_ms": max(samples),
            }
            for name, samples in values.items()
        }

    def reset(self) -> None:
        torch.cuda.synchronize()
        self.pending.clear()
        self.pairs.clear()


class EP2Runtime:
    def __init__(self, rank: int, instrument: bool, local_expert_backend: str,
                 analyze_branch_duplicates: bool):
        self.rank = rank
        self.world = 2
        self.instrument = instrument
        self.local_expert_backend = local_expert_backend
        self.analyze_branch_duplicates = analyze_branch_duplicates
        self.calls = 0
        self.layers: list[dict[str, Any]] = []
        self._events: list[tuple[dict[str, Any], dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]]] = []

    def event_pair(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def install(self, model: nn.Module) -> dict[str, Any]:
        num_experts = int(model.config.num_experts)
        if num_experts != 128 or num_experts % self.world:
            raise RuntimeError(f"expected 128 experts divisible by EP2, got {num_experts}")
        per_rank = num_experts // self.world
        ownership = {}
        for layer_id, layer in enumerate(model.model.layers):
            block = layer.mlp
            if len(block.experts) != num_experts:
                raise RuntimeError(f"layer {layer_id}: unexpected expert count")
            if self.local_expert_backend == "vllm_fused":
                local = list(range(self.rank * per_rank, (self.rank + 1) * per_rank))
                w1 = torch.stack([
                    torch.cat([
                        block.experts[i].gate_proj.weight,
                        block.experts[i].up_proj.weight,
                    ], dim=0)
                    for i in local
                ]).contiguous()
                w2 = torch.stack([
                    block.experts[i].down_proj.weight for i in local
                ]).contiguous()
                block.register_buffer("_ep_w1", w1)
                block.register_buffer("_ep_w2", w2)
                for expert_id in range(num_experts):
                    block.experts[expert_id] = RemovedExpert()
            else:
                for expert_id in range(num_experts):
                    owner = expert_id // per_rank
                    if owner != self.rank:
                        block.experts[expert_id] = RemovedExpert()
            block.forward = types.MethodType(self._make_forward(layer_id), block)
            ownership[layer_id] = [self.rank * per_rank, (self.rank + 1) * per_rank - 1]
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "num_experts": num_experts,
            "experts_per_rank": per_rank,
            "rank": self.rank,
            "owned_expert_range": [self.rank * per_rank, (self.rank + 1) * per_rank - 1],
            "layer_ownership": ownership,
            "local_expert_backend": self.local_expert_backend,
        }

    def _make_forward(self, layer_id: int):
        runtime = self

        def ep_forward(block, hidden_states, past_hidden_states=None,
                       decoded_index=None, expert_limit_index=None):
            return runtime.execute(
                layer_id, block, hidden_states, past_hidden_states,
                decoded_index, expert_limit_index,
            )

        return ep_forward

    def execute(self, layer_id, block, hidden_states, past_hidden_states,
                decoded_index, expert_limit_index):
        rank = self.rank
        device = hidden_states.device
        batch, sequence, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        num_rows = flat.shape[0]
        router_logits = torch.zeros(
            (num_rows, block.num_experts), dtype=hidden_states.dtype, device=device
        )
        final = torch.zeros_like(flat)

        if past_hidden_states is None:
            compute_mask = torch.ones(num_rows, dtype=torch.bool, device=device)
        else:
            if decoded_index is None:
                decoded_index = torch.zeros(
                    batch, sequence, dtype=torch.bool, device=device
                )
            compute_mask = (~decoded_index).reshape(-1)
            final[~compute_mask] = past_hidden_states.reshape(-1, hidden_dim)[~compute_mask]

        stage_events = {}
        if self.instrument:
            total_start, total_end = self.event_pair()
            router_start, router_end = self.event_pair()

        selected = torch.empty((0, block.top_k), dtype=torch.long, device=device)
        weights = torch.empty((0, block.top_k), dtype=hidden_states.dtype, device=device)
        compute_hidden = flat[compute_mask] if rank == 0 else torch.empty(
            (0, hidden_dim), dtype=flat.dtype, device=device
        )
        if rank == 0 and compute_hidden.shape[0]:
            compute_logits = block.gate(compute_hidden)
            probs = F.softmax(compute_logits, dim=1, dtype=torch.float)
            weights, selected = torch.topk(probs, block.top_k, dim=-1)
            if past_hidden_states is not None:
                if expert_limit_index is None:
                    expert_limit_index = torch.zeros(
                        batch, sequence, dtype=torch.bool, device=device
                    )
                limited = expert_limit_index.reshape(-1)[compute_mask]
                if limited.any():
                    unrestricted = selected[~limited]
                    necessary = torch.zeros(block.num_experts, dtype=torch.bool, device=device)
                    if unrestricted.numel():
                        necessary[unrestricted.flatten()] = True
                    probs[:, ~necessary] = 0
                    weights, selected = torch.topk(probs, block.top_k, dim=-1)
            if block.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            weights = weights.to(flat.dtype)
            router_logits[compute_mask] = compute_logits.to(flat.dtype)
        if self.instrument:
            router_end.record()
            prepare_start, prepare_end = self.event_pair()

        # Only rank 0 is the replicated-model source.  Sort branch rows by
        # destination so NCCL A2A receives contiguous per-owner segments.
        if rank == 0 and selected.numel():
            branch_token = torch.arange(
                selected.shape[0], device=device, dtype=torch.long
            ).repeat_interleave(block.top_k)
            expert_ids = selected.reshape(-1)
            branch_weights = weights.reshape(-1)
            exact_duplicate_assignments = 0
            near_duplicate_assignments_1pct = 0
            near_duplicate_assignments_5pct = 0
            route_duplicate_assignments = 0
            if self.analyze_branch_duplicates:
                compute_positions = torch.where(compute_mask)[0]
                original_positions = compute_positions.index_select(0, branch_token)
                branch_period = 32 if sequence % 32 == 0 and sequence >= 128 else sequence
                logical_positions = torch.remainder(original_positions, branch_period)
                position_expert_key = logical_positions * block.num_experts + expert_ids
                _, route_counts = torch.unique(position_expert_key, return_counts=True)
                route_duplicate_assignments = int((route_counts - 1).clamp_min(0).sum().item())
                for key in torch.unique(position_expert_key):
                    members = torch.where(position_expert_key == key)[0]
                    if members.numel() < 2:
                        continue
                    xs = compute_hidden.index_select(0, branch_token[members]).float()
                    reference = xs[0]
                    rel = torch.linalg.vector_norm(xs - reference, dim=-1) / torch.linalg.vector_norm(
                        reference
                    ).clamp_min(1e-12)
                    exact_duplicate_assignments += int((rel[1:] == 0).sum().item())
                    near_duplicate_assignments_1pct += int((rel[1:] <= 0.01).sum().item())
                    near_duplicate_assignments_5pct += int((rel[1:] <= 0.05).sum().item())
            owners = torch.div(
                expert_ids, block.num_experts // self.world, rounding_mode="floor"
            )
            order = torch.argsort(owners, stable=True)
            send_hidden = compute_hidden.index_select(0, branch_token[order]).contiguous()
            send_expert = expert_ids[order].contiguous()
            send_token = branch_token[order].contiguous()
            send_weight = branch_weights[order].contiguous()
            send_counts = torch.bincount(owners, minlength=self.world).to(torch.int64)
        else:
            send_hidden = torch.empty((0, hidden_dim), dtype=flat.dtype, device=device)
            send_expert = torch.empty((0,), dtype=torch.long, device=device)
            send_token = torch.empty((0,), dtype=torch.long, device=device)
            send_weight = torch.empty((0,), dtype=flat.dtype, device=device)
            send_counts = torch.zeros(self.world, dtype=torch.int64, device=device)
            route_duplicate_assignments = 0
            exact_duplicate_assignments = 0
            near_duplicate_assignments_1pct = 0
            near_duplicate_assignments_5pct = 0

        dist.broadcast(send_counts, src=0)
        counts = [int(x) for x in send_counts.cpu().tolist()]
        input_splits = counts if rank == 0 else [0, 0]
        output_splits = [counts[rank], 0]
        recv_rows = counts[rank]
        recv_hidden = torch.empty((recv_rows, hidden_dim), dtype=flat.dtype, device=device)
        recv_expert = torch.empty((recv_rows,), dtype=torch.long, device=device)
        recv_token = torch.empty((recv_rows,), dtype=torch.long, device=device)
        recv_weight = torch.empty((recv_rows,), dtype=flat.dtype, device=device)
        if self.instrument:
            prepare_end.record()
            dispatch_start, dispatch_end = self.event_pair()
        dist.all_to_all_single(recv_hidden, send_hidden,
                               output_split_sizes=output_splits,
                               input_split_sizes=input_splits)
        dist.all_to_all_single(recv_expert, send_expert,
                               output_split_sizes=output_splits,
                               input_split_sizes=input_splits)
        dist.all_to_all_single(recv_token, send_token,
                               output_split_sizes=output_splits,
                               input_split_sizes=input_splits)
        dist.all_to_all_single(recv_weight, send_weight,
                               output_split_sizes=output_splits,
                               input_split_sizes=input_splits)
        if self.instrument:
            dispatch_end.record()
            expert_start, expert_end = self.event_pair()

        local_result = torch.empty_like(recv_hidden)
        local_begin = rank * (block.num_experts // self.world)
        local_end = (rank + 1) * (block.num_experts // self.world)
        if recv_rows and self.local_expert_backend == "vllm_fused":
            from vllm.model_executor.layers.fused_moe import fused_experts
            local_result = fused_experts(
                hidden_states=recv_hidden,
                w1=block._ep_w1,
                w2=block._ep_w2,
                topk_weights=recv_weight.unsqueeze(-1),
                topk_ids=(recv_expert - local_begin).unsqueeze(-1),
                inplace=False,
                activation="silu",
                is_act_and_mul=True,
            )
        else:
            for expert_id in range(local_begin, local_end):
                positions = torch.where(recv_expert == expert_id)[0]
                if positions.numel():
                    values = block.experts[expert_id](recv_hidden.index_select(0, positions))
                    values = values * recv_weight.index_select(0, positions).unsqueeze(-1)
                    local_result.index_copy_(0, positions, values.to(flat.dtype))
        if self.instrument:
            expert_end.record()
            combine_start, combine_end = self.event_pair()

        reverse_input = [recv_rows, 0]
        reverse_output = counts if rank == 0 else [0, 0]
        returned = torch.empty(
            (sum(counts) if rank == 0 else 0, hidden_dim),
            dtype=flat.dtype, device=device,
        )
        dist.all_to_all_single(returned, local_result,
                               output_split_sizes=reverse_output,
                               input_split_sizes=reverse_input)
        if rank == 0 and returned.numel():
            compute_final = torch.zeros_like(compute_hidden)
            compute_final.index_add_(0, send_token, returned)
            final[compute_mask] = compute_final
        dist.broadcast(final, src=0)
        if self.instrument:
            combine_end.record()
            total_end.record()

        row = {
            "call_id": self.calls,
            "layer": layer_id,
            "rank": rank,
            "physical_rows": int(num_rows),
            "computed_rows": int(compute_mask.sum().item()),
            "assignments": int(sum(counts)),
            "local_assignments": int(counts[rank]),
            "remote_assignments_from_source": int(counts[1]),
            "dispatch_bytes_hidden": int(sum(counts) * hidden_dim * flat.element_size()),
            "combine_bytes_hidden": int(sum(counts) * hidden_dim * flat.element_size()),
            "active_local_experts": int(torch.unique(recv_expert).numel()),
            "cross_branch_same_position_route_duplicates": route_duplicate_assignments,
            "cross_branch_exact_input_duplicates": exact_duplicate_assignments,
            "cross_branch_near_input_duplicates_1pct": near_duplicate_assignments_1pct,
            "cross_branch_near_input_duplicates_5pct": near_duplicate_assignments_5pct,
        }
        self.calls += 1
        if self.instrument:
            stage_events = {
                "router": (router_start, router_end),
                "prepare": (prepare_start, prepare_end),
                "dispatch": (dispatch_start, dispatch_end),
                "expert": (expert_start, expert_end),
                "combine": (combine_start, combine_end),
                "moe": (total_start, total_end),
            }
            self._events.append((row, stage_events))
        else:
            self.layers.append(row)
        return final.reshape(batch, sequence, hidden_dim), router_logits

    def finish(self) -> list[dict[str, Any]]:
        torch.cuda.synchronize()
        for row, pairs in self._events:
            for name, (start, end) in pairs.items():
                row[f"{name}_ms"] = float(start.elapsed_time(end))
            self.layers.append(row)
        self._events.clear()
        return self.layers

    def reset(self) -> None:
        torch.cuda.synchronize()
        self.calls = 0
        self.layers.clear()
        self._events.clear()


def run_rank(rank: int, port: int, args) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
        device_id=torch.device(f"cuda:{rank}"),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wrapper = (
        "huggingface_bd3_original.py" if args.mode == "baseline"
        else "huggingface_bd3_decoded_jump_expert_limit_speculative.py"
    )
    generator = load_official_generator(
        args.team_repo / "evaluation/opencompass/opencompass/models" / wrapper
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": rank},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    validation_hidden = None
    validation_reference = None
    validation_reference_logits = None
    if args.validate_layer:
        validation_hidden = torch.empty(
            (1, args.validation_rows, model.config.hidden_size),
            dtype=torch.float16, device=rank,
        )
        if rank == 0:
            generator_state = torch.Generator(device=rank).manual_seed(args.seed + 991)
            validation_hidden.normal_(mean=0.0, std=0.2, generator=generator_state)
        dist.broadcast(validation_hidden, src=0)
        if rank == 0:
            with torch.inference_mode():
                validation_reference, validation_reference_logits = model.model.layers[0].mlp(
                    validation_hidden
                )
                validation_reference = validation_reference.detach().clone()
                validation_reference_logits = validation_reference_logits.detach().clone()
        dist.barrier()
    runtime = EP2Runtime(
        rank, args.instrument, args.local_expert_backend,
        args.analyze_branch_duplicates,
    )
    ownership = runtime.install(model)
    layer_validation = None
    if args.validate_layer:
        with torch.inference_mode():
            validation_actual, validation_actual_logits = model.model.layers[0].mlp(
                validation_hidden
            )
        torch.cuda.synchronize(rank)
        if rank == 0:
            ref = validation_reference.float().reshape(-1)
            actual = validation_actual.float().reshape(-1)
            layer_validation = {
                "layer": 0,
                "rows": args.validation_rows,
                "output_cosine": float(F.cosine_similarity(ref, actual, dim=0).item()),
                "output_rel_l2": float(
                    torch.linalg.vector_norm(actual - ref).div(
                        torch.linalg.vector_norm(ref).clamp_min(1e-12)
                    ).item()
                ),
                "output_max_abs": float((validation_actual.float() - validation_reference.float()).abs().max().item()),
                "router_logits_max_abs": float(
                    (validation_actual_logits.float() - validation_reference_logits.float()).abs().max().item()
                ),
            }
        runtime.reset()
    module_events = None
    if args.instrument:
        module_events = ModuleEventRecorder()
        for layer_id, layer in enumerate(model.model.layers):
            module_events.add(layer.self_attn, "attention")
            module_events.add(layer, "decoder_layer")
        module_events.add(model.lm_head, "lm_head")
    memory_after_shard = int(torch.cuda.memory_allocated(rank))

    if args.validation_only:
        payload = {
            "mode": args.mode,
            "rank": rank,
            "physical_gpu": 6 + rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "backend": "torch.distributed.nccl_all_to_all_single_reference",
            "local_expert_backend": args.local_expert_backend,
            "topology": {"tp": 1, "dp": 1, "ep": 2},
            "ownership": ownership,
            "memory_after_shard_bytes": memory_after_shard,
            "layer_validation": layer_validation,
            "records": [],
            "stage_rows": [],
        }
        out = args.output.with_name(args.output.stem + f"_rank{rank}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        dist.barrier()
        dist.destroy_process_group()
        return

    prompts = [json.loads(x) for x in args.prompts.read_text().splitlines() if x.strip()]
    wanted = set(args.ids.split(",")) if args.ids else None
    if wanted is not None:
        prompts = [x for x in prompts if x.get("id") in wanted]
    if args.max_samples is not None:
        prompts = prompts[:args.max_samples]

    def generate_once(item):
        tokens = tokenize(tokenizer, item["prompt"], model.device)
        output = generator(
            model=model,
            tokenizer=tokenizer,
            prompt=tokens,
            mask_id=151669,
            gen_length=args.gen_length,
            block_length=args.block_length,
            denoising_steps=args.denoising_steps,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            remasking="low_confidence",
            threshold=args.threshold,
            stopping_criteria_idx=stop_ids(tokenizer),
        )
        return output, int(tokens["input_ids"].shape[1])

    for i in range(args.warmup):
        generate_once(prompts[i % len(prompts)])
        torch.cuda.synchronize(rank)
        dist.barrier()
    runtime.reset()
    if module_events is not None:
        module_events.reset()

    records = []
    dist.barrier()
    request_start = time.perf_counter()
    for item in prompts:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        dist.barrier()
        start = time.perf_counter()
        output, prompt_length = generate_once(item)
        torch.cuda.synchronize(rank)
        elapsed = time.perf_counter() - start
        if rank == 0:
            decoded = tokenizer.batch_decode(
                output[:, prompt_length:], skip_special_tokens=False
            )[0].replace("<|MASK|>", "")
            records.append({
                "id": item.get("id"), "task": item.get("task"),
                "reference": item.get("reference"), "elapsed_s": elapsed,
                "output": decoded,
            })
    request_wall = time.perf_counter() - request_start
    layers = runtime.finish()
    module_summary = module_events.finish() if module_events is not None else None

    payload = {
        "mode": args.mode,
        "rank": rank,
        "physical_gpu": 6 + rank,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "backend": "torch.distributed.nccl_all_to_all_single_reference",
        "local_expert_backend": args.local_expert_backend,
        "topology": {"tp": 1, "dp": 1, "ep": 2},
        "ownership": ownership,
        "memory_after_shard_bytes": memory_after_shard,
        "wrapper": wrapper,
        "generation": {
            "gen_length": args.gen_length, "block_length": args.block_length,
            "denoising_steps": args.denoising_steps, "threshold": args.threshold,
        },
        "instrument": args.instrument,
        "request_wall_s": request_wall,
        "records": records if rank == 0 else None,
        "stage_rows": layers,
        "module_event_summary": module_summary,
        "layer_validation": layer_validation,
    }
    out = args.output.with_name(args.output.stem + f"_rank{rank}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "team"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--team-repo", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument(
        "--local-expert-backend", choices=["python", "vllm_fused"],
        default="vllm_fused",
    )
    parser.add_argument("--analyze-branch-duplicates", action="store_true")
    parser.add_argument("--validate-layer", action="store_true")
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.validation_only and not args.validate_layer:
        raise SystemExit("--validation-only requires --validate-layer")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly {EXPECTED_VISIBLE}")
    if torch.cuda.device_count() != 2:
        raise SystemExit(f"expected two visible GPUs, got {torch.cuda.device_count()}")
    mp.spawn(run_rank, args=(free_port(), args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
