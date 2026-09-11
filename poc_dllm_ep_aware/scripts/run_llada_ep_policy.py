#!/usr/bin/env python3
"""Reference assignment-level EP runner for LLaDA-MoE policies.

This is intentionally a mechanism-screening substrate, not a production
backend.  Rank zero makes the semantic routing decision, token/expert rows are
sent once to their owning rank, local fused experts execute, and weighted
updates return to rank zero.  The completed MoE tensor is broadcast so the
replicated attention path remains in lockstep.

Policies:
  vanilla  - native fixed top-8 routing.
  reflex   - paper-faithful RRB + FGER expert-count allocation.
  des_vote - paper-faithful sequence saliency vote and constrained top-8.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import types

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


VISIBLE = {1: "0", 2: "0,1", 4: "0,1,2,3"}
UUIDS = {
    0: "f217c8a0-1142-20f4-d84b-af29f3a47a0d",
    1: "a77f3471-67d4-20b0-9fab-e502d4de5adb",
    2: "24200107-8a7f-de46-1bc8-b81f8d3af13e",
    3: "17488c15-2d4c-5d9e-d503-29b0d959a8a8",
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ReferenceEP:
    def __init__(self, rank: int, world: int, policy: str, beta: float,
                 instrument: bool, mask_id: int):
        self.rank = rank
        self.world = world
        self.policy = policy
        self.beta = beta
        self.instrument = instrument
        self.mask_id = mask_id
        self.context: dict = {}
        self.token_ids = None
        self.confidence_history: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
        self.rows = []
        self.events = []
        self.call_id = 0

    def set_request(self, request_id: str, prompt_length: int) -> None:
        self.context = {"request_id": request_id, "prompt_length": prompt_length,
                        "block_id": -1, "iteration_id": -1}
        self.confidence_history.clear()

    def set_iteration(self, block_id: int, iteration_id: int, token_ids: torch.Tensor) -> None:
        self.context.update(block_id=int(block_id), iteration_id=int(iteration_id))
        self.token_ids = token_ids.detach()

    def add_confidence(self, block_id: int, logits: torch.Tensor) -> None:
        if self.rank != 0:
            return
        confidence = torch.softmax(logits.float(), dim=-1).amax(dim=-1).detach().flatten()
        key = (self.context["request_id"], int(block_id))
        self.confidence_history[key].append(confidence)
        self.confidence_history[key] = self.confidence_history[key][-2:]

    def install(self, model) -> dict:
        ownership = {}
        for layer_id, layer in enumerate(model.model.layers):
            block = layer.mlp
            block.forward = types.MethodType(self._forward(layer_id), block)
            if block.experts.expert_map is None:
                local = list(range(block.num_experts))
            else:
                local = torch.where(block.experts.expert_map >= 0)[0].tolist()
            ownership[layer_id] = [int(x) for x in local]
        return {
            "physical_backend": "reference_assignment_a2a_single_source",
            "world_size": self.world,
            "rank": self.rank,
            "num_experts": int(model.config.num_experts),
            "experts_per_rank": int(model.config.num_experts // self.world),
            "owned_experts": ownership,
            "policy": self.policy,
        }

    def _forward(self, layer_id: int):
        runtime = self

        def forward(block, hidden_states):
            return runtime.execute(layer_id, block, hidden_states)

        return forward

    def _reflex_counts(self, rows: int, device: torch.device) -> torch.Tensor:
        prompt = int(self.context["prompt_length"])
        active = int(self.context["block_id"]) + 1
        block_len = int(self.context.get("block_length", 32))
        positions = torch.arange(rows, device=device)
        block_index = torch.where(
            positions < prompt,
            torch.zeros_like(positions),
            1 + torch.div((positions - prompt).clamp_min(0), block_len,
                          rounding_mode="floor"),
        )
        distance = (block_index - active).abs()
        counts = torch.where(distance <= 1, 12,
                             torch.where(distance == 2, 8, 4)).to(torch.long)

        if self.token_ids is None or self.token_ids.numel() != rows:
            return counts
        unresolved = (self.token_ids.reshape(-1) == self.mask_id) & (block_index == active)
        indices = torch.where(unresolved)[0]
        if not indices.numel():
            return counts
        history = self.confidence_history.get(
            (self.context["request_id"], int(self.context["block_id"])), [])
        if not history:
            return counts
        c1 = history[-1]
        # Decoder logits cover only the active 32-token block.
        active_offset = positions[indices] - (prompt + (active - 1) * block_len)
        valid = (active_offset >= 0) & (active_offset < c1.numel())
        indices, active_offset = indices[valid], active_offset[valid]
        if not indices.numel():
            return counts
        c1 = c1.index_select(0, active_offset)
        if len(history) >= 2:
            c2 = history[-2].index_select(0, active_offset)
            velocity = (c1 - c2).clamp(-0.15, 0.15)
            forecast = (c1 + velocity).clamp(0, 1)
            progress = torch.sigmoid((0.01 - velocity) / 0.05)
        else:
            forecast = c1
            progress = torch.ones_like(c1)
        frontier = torch.sigmoid((forecast - 0.9 + 0.30) / 0.05)
        frontier *= torch.sigmoid((0.9 - forecast) / 0.05)
        score = frontier * progress
        order = torch.argsort(score, descending=True, stable=True)
        outer = min(int(math.floor(indices.numel() * 0.25)), indices.numel() // 2)
        if outer:
            counts[indices[order[:outer]]] = 14
            counts[indices[order[-outer:]]] = 10
        return counts

    def _select(self, router_logits: torch.Tensor, block) -> tuple[torch.Tensor, ...]:
        probs = torch.softmax(router_logits.float(), dim=-1)
        num_rows, num_experts = probs.shape
        coreset = None
        renorm_rows = None
        if self.policy == "vanilla":
            counts = torch.full((num_rows,), 8, device=probs.device, dtype=torch.long)
        elif self.policy == "reflex":
            counts = self._reflex_counts(num_rows, probs.device)
        elif self.policy == "des_vote":
            # DES forms its consensus over the parallel decoding block.  This
            # no-cache reference forward also contains prompt and future mask
            # positions, so applying one coreset to every row would be an
            # unfaithful port (future masks dominate the vote).  Constrain only
            # the active 32-token block, which is the MoE working set in the
            # paper's Fast-dLLM substrate; leave context rows on native top-8.
            prompt = int(self.context["prompt_length"])
            active = int(self.context["block_id"])
            begin = prompt + active * int(self.context.get("block_length", 32))
            target = torch.zeros(num_rows, device=probs.device, dtype=torch.bool)
            target[max(0, begin):min(num_rows, begin + int(self.context.get("block_length", 32)))] = True
            vote_probs = probs[target] if target.any() else probs
            local_weights, local_ids = torch.topk(vote_probs, 8, dim=-1)
            votes = torch.zeros(num_experts, device=probs.device, dtype=probs.dtype)
            votes.scatter_add_(0, local_ids.reshape(-1), local_weights.reshape(-1))
            core_size = max(8, min(num_experts, int(round(self.beta * num_experts))))
            coreset = torch.topk(votes, core_size, sorted=False).indices
            allowed = torch.zeros(num_experts, device=probs.device, dtype=torch.bool)
            allowed[coreset] = True
            probs = torch.where(target.unsqueeze(1),
                                probs.masked_fill(~allowed.unsqueeze(0), 0), probs)
            renorm_rows = target
            counts = torch.full((num_rows,), 8, device=probs.device, dtype=torch.long)
        else:
            raise ValueError(self.policy)

        maximum = int(counts.max().item())
        weights, experts = torch.topk(probs, maximum, dim=-1, sorted=True)
        keep = torch.arange(maximum, device=probs.device).unsqueeze(0) < counts.unsqueeze(1)
        token_ids = torch.arange(num_rows, device=probs.device).unsqueeze(1).expand_as(experts)[keep]
        expert_ids = experts[keep]
        weights = weights[keep]
        # REFLEX and DES explicitly renormalize the selected weights.
        if self.policy != "vanilla":
            sums = torch.zeros(num_rows, device=probs.device, dtype=weights.dtype)
            sums.index_add_(0, token_ids, weights)
            normalized = weights / sums.index_select(0, token_ids).clamp_min(1e-12)
            if renorm_rows is None:
                weights = normalized
            else:
                weights = torch.where(renorm_rows.index_select(0, token_ids),
                                      normalized, weights)
        return token_ids, expert_ids, weights, counts, coreset

    @staticmethod
    def _pair():
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        return a, b

    def execute(self, layer_id, block, hidden_states):
        shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, shape[-1])
        rank, world, device = self.rank, self.world, hidden.device
        event_pairs = {}
        if self.instrument:
            event_pairs["moe"] = self._pair()
            event_pairs["router"] = self._pair()

        if rank == 0:
            router_logits, _ = block.gate(hidden.float())
        else:
            router_logits = torch.empty((0, block.num_experts), device=device,
                                        dtype=torch.float32)
        if self.instrument:
            event_pairs["router"][1].record()
            event_pairs["prepare"] = self._pair()
        if rank == 0:
            token_ids, expert_ids, weights, k_counts, coreset = self._select(
                router_logits, block)
            owners = torch.div(expert_ids, block.num_experts // world,
                               rounding_mode="floor")
            order = torch.argsort(owners, stable=True)
            send_hidden = hidden.index_select(0, token_ids[order]).contiguous()
            send_expert = expert_ids[order].contiguous()
            send_token = token_ids[order].contiguous()
            send_weight = weights[order].to(hidden.dtype).contiguous()
            counts = torch.bincount(owners, minlength=world).to(torch.int64)
            selected_unique = int(torch.unique(expert_ids).numel())
            avg_k = float(k_counts.float().mean().item())
            coreset_size = int(coreset.numel()) if coreset is not None else None
        else:
            send_hidden = torch.empty((0, hidden.shape[-1]), device=device, dtype=hidden.dtype)
            send_expert = torch.empty(0, device=device, dtype=torch.long)
            send_token = torch.empty(0, device=device, dtype=torch.long)
            send_weight = torch.empty(0, device=device, dtype=hidden.dtype)
            counts = torch.zeros(world, device=device, dtype=torch.int64)
            selected_unique, avg_k, coreset_size = 0, 0.0, None
        dist.broadcast(counts, src=0)
        count_list = [int(x) for x in counts.cpu().tolist()]
        if self.instrument:
            event_pairs["prepare"][1].record()
            event_pairs["dispatch"] = self._pair()
        input_splits = count_list if rank == 0 else [0] * world
        output_splits = [0] * world
        output_splits[0] = count_list[rank]
        recv_rows = count_list[rank]
        recv_hidden = torch.empty((recv_rows, hidden.shape[-1]), device=device, dtype=hidden.dtype)
        recv_expert = torch.empty(recv_rows, device=device, dtype=torch.long)
        recv_token = torch.empty(recv_rows, device=device, dtype=torch.long)
        recv_weight = torch.empty(recv_rows, device=device, dtype=hidden.dtype)
        for output, source in ((recv_hidden, send_hidden), (recv_expert, send_expert),
                               (recv_token, send_token), (recv_weight, send_weight)):
            dist.all_to_all_single(output, source, output_split_sizes=output_splits,
                                   input_split_sizes=input_splits)
        if self.instrument:
            event_pairs["dispatch"][1].record()
            event_pairs["expert"] = self._pair()

        from vllm.model_executor.layers.fused_moe import fused_experts
        local_begin = rank * (block.num_experts // world)
        if recv_rows:
            local_output = fused_experts(
                hidden_states=recv_hidden,
                w1=block.experts.w13_weight,
                w2=block.experts.w2_weight,
                topk_weights=recv_weight.unsqueeze(-1),
                topk_ids=(recv_expert - local_begin).unsqueeze(-1),
                inplace=False,
                activation="silu",
                is_act_and_mul=True,
            )
        else:
            local_output = torch.empty_like(recv_hidden)
        if self.instrument:
            event_pairs["expert"][1].record()
            event_pairs["combine"] = self._pair()
        reverse_input = [0] * world
        reverse_input[0] = recv_rows
        reverse_output = count_list if rank == 0 else [0] * world
        returned = torch.empty((sum(count_list) if rank == 0 else 0, hidden.shape[-1]),
                               device=device, dtype=hidden.dtype)
        dist.all_to_all_single(returned, local_output,
                               output_split_sizes=reverse_output,
                               input_split_sizes=reverse_input)
        final = torch.zeros_like(hidden)
        if rank == 0 and returned.numel():
            final.index_add_(0, send_token, returned)
        dist.broadcast(final, src=0)
        if self.instrument:
            event_pairs["combine"][1].record()
            event_pairs["moe"][1].record()

        row = {
            **self.context,
            "call_id": self.call_id,
            "layer_id": layer_id,
            "rank": rank,
            "policy": self.policy,
            "physical_rows": int(hidden.shape[0]),
            "assignments": int(sum(count_list)),
            "avg_k": avg_k if rank == 0 else None,
            "unique_experts": selected_unique if rank == 0 else None,
            "coreset_size": coreset_size,
            "rank_assignment_counts": count_list,
            "critical_rank_assignments": max(count_list),
            "remote_assignments": int(sum(count_list[1:])),
            "remote_fraction": (sum(count_list[1:]) / sum(count_list)) if sum(count_list) else 0,
            "destination_rank_fanout": int(sum(x > 0 for x in count_list)),
            "remote_fanout": int(sum(x > 0 for x in count_list[1:])),
            "dispatch_bytes": int(sum(count_list) * hidden.shape[-1] * hidden.element_size()),
            "remote_dispatch_bytes": int(sum(count_list[1:]) * hidden.shape[-1] * hidden.element_size()),
            "combine_bytes": int(sum(count_list) * hidden.shape[-1] * hidden.element_size()),
            "local_active_experts": int(torch.unique(recv_expert).numel()),
        }
        self.call_id += 1
        if self.instrument:
            self.events.append((row, event_pairs))
        else:
            self.rows.append(row)
        return final.reshape(shape)

    def finish(self):
        torch.cuda.synchronize(self.rank)
        for row, pairs in self.events:
            for name, (start, end) in pairs.items():
                row[f"{name}_ms"] = float(start.elapsed_time(end))
            self.rows.append(row)
        self.events.clear()
        return self.rows

    def reset(self):
        torch.cuda.synchronize(self.rank)
        self.rows.clear()
        self.events.clear()
        self.call_id = 0


def worker(rank: int, world: int, port: int, args) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE[world]:
        raise RuntimeError("unsafe CUDA visibility")
    torch.cuda.set_device(rank)
    props = torch.cuda.get_device_properties(rank)
    if str(props.uuid) != UUIDS[rank]:
        raise RuntimeError(f"GPU UUID mismatch on logical {rank}: {props.uuid}")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE=str(world))

    from transformers import AutoConfig, AutoTokenizer
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from dinfer import BlockIteratorFactory, ThresholdParallelDecoder
    from dinfer.decoding.generate_uniform import BlockWiseDiffusionLLM
    from dinfer.model import LLaDAMoeModelLM

    parallel = ParallelConfig(tensor_parallel_size=1, data_parallel_size=world,
                              data_parallel_size_local=world, data_parallel_rank=rank,
                              data_parallel_rank_local=rank, enable_expert_parallel=True,
                              disable_custom_all_reduce=True)
    vconfig = VllmConfig(parallel_config=parallel)
    distributed.init_distributed_environment(world, rank, "env://", rank, "nccl")
    with set_current_vllm_config(vconfig):
        distributed.initialize_model_parallel(1, backend="nccl")
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=config).eval()
        model.load_weights(args.model, torch_dtype=torch.bfloat16)
        model = model.to(torch.device(f"cuda:{rank}"))
        runtime = ReferenceEP(rank, world, args.policy, args.des_beta, args.trace,
                              tokenizer.mask_token_id)
        audit = runtime.install(model)

        original_forward = model.forward
        def contextual_forward(self, input_ids=None, *positional, **kwargs):
            if input_ids is not None:
                runtime.token_ids = input_ids.detach()
                num_tokens = input_ids.numel()
            else:
                value = kwargs["inputs_embeds"]
                num_tokens = value.shape[0] * value.shape[1]
            with set_forward_context(None, vconfig, num_tokens=num_tokens):
                return original_forward(input_ids, *positional, **kwargs)
        model.forward = types.MethodType(contextual_forward, model)

        decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold,
                                           mask_id=tokenizer.mask_token_id,
                                           eos_id=tokenizer.eos_token_id)
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True),
                                    cache_factory=None, early_stop=True)
        original_iteration = dllm.diff_iteration.forward
        def wrapped_iteration(self, model_arg, decoder_arg, x, kv_cache, block,
                              block_loc, block_id):
            runtime.context["block_length"] = args.block_length
            runtime.set_iteration(block_id, self.iter_no, x.data)
            result = original_iteration(model_arg, decoder_arg, x, kv_cache,
                                        block, block_loc, block_id)
            logits = result[-1] if isinstance(result, tuple) else result.logits
            runtime.add_confidence(block_id, logits)
            return result
        dllm.diff_iteration.forward = types.MethodType(wrapped_iteration, dllm.diff_iteration)

        def encode(text):
            formatted = ("<role>SYSTEM</role>detailed thinking off<|role_end|>"
                         "<role>HUMAN</role>" + text +
                         "<|role_end|><role>ASSISTANT</role>")
            ids = tokenizer(formatted, return_tensors="pt")["input_ids"][0]
            if args.fixed_prompt_length:
                if ids.numel() < args.fixed_prompt_length:
                    ids = ids.repeat((args.fixed_prompt_length + ids.numel() - 1) // ids.numel())
                ids = ids[:args.fixed_prompt_length]
            return ids.unsqueeze(0).to(f"cuda:{rank}")

        prompt_rows = json.loads(Path(args.prompts).read_text())
        prompts = [row if isinstance(row, str) else row["prompt"] for row in prompt_rows]
        metadata = [({"dataset": "synthetic", "id": f"prompt_{i}"}
                     if isinstance(row, str) else row) for i, row in enumerate(prompt_rows)]
        for i in range(args.warmup_requests):
            ids = encode(prompts[i % len(prompts)])
            dist.broadcast(ids, src=0)
            runtime.set_request(f"warmup{i}", ids.shape[1])
            with torch.inference_mode():
                dllm.generate(ids, gen_length=args.gen_length, block_length=args.block_length)
            torch.cuda.synchronize(rank); dist.barrier()
        runtime.reset()

        output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
        timings, outputs = [], []
        for i in range(args.num_requests):
            prompt_index = (args.warmup_requests + i) % len(prompts)
            text = prompts[prompt_index]
            ids = encode(text); dist.broadcast(ids, src=0)
            request_id = f"q{i:04d}"
            runtime.set_request(request_id, ids.shape[1])
            torch.cuda.synchronize(rank); dist.barrier()
            before = dllm.num_forwards
            start = time.perf_counter()
            with torch.inference_mode():
                generated = dllm.generate(ids, gen_length=args.gen_length,
                                          block_length=args.block_length)
            torch.cuda.synchronize(rank)
            elapsed = (time.perf_counter() - start) * 1000
            timings.append({"request_id": request_id, "rank": rank,
                            "elapsed_ms": elapsed, "nfe": dllm.num_forwards - before})
            if rank == 0:
                outputs.append({
                    "request_id": request_id,
                    "source_id": metadata[prompt_index].get("id"),
                    "dataset": metadata[prompt_index].get("dataset"),
                    "reference": metadata[prompt_index].get("reference"),
                    "prompt": text,
                    "generated": tokenizer.decode(
                        generated[0, ids.shape[1]:], skip_special_tokens=True),
                })
            dist.barrier()
        rows = runtime.finish()
        (output_dir / f"rank{rank}_timing.json").write_text(json.dumps(timings, indent=2))
        (output_dir / f"rank{rank}_trace.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        (output_dir / f"rank{rank}_audit.json").write_text(json.dumps(audit, indent=2))
        if rank == 0:
            (output_dir / "outputs.json").write_text(json.dumps(outputs, indent=2))
        dist.barrier()
        distributed.destroy_model_parallel(); dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=("vanilla", "reflex", "des_vote"), required=True)
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--fixed-prompt-length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--des-beta", type=float, default=0.6)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE[args.world_size]:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE[args.world_size]}")
    if torch.cuda.device_count() != args.world_size:
        raise SystemExit("visible GPU count mismatch")
    mp.spawn(worker, args=(args.world_size, free_port(), args),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
