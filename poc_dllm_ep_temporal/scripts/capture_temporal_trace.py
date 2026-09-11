#!/usr/bin/env python3
"""Capture request/block/iteration/layer routing on dispatch-capable EP4.

Requests are deliberately replicated across all four dispatchers.  This keeps
the official diffusion loop in lockstep while producing one independent
per-request route trajectory at a time.  Batching policies are evaluated
offline at a fixed budget; this trace runner does not claim online scheduling.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import types

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "4,5,6,7"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TraceCollector:
    def __init__(self, rank, enabled):
        self.rank = rank
        self.enabled = enabled and rank == 0
        self.context = None
        self.current_layer = None
        self.routes = []
        self.stage_events = []
        self.iteration_events = []

    def set_context(self, **kwargs):
        self.context = kwargs

    def event_wrap(self, stage, function):
        def wrapped(*args, **kwargs):
            if not self.enabled or self.context is None:
                return function(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = function(*args, **kwargs)
            end.record()
            self.stage_events.append((dict(self.context), self.current_layer, stage, start, end))
            return result
        return wrapped

    def add_route(self, layer, counts, token_count, total_start, total_end):
        if self.enabled:
            self.routes.append((dict(self.context), layer, counts.detach(), token_count, total_start, total_end))

    def add_iteration(self, context, masked_before, masked_after, start, end):
        if self.enabled:
            self.iteration_events.append((dict(context), masked_before, masked_after, start, end))

    def flush_request(self, top_k, num_experts, ep_size, ownership_hash):
        if not self.enabled:
            return []
        torch.cuda.synchronize(self.rank)
        stage_by_key = {}
        for context, layer, stage, start, end in self.stage_events:
            key = (context["request_id"], context["block_id"], context["iteration_id"], layer)
            stage_by_key.setdefault(key, {})[stage] = float(start.elapsed_time(end))
        iteration_by_key = {}
        for context, before, after, start, end in self.iteration_events:
            key = (context["request_id"], context["block_id"], context["iteration_id"])
            iteration_by_key[key] = (before, after, float(start.elapsed_time(end)))

        records = []
        ownership = np.arange(num_experts) // (num_experts // ep_size)
        for context, layer, counts_gpu, token_count, total_start, total_end in self.routes:
            counts = counts_gpu.cpu().numpy().astype(int)
            ranks = np.array([counts[ownership == rank].sum() for rank in range(ep_size)], dtype=int)
            maximum = int(ranks.max())
            mean = float(ranks.mean())
            iteration = iteration_by_key[(context["request_id"], context["block_id"], context["iteration_id"])]
            stage = stage_by_key.get((context["request_id"], context["block_id"], context["iteration_id"], layer), {})
            records.append({
                **context,
                "layer_id": layer,
                "nfe": context["iteration_id"] + 1,
                "num_positions_total": token_count,
                "num_positions_active": token_count,
                "num_masked_before": iteration[0],
                "num_masked_after": iteration[1],
                "num_newly_accepted": iteration[0] - iteration[1],
                "top_k": top_k,
                "num_experts": num_experts,
                "ep_size": ep_size,
                "expert_assignment_counts": counts.tolist(),
                "rank_assignment_counts": ranks.tolist(),
                "max_rank_load": maximum,
                "mean_rank_load": mean,
                "max_over_mean": maximum / mean if mean else 0.0,
                "rank_load_cv": float(ranks.std() / mean) if mean else 0.0,
                "expert_to_rank_hash": ownership_hash,
                "dispatch_ms": stage.get("dispatch"),
                "expert_ms": stage.get("expert"),
                "combine_ms": stage.get("combine"),
                "moe_total_ms": float(total_start.elapsed_time(total_end)),
                "iteration_wall_ms": iteration[2],
            })
        self.routes.clear()
        self.stage_events.clear()
        self.iteration_events.clear()
        return records


def worker(rank, world, port, args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError("unsafe CUDA visibility")
    torch.cuda.set_device(rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE=str(world))

    from transformers import AutoConfig, AutoTokenizer
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.distributed import get_ep_group
    from dinfer import BlockIteratorFactory, ThresholdParallelDecoder
    from dinfer.decoding.generate_uniform import BlockWiseDiffusionLLM
    from dinfer.model import LLaDAMoeModelLM

    parallel = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=world,
        data_parallel_size_local=world,
        data_parallel_rank=rank,
        data_parallel_rank_local=rank,
        enable_expert_parallel=True,
        disable_custom_all_reduce=True,
    )
    vconfig = VllmConfig(parallel_config=parallel)
    # Hooks are installed before warmup, but collection itself is enabled only
    # after full diffusion/JIT warmup so compile and allocator costs cannot be
    # mistaken for temporal EP tails.
    collector = TraceCollector(rank, False)
    # See audit_ep_runtime.py: DP must not be installed while WORLD itself is
    # created, or vLLM interprets it as an external-DP deployment and expands
    # a four-process job into a virtual 16-rank world.
    distributed.init_distributed_environment(world, rank, "env://", rank, "nccl")
    with set_current_vllm_config(vconfig):
        distributed.initialize_model_parallel(1, backend="nccl")
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=config).eval()
        model.load_weights(args.model, torch_dtype=torch.bfloat16)
        model = model.to(torch.device(f"cuda:{rank}"))

        # DP-aware forward context is required by vLLM's dispatch/combine path.
        original_model_forward = model.forward
        def contextual_forward(self, input_ids=None, *positional, **kwargs):
            token_count = input_ids.numel() if input_ids is not None else kwargs["inputs_embeds"].shape[0] * kwargs["inputs_embeds"].shape[1]
            with set_forward_context(None, vconfig, num_tokens=token_count):
                return original_model_forward(input_ids, *positional, **kwargs)
        model.forward = types.MethodType(contextual_forward, model)

        manager = get_ep_group().device_communicator.all2all_manager
        if args.trace:
            manager.dispatch = collector.event_wrap("dispatch", manager.dispatch)
            manager.combine = collector.event_wrap("combine", manager.combine)

            for layer_id, layer in enumerate(model.model.layers):
                moe = layer.mlp
                experts = moe.experts
                experts.quant_method.apply = collector.event_wrap("expert", experts.quant_method.apply)

                def traced_moe_forward(self, hidden_states, _layer_id=layer_id):
                    original_shape = hidden_states.shape
                    hidden_dim = hidden_states.shape[-1]
                    flat = hidden_states.view(-1, hidden_dim)
                    dtype = flat.dtype
                    router_logits, _ = self.gate(flat.float())
                    top_ids = torch.topk(router_logits, k=self.top_k, dim=-1, sorted=False).indices
                    counts = torch.bincount(top_ids.reshape(-1), minlength=self.num_experts)
                    collector.current_layer = _layer_id
                    total_start, total_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    total_start.record()
                    output = self.experts.forward_impl(flat.to(dtype), router_logits)
                    total_end.record()
                    collector.add_route(_layer_id, counts, flat.shape[0], total_start, total_end)
                    return output.view(original_shape)
                moe.forward = types.MethodType(traced_moe_forward, moe)

        decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=156895, eos_id=156892)
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=None, early_stop=True)
        original_iteration_forward = dllm.diff_iteration.forward

        def traced_iteration(self, model_arg, decoder_arg, x, kv_cache, block, block_loc, block_id):
            context = {
                "run_id": args.run_id,
                "request_id": collector.context["request_id"],
                "batch_id": collector.context["batch_id"],
                "block_id": block_id,
                "iteration_id": self.iter_no,
                "prompt_length": collector.context["prompt_length"],
                "generation_budget": args.gen_length,
                "block_length": args.block_length,
            }
            collector.set_context(**context)
            before = int((block == decoder_arg.mask_id).sum().item())
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if collector.enabled:
                start.record()
            result = original_iteration_forward(model_arg, decoder_arg, x, kv_cache, block, block_loc, block_id)
            if collector.enabled:
                end.record()
                after = int((block == decoder_arg.mask_id).sum().item())
                collector.add_iteration(context, before, after, start, end)
            return result
        if args.trace:
            dllm.diff_iteration.forward = types.MethodType(traced_iteration, dllm.diff_iteration)

        def encode_prompt(prompt):
            formatted = "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>" + prompt + "<|role_end|><role>ASSISTANT</role>"
            ids = tokenizer(formatted, return_tensors="pt")["input_ids"][0]
            if args.fixed_prompt_length:
                # Repetition preserves prompt-specific lexical content while
                # enforcing one compile shape and a fixed batching budget.
                if ids.numel() < args.fixed_prompt_length:
                    repeats = (args.fixed_prompt_length + ids.numel() - 1) // ids.numel()
                    ids = ids.repeat(repeats)
                ids = ids[:args.fixed_prompt_length]
            return ids.unsqueeze(0).to(device=f"cuda:{rank}")

        all_prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
        warmup_prompts = all_prompts[:args.warmup_requests]
        for warmup_index, prompt in enumerate(warmup_prompts):
            input_ids = encode_prompt(prompt)
            dist.broadcast(input_ids, src=0)
            collector.set_context(request_id=f"warmup{warmup_index}", batch_id=-1,
                                  prompt_length=input_ids.shape[1])
            with torch.inference_mode():
                dllm.generate(input_ids, gen_length=args.gen_length,
                              block_length=args.block_length)
            torch.cuda.synchronize(rank)
            dist.barrier()

        collector.enabled = bool(args.trace and rank == 0)
        collector.routes.clear()
        collector.stage_events.clear()
        collector.iteration_events.clear()
        prompts = all_prompts[args.warmup_requests:
                              args.warmup_requests + args.num_requests]
        ownership_hash = hashlib.sha256(bytes(np.repeat(np.arange(4, dtype=np.uint8), config.num_experts // 4))).hexdigest()[:16]
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / f"rank{rank}_{'trace' if args.trace else 'clean'}.jsonl"
        timing_rows, outputs = [], []
        trace_stream = trace_path.open("w", encoding="utf-8") if collector.enabled else nullcontext(None)
        with trace_stream as stream:
            for request_index, prompt in enumerate(prompts):
                input_ids = encode_prompt(prompt)
                dist.broadcast(input_ids, src=0)
                collector.set_context(request_id=f"q{request_index:04d}", batch_id=request_index, prompt_length=input_ids.shape[1])
                torch.cuda.synchronize(rank)
                nfe_before = dllm.num_forwards
                start = time.perf_counter()
                generated = dllm.generate(input_ids, gen_length=args.gen_length, block_length=args.block_length)
                torch.cuda.synchronize(rank)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                timing_rows.append({"request_id": f"q{request_index:04d}", "rank": rank, "elapsed_ms": elapsed_ms, "nfe": dllm.num_forwards - nfe_before})
                if rank == 0:
                    outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
                    for record in collector.flush_request(config.num_experts_per_tok, config.num_experts, 4, ownership_hash):
                        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                dist.barrier()

        (output_dir / f"rank{rank}_{'trace' if args.trace else 'clean'}_timing.json").write_text(json.dumps(timing_rows, indent=2), encoding="utf-8")
        if rank == 0:
            (output_dir / f"outputs_{'trace' if args.trace else 'clean'}.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        dist.barrier()
        distributed.destroy_model_parallel()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--fixed-prompt-length", type=int, default=64)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly {EXPECTED_VISIBLE}")
    if torch.cuda.device_count() != 4:
        raise SystemExit("expected exactly four visible GPUs")
    mp.spawn(worker, args=(4, free_port(), args), nprocs=4, join=True)


if __name__ == "__main__":
    main()
