#!/usr/bin/env python3
"""True-EP2 audit, fixed-work scaling, and real dInfer trajectory capture.

The process topology is TP1/DP2/EP2.  Each DP rank is an EP dispatcher; expert
weights are disjoint and the all-to-all manager exchanges activations/results.
The runner is intentionally pinned to physical GPUs 6 and 7.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
import types
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "6,7"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class EventRecorder:
    """Same-device CUDA-event measurements for exactly one logical call."""

    def __init__(self, rank: int):
        self.rank = rank
        self.enabled = False
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def wrap(self, name: str, function):
        def timed(*args, **kwargs):
            if not self.enabled:
                return function(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            value = function(*args, **kwargs)
            end.record()
            self.events.append((name, start, end))
            return value
        return timed

    def reset(self) -> None:
        self.events.clear()

    def finish(self) -> dict[str, float]:
        torch.cuda.synchronize(self.rank)
        values: dict[str, float] = {}
        for name, start, end in self.events:
            values[name] = values.get(name, 0.0) + float(start.elapsed_time(end))
        return values


def install_stage_hooks(model, manager, recorder: EventRecorder, layer_id: int = 0) -> dict[str, Any]:
    """Instrument naive manager calls or modular PPLX/DeepEP boundaries."""
    layer = model.model.layers[layer_id]
    experts = layer.mlp.experts
    quant = experts.quant_method
    backend_path: dict[str, Any] = {
        "manager_class": type(manager).__name__,
        "quant_method": type(quant).__name__,
        "modular": False,
    }

    modular = getattr(quant, "fused_experts", None)
    if modular is not None and hasattr(modular, "prepare_finalize"):
        backend_path.update(
            modular=True,
            modular_class=type(modular).__name__,
            prepare_finalize_class=type(modular.prepare_finalize).__name__,
            expert_impl_class=type(modular.fused_experts).__name__,
        )
        modular.prepare_finalize.prepare = recorder.wrap(
            "dispatch", modular.prepare_finalize.prepare)
        modular.prepare_finalize.finalize = recorder.wrap(
            "combine", modular.prepare_finalize.finalize)
        modular.fused_experts.apply = recorder.wrap(
            "expert", modular.fused_experts.apply)
    else:
        manager.dispatch = recorder.wrap("dispatch", manager.dispatch)
        manager.combine = recorder.wrap("combine", manager.combine)
        quant.apply = recorder.wrap("expert", quant.apply)
    return backend_path


def initialize(rank: int, world: int, port: int, args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError("unsafe CUDA visibility")
    if world != 2:
        raise RuntimeError("this runner is EP2-only")
    torch.cuda.set_device(rank)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        LOCAL_RANK=str(rank),
        RANK=str(rank),
        WORLD_SIZE=str(world),
    )

    from transformers import AutoConfig, AutoTokenizer
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import get_ep_group
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

    # As in the verified EP4 substrate, construct the physical WORLD before
    # installing the DP-aware config, avoiding vLLM external-DP rank expansion.
    distributed.init_distributed_environment(world, rank, "env://", rank, "nccl")
    config_guard = set_current_vllm_config(vconfig)
    config_guard.__enter__()
    distributed.initialize_model_parallel(1, backend="nccl")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = LLaDAMoeModelLM(config=config).eval()
    model.load_weights(args.model, torch_dtype=torch.bfloat16)
    model = model.to(torch.device(f"cuda:{rank}"))

    if args.backend != "naive":
        # Stock dInfer bypasses vLLM's GPU worker, which normally performs this
        # modular-kernel initialization after model loading.
        from vllm.distributed import prepare_communication_buffer_for_model
        prepare_communication_buffer_for_model(model)

    manager = get_ep_group().device_communicator.all2all_manager
    return distributed, config_guard, vconfig, config, tokenizer, model, manager


def forward_context(vconfig, token_count: int):
    from vllm.forward_context import set_forward_context
    return set_forward_context(None, vconfig, num_tokens=token_count)


def write_rank_json(output: Path, stem: str, rank: int, payload: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{stem}_rank{rank}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_audit(rank: int, world: int, port: int, args) -> None:
    distributed, guard, vconfig, config, _, model, manager = initialize(rank, world, port, args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(rank)
    backend_path = install_stage_hooks(model, manager, recorder)
    experts = model.model.layers[0].mlp.experts
    expert_map = experts.expert_map.detach().cpu()
    local_ids = torch.where(expert_map >= 0)[0].tolist()

    generator = torch.Generator(device=f"cuda:{rank}")
    generator.manual_seed(260911)
    hidden = torch.randn((args.audit_tokens, config.hidden_size), generator=generator,
                         device=f"cuda:{rank}", dtype=torch.bfloat16)
    dist.broadcast(hidden, src=0)

    with torch.inference_mode(), forward_context(vconfig, args.audit_tokens):
        model.model.layers[0].mlp(hidden)
    torch.cuda.synchronize(rank)
    recorder.reset()
    recorder.enabled = True
    total_start, total_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    profiler = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) if args.profile else nullcontext())
    wall_start = time.perf_counter()
    with profiler, torch.inference_mode(), forward_context(vconfig, args.audit_tokens):
        with torch.profiler.record_function("ep2_audit_moe"):
            total_start.record()
            result = model.model.layers[0].mlp(hidden)
            total_end.record()
    stage = recorder.finish()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    moe_ms = float(total_start.elapsed_time(total_end))

    gathered = [torch.empty_like(result) for _ in range(world)]
    dist.all_gather(gathered, result)
    max_abs = max(float((item - gathered[0]).abs().max()) for item in gathered)
    rel_l2 = max(float(torch.linalg.vector_norm((item - gathered[0]).float()) /
                       torch.linalg.vector_norm(gathered[0].float()).clamp_min(1e-12))
                 for item in gathered)

    local_weight_bytes = int(sum(t.numel() * t.element_size()
                                 for t in experts.get_expert_weights()))
    profiler_events = []
    if args.profile:
        profiler.export_chrome_trace(str(output / f"audit_{args.backend}_rank{rank}_trace.json"))
        for event in profiler.key_averages():
            device_us = float(getattr(event, "device_time_total", 0.0))
            if device_us > 0 or "nccl" in event.key.lower() or "broadcast" in event.key.lower():
                profiler_events.append({
                    "key": event.key,
                    "device_total_us": device_us,
                    "cpu_total_us": float(getattr(event, "cpu_time_total", 0.0)),
                    "count": int(event.count),
                })
        profiler_events.sort(key=lambda row: row["device_total_us"], reverse=True)

    audit = {
        "rank": rank,
        "physical_gpu": rank + 6,
        "backend": args.backend,
        "manager_class": type(manager).__name__,
        "backend_path": backend_path,
        "tp_size": experts.tp_size,
        "dp_size": experts.dp_size,
        "ep_size": experts.ep_size,
        "ep_rank": experts.ep_rank,
        "num_experts": config.num_experts,
        "local_expert_count": experts.local_num_experts,
        "local_expert_ids": local_ids,
        "local_expert_weight_bytes": local_weight_bytes,
        "one_expert_weight_bytes": local_weight_bytes // experts.local_num_experts,
        "expert_map_sha256": hashlib.sha256(expert_map.numpy().tobytes()).hexdigest(),
        "audit_tokens_per_dispatcher": args.audit_tokens,
        "stage_ms": stage,
        "moe_ms": moe_ms,
        "wall_ms": wall_ms,
        "cross_rank_output_max_abs": max_abs,
        "cross_rank_output_rel_l2": rel_l2,
        "profiler_events": profiler_events[:100],
    }
    write_rank_json(output, f"audit_{args.backend}", rank, audit)
    if rank == 0:
        torch.save(result.detach().cpu(), output / f"audit_{args.backend}_output.pt")
    dist.barrier()
    distributed.destroy_model_parallel()
    dist.destroy_process_group()
    guard.__exit__(None, None, None)


def run_scaling(rank: int, world: int, port: int, args) -> None:
    distributed, guard, vconfig, config, _, model, manager = initialize(rank, world, port, args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(rank)
    backend_path = install_stage_hooks(model, manager, recorder)
    moe = model.model.layers[0].mlp
    sizes = [int(item) for item in args.active_sizes.split(",") if item]
    batches = [int(item) for item in args.batch_sizes.split(",") if item]
    shapes = [(active, batch) for active in sizes for batch in batches
              if active * batch <= args.max_local_tokens]
    order = shapes.copy()
    random.Random(args.seed).shuffle(order)
    rows: list[dict[str, Any]] = []

    def make_hidden(active: int, batch: int):
        generator = torch.Generator(device=f"cuda:{rank}")
        # Flattened MoE semantics depend on local M, not the caller's
        # active-position/batch factorization.  Equal-M controls therefore use
        # exactly the same hidden states and routing.
        generator.manual_seed(args.seed + 1009 * (active * batch) + rank)
        return torch.randn((active * batch, config.hidden_size), generator=generator,
                           device=f"cuda:{rank}", dtype=torch.bfloat16)

    for active, batch in order:
        local_m = active * batch
        hidden = make_hidden(active, batch)
        with torch.inference_mode():
            gate_logits, _ = moe.gate(hidden.float())
            top_ids = torch.topk(gate_logits, k=moe.top_k, dim=-1, sorted=False).indices
            local_counts = torch.bincount(top_ids.reshape(-1), minlength=config.num_experts)
        global_counts = local_counts.clone()
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)

        for _ in range(args.warmup):
            with torch.inference_mode(), forward_context(vconfig, local_m):
                moe(hidden)
        torch.cuda.synchronize(rank)
        dist.barrier()

        for rep in range(args.repetitions):
            recorder.reset()
            recorder.enabled = True
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(rank)
            dist.barrier()
            wall_start = time.perf_counter()
            with torch.inference_mode(), forward_context(vconfig, local_m):
                total_start.record()
                result = moe(hidden)
                total_end.record()
            stages = recorder.finish()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            rows.append({
                "rank": rank,
                "physical_gpu": rank + 6,
                "backend": args.backend,
                "active_positions": active,
                "batch_size": batch,
                "local_m": local_m,
                "global_m": local_m * world,
                "top_k": moe.top_k,
                "routed_assignments": int(global_counts.sum().item()),
                "max_expert_assignments": int(global_counts.max().item()),
                "rep": rep,
                "dispatch_ms": stages.get("dispatch"),
                "expert_ms": stages.get("expert"),
                "combine_ms": stages.get("combine"),
                "moe_ms": float(total_start.elapsed_time(total_end)),
                "wall_ms": wall_ms,
                "output_sum": float(result.float().sum().item()),
                "backend_path": backend_path,
            })
        recorder.enabled = False

    with (output / f"scaling_{args.backend}_rank{rank}.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    dist.barrier()
    distributed.destroy_model_parallel()
    dist.destroy_process_group()
    guard.__exit__(None, None, None)


def run_trajectory(rank: int, world: int, port: int, args) -> None:
    distributed, guard, vconfig, config, tokenizer, model, manager = initialize(rank, world, port, args)
    from dinfer import BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder
    from dinfer.decoding.generate_uniform import BlockWiseDiffusionLLM

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(rank)
    if args.trace:
        first_path = install_stage_hooks(model, manager, recorder)
        if first_path["modular"]:
            for layer_id in range(1, len(model.model.layers)):
                install_stage_hooks(model, manager, recorder, layer_id=layer_id)
        else:
            # Naive dispatch/combine are manager-wide and must be wrapped once;
            # expert kernels are layer-local.
            for layer in model.model.layers[1:]:
                quant = layer.mlp.experts.quant_method
                quant.apply = recorder.wrap("expert", quant.apply)

    cache_factory = (KVCacheFactory(args.cache, is_bd_model=False)
                     if args.cache in ("prefix", "dual") else None)
    eos_id = -1 if args.disable_eos else 156892
    decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold,
                                       mask_id=156895, eos_id=eos_id)
    dllm = BlockWiseDiffusionLLM(model, decoder,
                                 BlockIteratorFactory(start_block_align=True),
                                 cache_factory=cache_factory,
                                 early_stop=not args.disable_early_stop)

    current: dict[str, Any] = {}
    route_rows: list[dict[str, Any]] = []
    original_model_forward = model.forward

    def contextual_forward(self, input_ids=None, *positional, **kwargs):
        if input_ids is not None:
            token_count = int(input_ids.numel())
        else:
            embeds = kwargs["inputs_embeds"]
            token_count = int(embeds.shape[0] * embeds.shape[1])
        current["model_positions"] = token_count
        current["model_calls"] = current.get("model_calls", 0) + 1
        with forward_context(vconfig, token_count):
            return original_model_forward(input_ids, *positional, **kwargs)
    model.forward = types.MethodType(contextual_forward, model)

    if args.trace:
        for layer_id, layer in enumerate(model.model.layers):
            original_moe = layer.mlp.forward

            def traced_moe(self, hidden_states, _orig=original_moe, _layer=layer_id):
                flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                with torch.inference_mode():
                    logits, _ = self.gate(flat.float())
                    ids = torch.topk(logits, k=self.top_k, dim=-1, sorted=False).indices
                    counts = torch.bincount(ids.reshape(-1), minlength=self.num_experts)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = _orig(hidden_states)
                end.record()
                route_rows.append({
                    **current,
                    "layer_id": _layer,
                    "moe_positions": int(flat.shape[0]),
                    "expert_counts_gpu": counts,
                    "moe_start": start,
                    "moe_end": end,
                })
                return result
            layer.mlp.forward = types.MethodType(traced_moe, layer.mlp)

    original_iteration = dllm.diff_iteration.forward
    iteration_rows: list[dict[str, Any]] = []

    def traced_iteration(self, model_arg, decoder_arg, x, kv_cache, block, block_loc, block_id):
        before = int((block == decoder_arg.mask_id).sum().item())
        current.clear()
        current.update(request_id=args.run_id, block_id=int(block_id),
                       iteration_id=int(self.iter_no), masked_before=before,
                       total_block_positions=int(block.numel()), model_calls=0)
        wall_start = time.perf_counter()
        result = original_iteration(model_arg, decoder_arg, x, kv_cache, block, block_loc, block_id)
        torch.cuda.synchronize(rank)
        after = int((block == decoder_arg.mask_id).sum().item())
        iteration_rows.append({
            **{k: v for k, v in current.items() if k != "expert_counts_gpu"},
            "masked_after": after,
            "newly_accepted": before - after,
            "iteration_wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        })
        return result
    if args.trace:
        dllm.diff_iteration.forward = types.MethodType(traced_iteration, dllm.diff_iteration)

    def encode(prompt: str):
        formatted = ("<role>SYSTEM</role>detailed thinking off<|role_end|>"
                     "<role>HUMAN</role>" + prompt +
                     "<|role_end|><role>ASSISTANT</role>")
        ids = tokenizer(formatted, return_tensors="pt")["input_ids"][0]
        if ids.numel() < args.prompt_length:
            ids = ids.repeat((args.prompt_length + ids.numel() - 1) // ids.numel())
        return ids[:args.prompt_length]

    prompts = [
        "Explain why the sky is blue in one concise paragraph.",
        "Solve 17 times 23 and briefly show the arithmetic.",
        "Write a short summary of expert parallel inference.",
        "List three causes of ocean tides.",
    ]
    batch_ids = torch.stack([encode(prompts[i % len(prompts)])
                             for i in range(args.batch_size)]).to(f"cuda:{rank}")
    dist.broadcast(batch_ids, src=0)

    for _ in range(args.warmup_requests):
        recorder.enabled = False
        with torch.inference_mode():
            dllm.generate(batch_ids, gen_length=args.gen_length,
                          block_length=args.block_length)
        torch.cuda.synchronize(rank)
        dist.barrier()

    route_rows.clear()
    iteration_rows.clear()
    recorder.reset()
    recorder.enabled = bool(args.trace)
    torch.cuda.synchronize(rank)
    dist.barrier()
    nfe_before = dllm.num_forwards
    request_start = time.perf_counter()
    with torch.inference_mode():
        generated = dllm.generate(batch_ids, gen_length=args.gen_length,
                                  block_length=args.block_length)
    stages = recorder.finish() if args.trace else {}
    torch.cuda.synchronize(rank)
    request_ms = (time.perf_counter() - request_start) * 1000.0

    if args.trace and rank == 0:
        torch.cuda.synchronize(rank)
        ownership = np.arange(config.num_experts) // (config.num_experts // world)
        serializable = []
        for row in route_rows:
            counts = row.pop("expert_counts_gpu").cpu().numpy().astype(int)
            row["moe_ms"] = float(row.pop("moe_start").elapsed_time(row.pop("moe_end")))
            row["expert_counts"] = counts.tolist()
            row["rank_load"] = [int(counts[ownership == r].sum()) for r in range(world)]
            serializable.append(row)
        with (output / f"trajectory_{args.run_id}_rank0.jsonl").open("w", encoding="utf-8") as stream:
            for row in serializable:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    payload = {
        "rank": rank,
        "physical_gpu": rank + 6,
        "run_id": args.run_id,
        "backend": args.backend,
        "trace": args.trace,
        "batch_size": args.batch_size,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "prompt_length": args.prompt_length,
        "cache": args.cache,
        "threshold": args.threshold,
        "early_stop": not args.disable_early_stop,
        "eos_disabled": args.disable_eos,
        "request_ms": request_ms,
        "nfe": dllm.num_forwards - nfe_before,
        "stage_sums_ms": stages,
        "iterations": iteration_rows,
        "output_ids": generated.detach().cpu().tolist() if rank == 0 else None,
        "output_text": [tokenizer.decode(item, skip_special_tokens=True)
                        for item in generated] if rank == 0 else None,
    }
    write_rank_json(output, f"trajectory_{args.run_id}_{'trace' if args.trace else 'clean'}", rank, payload)
    dist.barrier()
    distributed.destroy_model_parallel()
    dist.destroy_process_group()
    guard.__exit__(None, None, None)


def worker(rank: int, world: int, port: int, args) -> None:
    if args.operation == "audit":
        run_audit(rank, world, port, args)
    elif args.operation == "scaling":
        run_scaling(rank, world, port, args)
    elif args.operation == "trajectory":
        run_trajectory(rank, world, port, args)
    else:
        raise ValueError(args.operation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", default="naive",
                        choices=("naive", "pplx", "deepep_high_throughput",
                                 "deepep_low_latency", "allgather_reducescatter"))
    parser.add_argument("--operation", required=True,
                        choices=("audit", "scaling", "trajectory"))
    parser.add_argument("--audit-tokens", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--active-sizes", default="2,4,8,16,32,48,64,96,128,256")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--max-local-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=260911)
    parser.add_argument("--run-id", default="run0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--cache", choices=("none", "prefix", "dual"), default="none")
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--disable-eos", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must equal {EXPECTED_VISIBLE}")
    if torch.cuda.device_count() != 2:
        raise SystemExit("exactly two visible GPUs are required")
    if args.backend == "allgather_reducescatter":
        raise SystemExit("this pinned vLLM does not expose allgather_reducescatter")
    os.environ["VLLM_ALL2ALL_BACKEND"] = args.backend
    mp.spawn(worker, args=(2, free_port(), args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
