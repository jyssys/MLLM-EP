#!/usr/bin/env python3
"""Clean or same-device-event stage timing for a real true-EP2 request."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


EXPECTED_VISIBLE = "6,7"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from poc_dllm_ep2_policy.scripts.ep2_runner import (  # noqa: E402
    EventRecorder,
    forward_context,
    free_port,
    initialize,
    install_stage_hooks,
    write_rank_json,
)


def run(rank, world, port, args):
    distributed, guard, vconfig, config, tokenizer, model, manager = initialize(
        rank, world, port, args
    )
    from dinfer import BlockIteratorFactory, ThresholdParallelDecoder
    from dinfer.decoding.generate_uniform import BlockWiseDiffusionLLM

    recorder = EventRecorder(rank)
    if args.instrument:
        first = install_stage_hooks(model, manager, recorder, layer_id=0)
        for layer_id in range(1, len(model.model.layers)):
            if first["modular"]:
                install_stage_hooks(model, manager, recorder, layer_id=layer_id)
            else:
                quant = model.model.layers[layer_id].mlp.experts.quant_method
                quant.apply = recorder.wrap("expert", quant.apply)
        for layer in model.model.layers:
            layer.mlp.gate.forward = recorder.wrap("router_linear", layer.mlp.gate.forward)
            layer.mlp.forward = recorder.wrap("moe_total", layer.mlp.forward)

    decoder = ThresholdParallelDecoder(
        temperature=0, threshold=args.threshold, mask_id=156895, eos_id=-1
    )
    dllm = BlockWiseDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(start_block_align=True),
        cache_factory=None,
        early_stop=False,
    )

    original_forward = model.forward

    def contextual_forward(self, input_ids=None, *positional, **kwargs):
        token_count = int(input_ids.numel()) if input_ids is not None else int(
            kwargs["inputs_embeds"].shape[0] * kwargs["inputs_embeds"].shape[1]
        )
        with forward_context(vconfig, token_count):
            return original_forward(input_ids, *positional, **kwargs)

    model.forward = types.MethodType(contextual_forward, model)

    text = (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        "<role>HUMAN</role>Explain why the sky is blue in one concise paragraph."
        "<|role_end|><role>ASSISTANT</role>"
    )
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    if ids.numel() < args.prompt_length:
        ids = ids.repeat((args.prompt_length + ids.numel() - 1) // ids.numel())
    ids = ids[:args.prompt_length].unsqueeze(0).to(f"cuda:{rank}")
    dist.broadcast(ids, src=0)

    for _ in range(args.warmup_requests):
        recorder.enabled = False
        with torch.inference_mode():
            dllm.generate(ids, gen_length=args.gen_length, block_length=args.block_length)
        torch.cuda.synchronize(rank)
        dist.barrier()

    recorder.reset()
    recorder.enabled = args.instrument
    torch.cuda.synchronize(rank)
    dist.barrier()
    before = dllm.num_forwards
    start = time.perf_counter()
    with torch.inference_mode():
        generated = dllm.generate(
            ids, gen_length=args.gen_length, block_length=args.block_length
        )
    stages = recorder.finish() if args.instrument else {}
    torch.cuda.synchronize(rank)
    request_ms = (time.perf_counter() - start) * 1000.0
    payload = {
        "run_id": args.run_id,
        "rank": rank,
        "physical_gpu": rank + 6,
        "backend": args.backend,
        "instrument": args.instrument,
        "request_ms": request_ms,
        "nfe": dllm.num_forwards - before,
        "stage_sums_ms": stages,
        "output_ids": generated.cpu().tolist() if rank == 0 else None,
        "tp": 1,
        "dp": 2,
        "ep": 2,
        "num_experts": int(config.num_experts),
        "top_k": int(config.num_experts_per_tok),
    }
    write_rank_json(Path(args.output), args.run_id, rank, payload)
    dist.barrier()
    distributed.destroy_model_parallel()
    dist.destroy_process_group()
    guard.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", default="deepep_high_throughput")
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--warmup-requests", type=int, default=2)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must equal {EXPECTED_VISIBLE}")
    if torch.cuda.device_count() != 2:
        raise SystemExit("exactly two visible GPUs are required")
    os.environ["VLLM_ALL2ALL_BACKEND"] = args.backend
    mp.spawn(run, args=(2, free_port(), args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
