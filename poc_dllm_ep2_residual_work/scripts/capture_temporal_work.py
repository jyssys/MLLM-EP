#!/usr/bin/env python3
"""Capture iteration/layer/token routing and expert-branch evolution on true EP2.

The script is intentionally observer-heavy.  Clean latency comes from the
existing, separately executed EP2 runner.  Full hidden/router/MoE-output state
is captured on rank 0; exact individual branch outputs are reconstructed from
the actual sharded expert weights for representative layers on both ranks and
summed across the EP group.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


EXPECTED_VISIBLE = "6,7"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from poc_dllm_ep2_policy.scripts.ep2_runner import (  # noqa: E402
    forward_context,
    free_port,
    initialize,
    write_rank_json,
)


def _encode(tokenizer, prompt: str, length: int) -> torch.Tensor:
    formatted = (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        "<role>HUMAN</role>" + prompt
        + "<|role_end|><role>ASSISTANT</role>"
    )
    ids = tokenizer(formatted, return_tensors="pt")["input_ids"][0]
    if ids.numel() < length:
        ids = ids.repeat((length + ids.numel() - 1) // ids.numel())
    return ids[:length]


def _manual_branches(
    rank: int,
    model,
    records: list[dict[str, Any]],
    selected_layers: list[int],
    prompt_length: int,
    gen_length: int,
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    """Evaluate actual selected expert branches and reconstruct fused output.

    Each EP rank computes only experts it owns.  An EP all-reduce then produces
    one dense [call, generated_position, branch, hidden] tensor on both ranks.
    This is post-hoc observer work, not a candidate implementation.
    """
    by_key = {(int(r["call_id"]), int(r["layer_id"])): r for r in records}
    call_ids = sorted({int(r["call_id"]) for r in records})
    outputs: dict[int, torch.Tensor] = {}
    validation: dict[int, dict[str, float]] = {}

    sequence_length = prompt_length + gen_length
    generated_rows = torch.cat([
        torch.arange(
            b * sequence_length + prompt_length,
            b * sequence_length + prompt_length + gen_length,
            device=f"cuda:{rank}",
        )
        for b in range(batch_size)
    ])
    generated_positions = batch_size * gen_length

    for layer_id in selected_layers:
        layer = model.model.layers[layer_id]
        experts = layer.mlp.experts
        expert_map = experts.expert_map
        hidden_size = int(layer.mlp.hidden_size)
        top_k = int(layer.mlp.top_k)
        branches = torch.zeros(
            (len(call_ids), generated_positions, top_k, hidden_size),
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )

        for ci, call_id in enumerate(call_ids):
            rec = by_key[(call_id, layer_id)]
            hidden = rec["hidden"].index_select(0, generated_rows)
            top_ids = rec["top_ids"].index_select(0, generated_rows)
            for global_expert in range(int(layer.mlp.num_experts)):
                local_expert = int(expert_map[global_expert].item())
                if local_expert < 0:
                    continue
                locations = (top_ids == global_expert).nonzero(as_tuple=False)
                if locations.numel() == 0:
                    continue
                token_rows = locations[:, 0]
                branch_rows = locations[:, 1]
                x = hidden.index_select(0, token_rows)
                gate_up = F.linear(x, experts.w13_weight[local_expert])
                half = gate_up.shape[-1] // 2
                intermediate = F.silu(gate_up[:, :half]) * gate_up[:, half:]
                value = F.linear(intermediate, experts.w2_weight[local_expert])
                branches[ci, token_rows, branch_rows] = value

        dist.all_reduce(branches, op=dist.ReduceOp.SUM)

        # Confirm the manual decomposition against the runtime's actual MoE
        # output.  Different fused-GEMM accumulation order may cause BF16 drift.
        rel_errors = []
        cosine = []
        for ci, call_id in enumerate(call_ids):
            rec = by_key[(call_id, layer_id)]
            weights = rec["top_weights"].index_select(0, generated_rows)
            manual = (branches[ci].float() * weights.float().unsqueeze(-1)).sum(dim=1)
            actual = rec["moe_output"].index_select(0, generated_rows).float()
            rel = torch.linalg.vector_norm(manual - actual) / torch.linalg.vector_norm(actual).clamp_min(1e-12)
            rel_errors.append(float(rel.item()))
            cosine.append(float(F.cosine_similarity(manual, actual, dim=-1).mean().item()))
        validation[layer_id] = {
            "max_rel_l2": max(rel_errors),
            "mean_rel_l2": sum(rel_errors) / len(rel_errors),
            "min_mean_token_cosine": min(cosine),
        }
        if rank == 0:
            outputs[layer_id] = branches.cpu()
        del branches

    return outputs, validation


def run_capture(rank: int, world: int, port: int, args) -> None:
    distributed, guard, vconfig, config, tokenizer, model, manager = initialize(
        rank, world, port, args
    )
    from dinfer import BlockIteratorFactory, ThresholdParallelDecoder
    from dinfer.decoding.generate_uniform import BlockWiseDiffusionLLM

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    selected_layers = [int(x) for x in args.branch_layers.split(",") if x]
    dense_layers = [int(x) for x in args.dense_layers.split(",") if x]
    if any(x < 0 or x >= len(model.model.layers) for x in selected_layers):
        raise ValueError("branch layer outside model")

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

    prompts = {
        "expository": "Explain why the sky is blue in one concise paragraph.",
        "math": "Solve 17 times 23 and briefly show the arithmetic.",
        "systems": "Explain expert parallel inference and its communication costs.",
        "creative": "Write a short story about a lighthouse during a storm.",
    }
    prompt = prompts[args.prompt_kind]
    prompt_order = ["expository", "math", "systems", "creative"]
    input_ids = torch.stack([
        _encode(tokenizer, prompts[prompt_order[(prompt_order.index(args.prompt_kind) + b) % len(prompt_order)]], args.prompt_length)
        for b in range(args.batch_size)
    ]).to(f"cuda:{rank}")
    dist.broadcast(input_ids, src=0)

    current: dict[str, Any] = {}
    call_counter = 0
    capture_enabled = False
    records: list[dict[str, Any]] = []
    iteration_states: list[dict[str, Any]] = []

    original_model_forward = model.forward

    def contextual_forward(self, input_ids=None, *positional, **kwargs):
        nonlocal call_counter
        if input_ids is not None:
            token_count = int(input_ids.numel())
        else:
            embeds = kwargs["inputs_embeds"]
            token_count = int(embeds.shape[0] * embeds.shape[1])
        current["call_id"] = call_counter
        current["model_positions"] = token_count
        call_counter += 1
        with forward_context(vconfig, token_count):
            return original_model_forward(input_ids, *positional, **kwargs)

    model.forward = types.MethodType(contextual_forward, model)

    for layer_id, layer in enumerate(model.model.layers):
        original_moe = layer.mlp.forward

        def capture_moe(self, hidden_states, _orig=original_moe, _layer=layer_id):
            if not capture_enabled:
                return _orig(hidden_states)
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            with torch.inference_mode():
                router_logits, _ = self.gate(flat.float())
                probs = torch.softmax(router_logits, dim=-1)
                top_weights, top_ids = torch.topk(
                    probs, k=self.top_k, dim=-1, sorted=True
                )
                result = _orig(hidden_states)
            # Representative-layer hidden states are needed on both ranks for
            # branch reconstruction; all other dense tensors are rank-0 only.
            if rank == 0 or _layer in selected_layers:
                rec = {
                    "call_id": int(current["call_id"]),
                    "iteration_id": int(current["iteration_id"]),
                    "block_id": int(current["block_id"]),
                    "masked_before": int(current["masked_before"]),
                    "layer_id": _layer,
                    "model_positions": int(flat.shape[0]),
                    "top_ids": top_ids.detach().clone(),
                    "top_weights": top_weights.detach().clone(),
                }
                if _layer in dense_layers or _layer in selected_layers:
                    rec["hidden"] = flat.detach().clone()
                    rec["moe_output"] = result.reshape(-1, result.shape[-1]).detach().clone()
                if rank == 0:
                    rec["router_logits"] = router_logits.detach().to(torch.float16).clone()
                records.append(rec)
            return result

        layer.mlp.forward = types.MethodType(capture_moe, layer.mlp)

    original_iteration = dllm.diff_iteration.forward

    def capture_iteration(self, model_arg, decoder_arg, x, kv_cache, block, block_loc, block_id):
        current.clear()
        current.update(
            iteration_id=int(self.iter_no),
            block_id=int(block_id),
            masked_before=int((block == decoder_arg.mask_id).sum().item()),
        )
        before_tokens = block.detach().clone()
        wall_start = time.perf_counter()
        result = original_iteration(
            model_arg, decoder_arg, x, kv_cache, block, block_loc, block_id
        )
        if capture_enabled:
            iteration_states.append({
                "iteration_id": int(current["iteration_id"]),
                "block_id": int(block_id),
                "call_id": int(current["call_id"]),
                "masked_before": int(current["masked_before"]),
                "masked_after": int((block == decoder_arg.mask_id).sum().item()),
                "tokens_before": before_tokens,
                "tokens_after": block.detach().clone(),
                "iteration_wall_ms_observer": (time.perf_counter() - wall_start) * 1000.0,
            })
        return result

    dllm.diff_iteration.forward = types.MethodType(capture_iteration, dllm.diff_iteration)

    for _ in range(args.warmup_requests):
        with torch.inference_mode():
            dllm.generate(input_ids, gen_length=args.gen_length, block_length=args.block_length)
        torch.cuda.synchronize(rank)
        dist.barrier()

    records.clear()
    iteration_states.clear()
    call_counter = 0
    capture_enabled = True
    torch.cuda.synchronize(rank)
    dist.barrier()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = dllm.generate(
            input_ids, gen_length=args.gen_length, block_length=args.block_length
        )
    torch.cuda.synchronize(rank)
    observer_request_ms = (time.perf_counter() - start) * 1000.0
    capture_enabled = False

    branch_outputs, branch_validation = _manual_branches(
        rank,
        model,
        records,
        selected_layers,
        args.prompt_length,
        args.gen_length,
        args.batch_size,
    )

    # Cross-rank hidden and routing agreement for representative layers.
    agreement = {"max_hidden_abs": 0.0, "route_mismatch": 0}
    selected_records = [r for r in records if int(r["layer_id"]) in selected_layers]
    for rec in selected_records:
        ref_hidden = rec["hidden"].clone()
        ref_ids = rec["top_ids"].clone()
        dist.broadcast(ref_hidden, src=0)
        dist.broadcast(ref_ids, src=0)
        agreement["max_hidden_abs"] = max(
            agreement["max_hidden_abs"],
            float((rec["hidden"] - ref_hidden).abs().max().item()),
        )
        agreement["route_mismatch"] += int((rec["top_ids"] != ref_ids).sum().item())

    if rank == 0:
        ordered = sorted(records, key=lambda r: (int(r["call_id"]), int(r["layer_id"])))
        calls = sorted({int(r["call_id"]) for r in ordered})
        layers = list(range(len(model.model.layers)))
        by_key = {(int(r["call_id"]), int(r["layer_id"])): r for r in ordered}
        artifact = {
            "schema_version": 1,
            "run_id": args.run_id,
            "prompt_kind": args.prompt_kind,
            "prompt_length": args.prompt_length,
            "batch_size": args.batch_size,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "backend": args.backend,
            "physical_gpus": [6, 7],
            "tp": 1,
            "dp": 2,
            "ep": 2,
            "top_k": int(config.num_experts_per_tok),
            "num_experts": int(config.num_experts),
            "observer_request_ms": observer_request_ms,
            "call_ids": calls,
            "layer_ids": layers,
            "call_metadata": [
                {
                    "call_id": int(s["call_id"]),
                    "iteration_id": int(s["iteration_id"]),
                    "block_id": int(s["block_id"]),
                    "masked_before": int(s["masked_before"]),
                    "masked_after": int(s["masked_after"]),
                }
                for s in iteration_states
            ],
            "tokens_before": torch.stack([s["tokens_before"].cpu() for s in iteration_states]),
            "tokens_after": torch.stack([s["tokens_after"].cpu() for s in iteration_states]),
            "top_ids": torch.stack([
                torch.stack([by_key[(c, l)]["top_ids"].cpu() for l in layers])
                for c in calls
            ]),
            "top_weights": torch.stack([
                torch.stack([by_key[(c, l)]["top_weights"].cpu() for l in layers])
                for c in calls
            ]),
            "router_logits": torch.stack([
                torch.stack([by_key[(c, l)]["router_logits"].cpu() for l in layers])
                for c in calls
            ]),
            "dense_layers": dense_layers,
            "hidden": {
                l: torch.stack([by_key[(c, l)]["hidden"].cpu() for c in calls])
                for l in dense_layers
            },
            "moe_output": {
                l: torch.stack([by_key[(c, l)]["moe_output"].cpu() for c in calls])
                for l in dense_layers
            },
            "branch_layers": selected_layers,
            "branch_outputs": branch_outputs,
            "branch_validation": branch_validation,
            "cross_rank_agreement": agreement,
            "generated_ids": generated.cpu(),
        }
        torch.save(artifact, out / f"temporal_{args.run_id}.pt")
        write_rank_json(out, f"temporal_{args.run_id}_summary", rank, {
            "run_id": args.run_id,
            "prompt_kind": args.prompt_kind,
            "calls": len(calls),
            "layers": len(layers),
            "positions": int(artifact["top_ids"].shape[2]),
            "observer_request_ms": observer_request_ms,
            "branch_layers": selected_layers,
            "branch_validation": branch_validation,
            "cross_rank_agreement": agreement,
            "generated_ids_sha": str(hash(tuple(generated.cpu().reshape(-1).tolist()))),
        })

    dist.barrier()
    distributed.destroy_model_parallel()
    dist.destroy_process_group()
    guard.__exit__(None, None, None)


def worker(rank: int, world: int, port: int, args) -> None:
    run_capture(rank, world, port, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", default="deepep_high_throughput")
    parser.add_argument("--run-id", default="trace0")
    parser.add_argument(
        "--prompt-kind",
        choices=("expository", "math", "systems", "creative"),
        default="expository",
    )
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--branch-layers", default="0,7,15")
    parser.add_argument("--dense-layers", default="0,7,15")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must equal {EXPECTED_VISIBLE}")
    if torch.cuda.device_count() != 2:
        raise SystemExit("exactly two visible GPUs are required")
    os.environ["VLLM_ALL2ALL_BACKEND"] = args.backend
    mp.spawn(worker, args=(2, free_port(), args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
