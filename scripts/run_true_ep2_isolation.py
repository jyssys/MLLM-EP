#!/usr/bin/env python3
"""Run exact official-HF LLaDA2.0-mini generation through true routed EP2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduction.run_reference import (
    REVISION,
    SNAPSHOT,
    normalize_generated_ids,
    parse_gsm8k,
    samples,
)
from virtual_ep.true_ep2_harness import TrueEP2Harness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("sanity", "gsm8k"), default="sanity")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--yaml", type=Path, default=Path("/home/esjung/external/dinfer-llada2-flash-poc/evaluations/tasks/gsm8k/gsm8k-llada-mini.yaml"))
    parser.add_argument("--protocol", choices=("raw-question", "dinfer-fourshot"), default="raw-question")
    parser.add_argument("--sample-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-parity", action="store_true")
    parser.add_argument("--moe-parity", action="store_true")
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if os.environ.get("SGL_ENABLE_JIT_DEEPGEMM") != "0":
        raise RuntimeError(
            "the BF16 correctness harness requires SGL_ENABLE_JIT_DEEPGEMM=0"
        )
    if int(os.environ.get("WORLD_SIZE", "-1")) != 2:
        raise RuntimeError("true EP2 requires torchrun world size exactly 2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, revision=REVISION, trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(f"cuda:{local_rank}").eval()
    cohort = samples(args.task, args.source, args.protocol, args.yaml)
    by_id = {row["id"]: row for row in cohort}
    moe_parity_hidden = None
    moe_parity_baseline = {}
    if args.moe_parity:
        generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(77001)
        moe_parity_hidden = torch.randn(
            (1, 32, model.config.hidden_size),
            dtype=torch.bfloat16,
            device=f"cuda:{local_rank}",
            generator=generator,
        )
        with torch.inference_mode():
            for layer_id, layer in enumerate(model.model.layers):
                if hasattr(layer.mlp, "experts"):
                    moe_parity_baseline[layer_id] = layer.mlp(moe_parity_hidden)[0].detach()
    parity_input = parity_baseline = None
    if args.forward_parity:
        parity_row = by_id[args.sample_ids[0]]
        parity_prompt = parity_row.get("raw_prompt", parity_row["question"])
        parity_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": parity_prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(local_rank)
        parity_length = parity_input.shape[1]
        parity_attention = torch.zeros(
            (1, 1, parity_length, parity_length),
            dtype=torch.bfloat16,
            device=parity_input.device,
        )
        parity_positions = torch.arange(
            parity_length, device=parity_input.device
        ).unsqueeze(0)
        with torch.inference_mode():
            parity_baseline = model(
                parity_input,
                attention_mask=parity_attention,
                position_ids=parity_positions,
            ).logits.detach()
    harness = TrueEP2Harness(model, args.output, trace_timing=not args.no_timing)
    if args.moe_parity:
        moe_parity_rows = []
        with torch.inference_mode():
            for layer_id, baseline in moe_parity_baseline.items():
                current = model.model.layers[layer_id].mlp(moe_parity_hidden)[0]
                difference = current.float() - baseline.float()
                moe_parity_rows.append({
                    "rank": rank,
                    "layer": layer_id,
                    "max_abs": float(difference.abs().max()),
                    "mean_abs": float(difference.abs().mean()),
                    "relative_l2": float(
                        difference.norm() / baseline.float().norm().clamp_min(1e-12)
                    ),
                })
        (args.output / f"moe_parity_rank{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in moe_parity_rows)
        )
        del moe_parity_hidden, moe_parity_baseline
    if args.forward_parity:
        with torch.inference_mode():
            parity_ep = model(
                parity_input,
                attention_mask=parity_attention,
                position_ids=parity_positions,
            ).logits.detach()
        delta = (parity_ep.float() - parity_baseline.float())
        baseline_norm = parity_baseline.float().norm().clamp_min(1e-12)
        parity = {
            "rank": rank,
            "max_abs": float(delta.abs().max()),
            "mean_abs": float(delta.abs().mean()),
            "relative_l2": float(delta.norm() / baseline_norm),
            "top1_match_rate": float(
                (parity_ep.argmax(dim=-1) == parity_baseline.argmax(dim=-1)).float().mean()
            ),
        }
        (args.output / f"forward_parity_rank{rank}.json").write_text(
            json.dumps(parity, indent=2) + "\n"
        )
        del parity_input, parity_attention, parity_positions, parity_baseline, parity_ep
    outputs = []
    original_forward = model.forward
    forward_counter = {"count": 0}

    def counted_forward(*forward_args, **forward_kwargs):
        forward_counter["count"] += 1
        return original_forward(*forward_args, **forward_kwargs)

    model.forward = counted_forward
    for sample_id in args.sample_ids:
        row = by_id[sample_id]
        prompt = row.get("raw_prompt", row["question"])
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(local_rank)
        torch.manual_seed(20260915 + sample_id)
        harness.request_id = sample_id
        forward_counter["count"] = 0
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        before.record()
        generated = model.generate(
            inputs=input_ids, temperature=0.0, threshold=0.95,
            block_length=32, steps=32, gen_length=args.gen_length,
            eos_early_stop=True,
        )
        after.record()
        after.synchronize()
        ids, included_prompt = normalize_generated_ids(generated, input_ids)
        output_text = tokenizer.decode(ids, skip_special_tokens=True)
        parsed, _ = parse_gsm8k(output_text)
        gold = (
            row["answer"].split("####")[-1].strip().replace(",", "")
            if args.task == "gsm8k"
            else None
        )
        rank_outputs = [None, None]
        dist.all_gather_object(rank_outputs, ids)
        rank_parity = rank_outputs[0] == rank_outputs[1]
        if not rank_parity:
            raise RuntimeError("EP2 ranks diverged in generated token IDs")
        outputs.append(
            {
                "sample_id": sample_id,
                "rank": rank,
                "output_ids": ids,
                "output": output_text,
                "parsed_answer": parsed,
                "ground_truth": gold,
                "correct": parsed == gold if gold is not None else None,
                "nfe": forward_counter["count"],
                "rank_token_identity": rank_parity,
                "official_return_included_prompt": included_prompt,
                "ep_isolation_harness_wall_ms": before.elapsed_time(after),
                "production_serving_latency": False,
            }
        )
        del input_ids, generated
        torch.cuda.empty_cache()
    (args.output / f"generation_rank{rank}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs)
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
