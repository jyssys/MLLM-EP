#!/usr/bin/env python3
"""Collect official-HF heavy routes on physical GPU 0 only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduction.run_reference import (
    REVISION,
    SNAPSHOT,
    normalize_generated_ids,
    parse_gsm8k,
    samples,
)
from virtual_ep.hf_trace import HFTraceCollector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, default=Path("/home/esjung/external/dinfer-llada2-flash-poc/evaluations/tasks/gsm8k/gsm8k-llada-mini.yaml"))
    parser.add_argument("--protocol", choices=("raw-question", "dinfer-fourshot"), default="dinfer-fourshot")
    parser.add_argument("--sample-ids", type=int, nargs="+", required=True)
    parser.add_argument("--gen-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--device-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--parity-first", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if args.gen_length % 32:
        raise ValueError("gen-length must be a multiple of block length 32")
    if args.output.exists() or args.generations.exists():
        raise FileExistsError("refusing to mix a fresh rollout with existing output")
    all_rows = samples(args.task, args.source, args.protocol, args.yaml)
    by_id = {row["id"]: row for row in all_rows}
    chosen = [by_id[sample_id] for sample_id in args.sample_ids]
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, revision=REVISION, trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(f"cuda:{args.device_index}").eval()
    collector = HFTraceCollector(
        model,
        tokenizer.mask_token_id,
        32,
        {
            "model": "inclusionAI/LLaDA2.0-mini",
            "revision": REVISION,
            "dtype": "bfloat16",
            "hidden_size": int(model.config.hidden_size),
            "num_routed_experts": int(model.config.num_experts),
            "num_shared_experts": int(model.config.num_shared_experts),
            "top_k": int(model.config.num_experts_per_tok),
            "num_layers": int(model.config.num_hidden_layers),
            "threshold": args.threshold,
            "steps_max_per_block": 32,
            "block_length": 32,
        },
    )
    args.generations.parent.mkdir(parents=True, exist_ok=True)
    for sequence, row in enumerate(chosen):
        raw_prompt = row.get("raw_prompt", row["question"])
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(f"cuda:{args.device_index}")
        seed = 20260915 + row["id"]
        reference = None
        if args.parity_first and sequence == 0:
            torch.manual_seed(seed)
            reference = model.generate(
                inputs=input_ids, temperature=0.0, threshold=args.threshold,
                block_length=32, steps=32, gen_length=args.gen_length,
                eos_early_stop=True,
            )
        torch.manual_seed(seed)
        collector.start(row["id"])
        generated = model.generate(
            inputs=input_ids, temperature=0.0, threshold=args.threshold,
            block_length=32, steps=32, gen_length=args.gen_length,
            eos_early_stop=True,
        )
        collector.stop()
        if reference is not None and not torch.equal(reference, generated):
            raise RuntimeError("trace hooks changed official generation output")
        generated_ids, returned_prompt = normalize_generated_ids(generated, input_ids)
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, parser_rule = parse_gsm8k(output_text)
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        with args.generations.open("a") as stream:
            stream.write(json.dumps({
                "sample_id": row["id"],
                "output_ids": generated_ids,
                "output": output_text,
                "parsed_answer": parsed,
                "parser_rule": parser_rule,
                "ground_truth": gold,
                "correct": parsed == gold,
                "generation_tokens": len(generated_ids),
                "remaining_mask_tokens": generated_ids.count(tokenizer.mask_token_id),
                "threshold": args.threshold,
                "official_return_included_prompt": returned_prompt,
                "trace_parity": reference is None or torch.equal(reference, generated),
            }, ensure_ascii=False) + "\n")
        del generated, input_ids, reference
        torch.cuda.empty_cache()
    collector.bundle().save(args.output)


if __name__ == "__main__":
    main()
