#!/usr/bin/env python3
"""Run paired LLaDA2.0-mini route-pruning quality trajectories on GPU0/1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduction.run_reference import (
    REVISION, SNAPSHOT, normalize_generated_ids, parse_gsm8k, samples,
)
from route_pruning import RoutePruningConfig, RoutePruningEmulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--policy", choices=("vanilla", "p2", "p4"), required=True)
    parser.add_argument("--mass-budget", type=float, required=True)
    parser.add_argument("--target-ep", type=int, choices=(4, 8), default=8)
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--gen-length", type=int, default=2048)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if args.gen_length % 32:
        raise ValueError("generation length must be block aligned")
    config = RoutePruningConfig(
        policy=args.policy, mass_budget=args.mass_budget,
        min_k=args.min_k, target_ep=args.target_ep,
    )
    if args.source.suffix == ".jsonl":
        captured = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
        rows = [{
            "id": int(row["sample_id"]),
            "question": row.get("original_prompt", row.get("raw_prompt", "")),
            "raw_prompt": row.get("original_prompt", row.get("raw_prompt", "")),
            "answer": (f"#### {row['ground_truth']}" if row.get("ground_truth") is not None else None),
            "test": row.get("test"), "entry_point": row.get("entry_point"),
        } for row in captured]
    else:
        rows = samples(args.task, args.source, "dinfer-fourshot", args.yaml)
    by_id = {int(row["id"]): row for row in rows}
    chosen = [by_id[sample_id] for sample_id in args.sample_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    generations_path = args.output_dir / "generations.jsonl"
    completed = {
        int(path.stem.split("_")[-1]) for path in trace_dir.glob("request_*.npz")
    }

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, revision=REVISION, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(json.dumps({
        "event": "worker_ready", "physical_gpu": args.device_index,
        "requests": len(chosen), "config": config.__dict__,
    }), flush=True)
    for sequence, row in enumerate(chosen):
        request_id = int(row["id"])
        if request_id in completed:
            continue
        raw_prompt = row.get("raw_prompt", row["question"])
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        emulator = RoutePruningEmulator(model, config, request_id)
        torch.manual_seed(20260915 + request_id)
        emulator.install(tokenizer.mask_token_id)
        with torch.inference_mode():
            generated = model.generate(
                inputs=input_ids, temperature=0.0, threshold=0.95,
                block_length=32, steps=32, gen_length=args.gen_length,
                eos_early_stop=True,
            )
        emulator.stop()
        generated_ids, returned_prompt = normalize_generated_ids(generated, input_ids)
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, parser_rule = parse_gsm8k(output)
        gold = (
            row["answer"].split("####")[-1].strip().replace(",", "")
            if row.get("answer") is not None else None
        )
        remaining_masks = generated_ids.count(tokenizer.mask_token_id)
        termination = (
            "REMAINING_MASK" if remaining_masks else
            "EOS" if tokenizer.eos_token_id in generated_ids else "GEN_LENGTH_CAP"
        )
        stats = emulator.summary()
        trace_path = trace_dir / f"request_{request_id:04d}.npz"
        emulator.save(trace_path, {
            "model": "inclusionAI/LLaDA2.0-mini", "revision": REVISION,
            "request_id": request_id, "threshold": 0.95,
            "block_length": 32, "steps_max_per_block": 32,
            "gen_length": args.gen_length, "task": args.task,
        })
        record = {
            "sample_id": request_id, "task": args.task,
            "policy": args.policy, "mass_budget": args.mass_budget,
            "target_ep": args.target_ep, "min_k": args.min_k,
            "correct": parsed == gold if gold is not None else None,
            "parsed_answer": parsed, "ground_truth": gold,
            "parser_rule": parser_rule, "output": output,
            "output_ids": generated_ids, "generation_tokens": len(generated_ids),
            "termination": termination, "remaining_masks": remaining_masks,
            "nfe": len(emulator.iterations),
            "blocks": len({row["block_id"] for row in emulator.iterations}),
            "official_return_included_prompt": returned_prompt,
            "trace": str(trace_path), "physical_gpu": args.device_index,
            **stats,
        }
        with generations_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "event": "request_complete", "request": request_id,
            "sequence": sequence + 1, "of": len(chosen),
            "correct": record["correct"], "nfe": record["nfe"],
            "actual_removed_mass": record["actual_removed_mass_fraction"],
            "avg_k": record["avg_k"], "termination": termination,
        }), flush=True)
        del emulator, generated, input_ids
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
