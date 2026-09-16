#!/usr/bin/env python3
"""Collect bounded aggregate discovery traces with independent GPU workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduction.run_reference import REVISION, SNAPSHOT, normalize_generated_ids, parse_gsm8k, samples
from virtual_ep.discovery_trace import DiscoveryTraceCollector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--worker", type=int, choices=(0, 1), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-tag", type=str,
                        help="separate generation log tag for rebalanced workers")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gen-length", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if args.threshold != 0.95:
        raise ValueError("discovery cohort is frozen at threshold 0.95")
    if args.workers != 2:
        raise ValueError("this collection protocol uses two independent workers")
    cohort = samples("gsm8k", args.source, "dinfer-fourshot", args.yaml)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        chosen = [row for row in cohort if row["id"] in wanted]
    else:
        chosen = cohort[args.start : args.start + args.limit]
    chosen = [row for row in chosen if row["id"] % args.workers == args.worker]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    tag = args.output_tag or f"worker{args.worker}"
    generations = args.output_dir / f"generations_{tag}.jsonl"
    completed = {int(path.stem.split("_")[-1]) for path in shard_dir.glob("request_*.npz")}

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, revision=REVISION, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    metadata = {
        "model": "inclusionAI/LLaDA2.0-mini", "revision": REVISION,
        "dtype": "bfloat16", "hidden_size": int(model.config.hidden_size),
        "num_routed_experts": int(model.config.num_experts),
        "num_shared_experts": int(model.config.num_shared_experts),
        "top_k": int(model.config.num_experts_per_tok),
        "num_layers": int(model.config.num_hidden_layers),
        "threshold": args.threshold, "steps_max_per_block": 32, "block_length": 32,
        "collection_topology": "independent_single_gpu_worker_not_ep2",
        "worker": args.worker, "physical_gpu": args.device_index,
    }
    print(json.dumps({"event": "worker_ready", "worker": args.worker, "requests": len(chosen),
                      "already_complete": len(completed), "device": str(device)}), flush=True)
    for sequence, row in enumerate(chosen):
        request_id = int(row["id"])
        if request_id in completed:
            continue
        raw_prompt = row.get("raw_prompt", row["question"])
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        collector = DiscoveryTraceCollector(model, tokenizer.mask_token_id, 32, metadata)
        torch.manual_seed(20260915 + request_id)
        collector.start(request_id, int(input_ids.shape[1]))
        with torch.inference_mode():
            generated = model.generate(
                inputs=input_ids, temperature=0.0, threshold=args.threshold,
                block_length=32, steps=32, gen_length=args.gen_length,
                eos_early_stop=True,
            )
        collector.stop()
        trace = collector.trace()
        target = shard_dir / f"request_{request_id:04d}.npz"
        trace.save(target)
        generated_ids, returned_prompt = normalize_generated_ids(generated, input_ids)
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, parser_rule = parse_gsm8k(output)
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        record = {
            "sample_id": request_id, "correct": parsed == gold,
            "parsed_answer": parsed, "ground_truth": gold, "parser_rule": parser_rule,
            "generation_tokens": len(generated_ids),
            "remaining_mask_tokens": generated_ids.count(tokenizer.mask_token_id),
            "prompt_tokens": int(input_ids.shape[1]),
            "nfe": int(len(trace.arrays["request_id"]) // 19),
            "blocks": int(trace.arrays["block_id"].max(initial=-1) + 1),
            "returned_prompt": returned_prompt, "output": output,
            "trace": str(target), "worker": args.worker,
        }
        with generations.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "request_complete", "request": request_id,
                          "sequence": sequence + 1, "of": len(chosen), "correct": parsed == gold,
                          "nfe": record["nfe"], "blocks": record["blocks"]}), flush=True)
        del generated, input_ids, collector, trace
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
