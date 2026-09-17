#!/usr/bin/env python3
"""Run one common-protocol offline method adapter on physical GPU0/1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from method_adapters.team_port import (
    TeamPortConfig, generate_team_port, save_team_trace,
)
from reproduction.run_reference import REVISION, SNAPSHOT, normalize_generated_ids, parse_gsm8k


def read_rows(path: Path, task: str) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {}
    for row in rows:
        rid = int(row["sample_id"])
        result[rid] = {
            "id": rid,
            "raw_prompt": row["raw_prompt"],
            "ground_truth": row.get("ground_truth"),
            "test": row.get("test"),
            "entry_point": row.get("entry_point"),
            "original_prompt": row.get("original_prompt", row["raw_prompt"]),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--method", choices=("team_port"), required=True)
    parser.add_argument("--gen-length", type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if args.worker >= args.workers or args.gen_length % 32:
        raise ValueError("invalid worker or generation length")
    by_id = read_rows(args.source, args.task)
    selected = [by_id[rid] for rid in args.sample_ids if rid % args.workers == args.worker]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output_dir / "traces"; trace_dir.mkdir(exist_ok=True)
    output_path = args.output_dir / f"generations_worker{args.worker}.jsonl"
    completed = {int(path.stem.split("_")[-1]) for path in trace_dir.glob("request_*.npz")}

    device = torch.device(f"cuda:{args.device_index}"); torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, revision=REVISION, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(json.dumps({"event": "worker_ready", "method": args.method,
                      "task": args.task, "requests": len(selected),
                      "physical_gpu": args.device_index}), flush=True)
    for ordinal, row in enumerate(selected):
        rid = row["id"]
        if rid in completed: continue
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["raw_prompt"]}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        torch.manual_seed(20260915 + rid)
        result = generate_team_port(
            model, input_ids, threshold=0.95, block_length=32, steps=32,
            gen_length=args.gen_length, eos_early_stop=True,
            eos_id=tokenizer.eos_token_id, mask_id=tokenizer.mask_token_id,
            request_id=rid, config=TeamPortConfig(),
        )
        generated_ids, returned_prompt = normalize_generated_ids(result.generated, input_ids)
        raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
        clean = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, parser_rule = parse_gsm8k(clean)
        gold = row["ground_truth"]
        if gold is not None: gold = str(gold).replace(",", "")
        remaining = generated_ids.count(tokenizer.mask_token_id)
        termination = ("REMAINING_MASK" if remaining else "EOS" if tokenizer.eos_token_id in generated_ids
                       else "GEN_LENGTH_CAP")
        trace_path = trace_dir / f"request_{rid:04d}.npz"
        save_team_trace(trace_path, result, {
            "schema_version": 1, "method_schema_version": 2,
            "method": "TEAM-PORT-DCD-LAC", "official_team_commit":
            "e9c502e5753ce79f660371e2fb4a8666f66cae75",
            "speculative_exploration_executed": False,
            "speculative_candidate_construction_unit_tested": True,
            "checkpoint": "inclusionAI/LLaDA2.0-mini", "revision": REVISION,
            "threshold": 0.95, "block_length": 32, "steps": 32,
            "gen_length": args.gen_length, "task": args.task,
            "routed_layer_ids": [i for i, layer in enumerate(model.model.layers)
                                 if hasattr(layer.mlp, "gate")],
            "dense_emulation_no_speedup_claim": True,
        })
        fresh_tokens = sum(int(r["fresh"].sum()) for r in result.records)
        reused_tokens = sum(int(r["reused"].sum()) for r in result.records)
        fresh_pairs = sum(int(r["hist"].sum()) for r in result.records)
        active_expert_events = sum(int(np.count_nonzero(r["hist"])) for r in result.records)
        invocations = len(result.records) * len(result.records[0]["hist"])
        record = {
            "sample_id": rid, "task": args.task, "method": "TEAM-PORT-DCD-LAC",
            "correct": parsed == gold if args.task == "gsm8k" else None,
            "parsed_answer": parsed, "ground_truth": gold,
            "parser_rule": parser_rule, "output": clean,
            "postprocessed_generation": clean, "raw_generation": raw,
            "output_ids": generated_ids, "generation_tokens": len(generated_ids),
            "nfe": len(result.records), "blocks": len({r["block_id"] for r in result.records}),
            "termination": termination, "termination_reason": termination,
            "remaining_masks": remaining, "remaining_mask_tokens": remaining,
            "fresh_token_updates": fresh_tokens, "reused_token_updates": reused_tokens,
            "fresh_expert_token_pairs": fresh_pairs, "avg_k": 8.0,
            "unique_activated_experts_per_forward": active_expert_events / max(invocations, 1),
            "official_return_included_prompt": returned_prompt,
            "trace": str(trace_path), "physical_gpu": args.device_index,
            "original_prompt": row["original_prompt"], "test": row["test"],
            "entry_point": row["entry_point"],
        }
        with output_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "request_complete", "request": rid,
                          "sequence": ordinal + 1, "of": len(selected),
                          "correct": record["correct"], "nfe": record["nfe"],
                          "fresh_pairs": fresh_pairs}), flush=True)
        del result, input_ids; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
