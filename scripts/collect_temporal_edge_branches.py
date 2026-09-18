#!/usr/bin/env python3
"""Collect bounded exact expert-branch drift for Temporal-Edge EP.

This is an observational duplicate-compute trace.  The official routed MoE
result is returned unchanged; five selected layers re-execute current-block
expert branches only so adjacent branch vectors can be compared exactly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduction.run_reference import (
    REVISION, SNAPSHOT, normalize_generated_ids, parse_gsm8k,
)
from selective_refinement.emulator import generate_selective, save_request_trace
from selective_refinement.policy import PolicyConfig


SELECTED_LAYERS = (1, 5, 10, 14, 19)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gen-length", type=int, default=2048)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if args.worker >= args.workers:
        raise ValueError("worker must be smaller than workers")

    captured = [
        json.loads(line) for line in args.source.read_text().splitlines() if line.strip()
    ]
    by_id = {int(row["sample_id"]): row for row in captured}
    chosen = [by_id[sample_id] for sample_id in args.sample_ids]
    chosen = [row for row in chosen if int(row["sample_id"]) % args.workers == args.worker]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    generations = args.output_dir / f"generations_worker{args.worker}.jsonl"
    completed = {int(path.stem.split("_")[-1]) for path in trace_dir.glob("request_*.npz")}

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT, revision=REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, revision=REVISION, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    policy = PolicyConfig(name="full", active_ratio=1.0)
    print(json.dumps({
        "event": "worker_ready", "worker": args.worker,
        "physical_gpu": args.device_index, "requests": len(chosen),
        "selected_layers": SELECTED_LAYERS,
    }), flush=True)

    for sequence, row in enumerate(chosen):
        request_id = int(row["sample_id"])
        if request_id in completed:
            continue
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["raw_prompt"]}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        torch.manual_seed(20260915 + request_id)
        result = generate_selective(
            model, input_ids, policy=policy, semantics="none",
            threshold=0.95, block_length=32, steps=32,
            gen_length=args.gen_length, eos_early_stop=True,
            eos_id=tokenizer.eos_token_id, mask_id=tokenizer.mask_token_id,
            request_id=request_id, capture_adjacent_hidden=False,
            capture_temporal_edges=True, temporal_edge_layers=SELECTED_LAYERS,
        )
        target = trace_dir / f"request_{request_id:04d}.npz"
        save_request_trace(target, result, {
            "schema_version": 2,
            "purpose": "temporal_edge_exact_branch_output_stability",
            "model": "inclusionAI/LLaDA2.0-mini", "revision": REVISION,
            "request_id": request_id, "threshold": 0.95,
            "block_length": 32, "steps": 32, "gen_length": args.gen_length,
            "selected_routed_layers": SELECTED_LAYERS,
            "observational_duplicate_compute_only": True,
            "production_timing_claim": False,
        })
        generated_ids, _returned_prompt = normalize_generated_ids(result.generated, input_ids)
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, rule = parse_gsm8k(output)
        gold = str(row.get("ground_truth", "")).replace(",", "")
        record = {
            "sample_id": request_id, "parsed_answer": parsed,
            "ground_truth": gold, "correct": parsed == gold,
            "parser_rule": rule, "nfe": len(result.records),
            "edge_records": len(result.edge_stability or []),
            "remaining_masks": generated_ids.count(tokenizer.mask_token_id),
            "trace": str(target), "worker": args.worker,
        }
        with generations.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({
            "event": "request_complete", "request": request_id,
            "sequence": sequence + 1, "of": len(chosen),
            "nfe": record["nfe"], "edge_records": record["edge_records"],
        }), flush=True)
        del result, input_ids
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
