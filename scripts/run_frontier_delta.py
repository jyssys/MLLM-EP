#!/usr/bin/env python3
"""Run bounded Phase-0/FrontierEP semantic trajectories on GPU0/1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from frontier_delta.emulator import (
    FrontierConfig, generate_frontier, save_frontier_trace,
)
from reproduction.run_reference import (
    REVISION, SNAPSHOT, normalize_generated_ids, parse_gsm8k,
)


CONFIGS = {
    "audit": FrontierConfig("NAIVE_EXACTNESS_AUDIT", None, audit_exactness=True),
    "block_cached": FrontierConfig("F0_BLOCK_CACHED", None),
    "frontier1": FrontierConfig("F1_ONE_FINALIZATION", 1),
    "frontier2": FrontierConfig("F2_TWO_FINALIZATIONS", 2),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mode", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--capture-delta", action="store_true")
    parser.add_argument("--gen-length", type=int, default=2048)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if args.worker >= args.workers:
        raise ValueError("worker must be smaller than workers")
    config = CONFIGS[args.mode]
    if args.capture_delta:
        config = FrontierConfig(
            config.name, config.finalization_updates,
            audit_exactness=config.audit_exactness, capture_delta=True,
        )
    source_rows = [
        json.loads(line) for line in args.source.read_text().splitlines() if line.strip()
    ]
    by_id = {int(row["sample_id"]): row for row in source_rows}
    chosen = [by_id[index] for index in args.sample_ids]
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
    print(json.dumps({
        "event": "worker_ready", "worker": args.worker,
        "physical_gpu": args.device_index, "mode": args.mode,
        "requests": len(chosen), "capture_delta": args.capture_delta,
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
        result = generate_frontier(
            model, input_ids, config=config, threshold=0.95,
            block_length=32, steps=32, gen_length=args.gen_length,
            temperature=0.0, eos_early_stop=True,
            eos_id=tokenizer.eos_token_id, mask_id=tokenizer.mask_token_id,
            request_id=request_id,
        )
        generated_ids, _returned_prompt = normalize_generated_ids(
            result.generated, input_ids
        )
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, parser_rule = parse_gsm8k(output)
        gold = str(row["ground_truth"]).replace(",", "")
        remaining_masks = generated_ids.count(tokenizer.mask_token_id)
        termination = (
            "REMAINING_MASK" if remaining_masks else
            "EOS" if tokenizer.eos_token_id in generated_ids else "GEN_LENGTH_CAP"
        )
        reference_output = row.get("postprocessed_generation", "")
        reference_nfe = int(row.get("nfe", -1))
        trace_path = trace_dir / f"request_{request_id:04d}.npz"
        save_frontier_trace(trace_path, result, {
            "schema_version": 1,
            "purpose": "frontier_delta_ep_discovery",
            "model": "inclusionAI/LLaDA2.0-mini", "revision": REVISION,
            "request_id": request_id, "threshold": 0.95,
            "block_length": 32, "steps": 32, "gen_length": args.gen_length,
            "mode": args.mode, "config": config.__dict__,
            "selected_layers": [1, 5, 10, 14, 19],
            "dense_semantic_emulation_no_wall_speedup_claim": True,
        })
        record = {
            "sample_id": request_id, "mode": args.mode,
            "correct": parsed == gold, "parsed_answer": parsed,
            "ground_truth": gold, "parser_rule": parser_rule,
            "postprocessed_generation": output,
            "exact_generation_identity": output == reference_output,
            "parsed_answer_identity": parsed == row.get("parsed_answer"),
            "reference_correct": bool(row.get("correct")),
            "reference_nfe": reference_nfe, "nfe": len(result.records),
            "nfe_identity": len(result.records) == reference_nfe,
            "generation_tokens": len(generated_ids),
            "termination": termination, "remaining_masks": remaining_masks,
            "total_fresh_rows": int(sum(r["fresh_rows"] for r in result.records)),
            "total_physical_rows": int(sum(r["physical_rows"] for r in result.records)),
            "fresh_current_rows": int(sum(r["fresh_current_rows"] for r in result.records)),
            "fresh_prior_rows": int(sum(r["fresh_prior_rows"] for r in result.records)),
            "exactness_records": len(result.exactness),
            "delta_records": len(result.delta),
            "trace": str(trace_path), "worker": args.worker,
            "physical_gpu": args.device_index,
        }
        with generations.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "event": "request_complete", "request": request_id,
            "sequence": sequence + 1, "of": len(chosen),
            "correct": record["correct"], "nfe": record["nfe"],
            "exact_generation_identity": record["exact_generation_identity"],
            "remaining_masks": remaining_masks,
        }), flush=True)
        del result, input_ids
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
