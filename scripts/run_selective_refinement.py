#!/usr/bin/env python3
"""Run bounded dense-emulation selective-refinement trajectories on GPU0/1."""

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
from selective_refinement.emulator import generate_selective, save_request_trace
from selective_refinement.policy import POLICIES, PolicyConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--semantics", choices=("none", "f0", "f1"), required=True)
    parser.add_argument("--active-ratio", type=float, required=True)
    parser.add_argument("--max-freeze-age", type=int, default=1)
    parser.add_argument("--periodic-full-refresh", type=int, default=0)
    parser.add_argument("--target-ep", type=int, choices=(4, 8), default=8)
    parser.add_argument("--lambda-max", type=float, default=0.5)
    parser.add_argument("--lambda-cv", type=float, default=0.25)
    parser.add_argument("--age-weight", type=float, default=0.1)
    parser.add_argument("--gen-length", type=int, default=2048)
    parser.add_argument("--parity-official", action="store_true")
    parser.add_argument("--record-label", default=None,
                        help="unique JSONL label for manually rebalanced workers")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("only physical GPUs 0,1 may be exposed")
    if args.worker >= args.workers:
        raise ValueError("worker must be smaller than workers")
    if args.gen_length % 32:
        raise ValueError("generation length must be block aligned")
    policy = PolicyConfig(
        name=args.policy, active_ratio=args.active_ratio,
        max_freeze_age=args.max_freeze_age,
        periodic_full_refresh=args.periodic_full_refresh,
        target_ep=args.target_ep, lambda_max=args.lambda_max,
        lambda_cv=args.lambda_cv, age_weight=args.age_weight,
    )
    if args.active_ratio == 1.0 and args.policy != "full":
        raise ValueError("active_ratio=1 parity runs must use policy=full")
    if args.source.suffix == ".jsonl":
        captured = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
        rows = [{
            "id": int(row["sample_id"]),
            "question": row["raw_prompt"],
            "raw_prompt": row["raw_prompt"],
            "answer": f"#### {row['ground_truth']}",
        } for row in captured]
    else:
        rows = samples("gsm8k", args.source, "dinfer-fourshot", args.yaml)
    by_id = {int(row["id"]): row for row in rows}
    chosen = [by_id[sample_id] for sample_id in args.sample_ids]
    chosen = [row for row in chosen if int(row["id"]) % args.workers == args.worker]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    label = args.record_label or f"worker{args.worker}"
    generations_path = args.output_dir / f"generations_{label}.jsonl"
    completed = {int(path.stem.split("_")[-1]) for path in trace_dir.glob("request_*.npz")}

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, revision=REVISION, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(json.dumps({
        "event": "worker_ready", "worker": args.worker,
        "physical_gpu": args.device_index, "requests": len(chosen),
        "policy": policy.__dict__, "semantics": args.semantics,
    }), flush=True)
    for sequence, row in enumerate(chosen):
        request_id = int(row["id"])
        target = trace_dir / f"request_{request_id:04d}.npz"
        if request_id in completed:
            continue
        raw_prompt = row.get("raw_prompt", row["question"])
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        seed = 20260915 + request_id
        reference = None
        if args.parity_official:
            torch.manual_seed(seed)
            with torch.inference_mode():
                reference = model.generate(
                    inputs=input_ids, temperature=0.0, threshold=0.95,
                    block_length=32, steps=32, gen_length=args.gen_length,
                    eos_early_stop=True,
                )
        torch.manual_seed(seed)
        result = generate_selective(
            model, input_ids, policy=policy, semantics=args.semantics,
            threshold=0.95, block_length=32, steps=32,
            gen_length=args.gen_length, eos_early_stop=True,
            eos_id=tokenizer.eos_token_id, mask_id=tokenizer.mask_token_id,
            request_id=request_id,
            # The contract caps the missing hidden-state audit at 16 fixed
            # requests. Policy rollouts still retain their F0/F1 cache-drift
            # diagnostics, but do not duplicate adjacent hidden captures.
            capture_adjacent_hidden=(args.semantics == "none" and request_id < 16),
        )
        parity = None if reference is None else bool(torch.equal(reference, result.generated))
        if args.parity_official and parity is not True:
            raise RuntimeError(f"modified full-active loop parity failed on request {request_id}")
        generated_ids, returned_prompt = normalize_generated_ids(result.generated, input_ids)
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, parser_rule = parse_gsm8k(output)
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        active_updates = sum(int(np.count_nonzero(r["masked"] & r["active"]))
                             for r in result.records)
        masked_updates = sum(int(np.count_nonzero(r["masked"])) for r in result.records)
        forced_updates = sum(int(np.count_nonzero(r["forced"])) for r in result.records)
        accepted = [int(np.count_nonzero(r["transfer"])) for r in result.records]
        remaining_masks = generated_ids.count(tokenizer.mask_token_id)
        termination = (
            "REMAINING_MASK" if remaining_masks else
            "EOS" if tokenizer.eos_token_id in generated_ids else "GEN_LENGTH_CAP"
        )
        metadata = {
            "schema_version": 1, "model": "inclusionAI/LLaDA2.0-mini",
            "revision": REVISION, "request_id": request_id,
            "threshold": 0.95, "block_length": 32, "steps": 32,
            "gen_length": args.gen_length, "policy": policy.__dict__,
            "semantics": args.semantics,
            "dense_emulation_no_speedup_claim": True,
            "adjacent_hidden_capture": args.semantics == "none" and request_id < 16,
            "source_partition_after_selection": "preserve_original_balanced_source_rank",
            "routed_layer_ids": [
                i for i, layer in enumerate(model.model.layers) if hasattr(layer.mlp, "gate")
            ],
        }
        save_request_trace(target, result, metadata)
        record = {
            "sample_id": request_id, "correct": parsed == gold,
            "parsed_answer": parsed, "ground_truth": gold,
            "parser_rule": parser_rule, "output": output,
            "output_ids": generated_ids, "generation_tokens": len(generated_ids),
            "termination": termination, "remaining_masks": remaining_masks,
            "nfe": len(result.records),
            "blocks": len({int(r["block_id"]) for r in result.records}),
            "active_updates": active_updates, "masked_updates": masked_updates,
            "active_work_reduction_vs_same_trajectory": (
                1 - active_updates / masked_updates if masked_updates else 0
            ),
            "forced_updates": forced_updates,
            "accepted_mean": float(np.mean(accepted)) if accepted else 0,
            "accepted_total": int(sum(accepted)),
            "max_observed_freeze_age": max(
                (int(row["freeze_age"]) for row in result.drift), default=0
            ),
            "official_parity_checked": args.parity_official,
            "official_parity": parity,
            "returned_prompt": returned_prompt,
            "trace": str(target), "worker": args.worker,
            "physical_gpu": args.device_index, "policy": policy.__dict__,
            "semantics": args.semantics,
        }
        with generations_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "event": "request_complete", "request": request_id,
            "sequence": sequence + 1, "of": len(chosen),
            "correct": record["correct"], "nfe": record["nfe"],
            "work_reduction": record["active_work_reduction_vs_same_trajectory"],
            "parity": parity,
        }), flush=True)
        del reference, result, input_ids
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
