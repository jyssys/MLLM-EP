#!/usr/bin/env python3
"""Validate composed semantic-oracle schedules in the full Qwen3-VL model.

This is a quality experiment: masked branches are still executed by the eager
expert implementation.  It reports label-conditioned teacher-forced metrics,
greedy benchmark answers, and routing/load consequences, but never presents
the run time as an optimization speedup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from poc_semantic_first_topk.semantic_sensitivity import (
    EXPECTED_VISIBLE,
    GroupMask,
    load_records,
    make_inputs,
)


def append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.%+-]+", " ", text.lower()).split())


def benchmark_correct(prediction: str, answers: list[str]) -> bool:
    pred = normalize(prediction)
    return any(normalize(str(answer)) == pred for answer in answers)


def aggregate_stats(stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats:
        return {"assignment_drop_fraction": 0.0, "max_rank_reduction_median": 0.0}
    assignments = sum(int(row["assignments"]) for row in stats)
    dropped = sum(int(row["dropped"]) for row in stats)
    reductions = [1 - row["max_after"] / max(row["max_before"], 1) for row in stats]
    result = {
        "assignment_drop_fraction": dropped / max(assignments, 1),
        "max_rank_reduction_median": float(np.median(reductions)),
        "max_rank_reduction_p10": float(np.quantile(reductions, .1)),
        "max_rank_reduction_p90": float(np.quantile(reductions, .9)),
    }
    if all("chosen_k" in row for row in stats):
        result["chosen_k_counts"] = np.asarray(
            [row["chosen_k"] for row in stats], dtype=np.int64).sum(axis=0).tolist()
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--schedules", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int, choices=range(4))
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--fractions", type=float, nargs="+", default=[.1, .2, .3])
    parser.add_argument("--use-global", action="store_true",
                        help="Use the calibration-aggregate schedule on unseen requests")
    parser.add_argument("--datasets", default="gqa,chartqa")
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--limit-per-dataset", type=int, default=16)
    parser.add_argument("--generate-tokens", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--experts", choices=["grouped_mm", "eager"], default="eager")
    parser.add_argument("--include-ep-refine", action="store_true")
    parser.add_argument("--only-ep-refine", action="store_true",
                        help="Skip repeated semantic full/coarse policies")
    parser.add_argument("--ep-risk-slack", type=float, default=.05)
    parser.add_argument("--fixed-ks", type=int, nargs="*", default=[])
    parser.add_argument("--router-thresholds", type=float, nargs="*", default=[])
    parser.add_argument("--contribution-fractions", type=float, nargs="*", default=[])
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError(f"illegal visibility: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)

    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    schedule_document = json.loads(args.schedules.read_text())
    schedules = schedule_document["requests"]
    if args.use_global:
        rows = load_records(args.data, args.datasets, args.split, 0, args.shard,
                            args.shards, args.limit_per_dataset)
    else:
        all_rows = [json.loads(line) for line in args.data.read_text().splitlines()]
        rows = [row for row in all_rows if row["id"] in schedules]
        rows = sorted(rows, key=lambda row: row["id"])[args.shard::args.shards]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"quality_{args.shard}.jsonl"
    output.write_text("")
    started = time.time()
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation=args.experts, low_cpu_mem_usage=True,
    ).to(f"cuda:{args.gpu}").eval()
    processor = AutoProcessor.from_pretrained(args.model)
    mask = GroupMask(model, processor, 16)
    mask.ep_risk_slack = args.ep_risk_slack

    for request_index, row in enumerate(rows):
        labeled, prompt_length = make_inputs(processor, row, model.device, True, args.max_pixels)
        targets = labeled["input_ids"][:, prompt_length:]
        prompt, _ = make_inputs(processor, row, model.device, False, args.max_pixels)
        mask.mode = "stock"; mask.stats = []
        reference_logits = model(**labeled, use_cache=False).logits[:, prompt_length - 1:-1].float()
        reference_prob = reference_logits.softmax(-1)
        reference_nll = float(F.cross_entropy(
            reference_logits.reshape(-1, reference_logits.shape[-1]), targets.reshape(-1)))
        reference_ids = model.generate(**prompt, max_new_tokens=args.generate_tokens,
                                       do_sample=False, use_cache=True)
        reference_new = reference_ids[:, prompt["input_ids"].shape[1]:]
        reference_text = processor.tokenizer.decode(reference_new[0], skip_special_tokens=True).strip()
        conditions: list[tuple[str, str, list[int] | float | int | None]] = [
            ("stock", "stock", None)
        ]
        for fraction in args.fractions:
            source = (schedule_document["global_policies"] if args.use_global else
                      schedules[row["id"]]["policies"])
            if not args.only_ep_refine:
                for grid in ("full", "coarse"):
                    key = f"semantic_{grid}_{fraction:g}"
                    conditions.append((key, "schedule", source[key]["group_k"]))
            if args.include_ep_refine:
                key = f"semantic_full_{fraction:g}"
                conditions.append((f"ep_refined_{fraction:g}_s{args.ep_risk_slack:g}",
                                   "schedule_ep_refine", source[key]["group_k"]))
        conditions.extend((f"fixed_vision_k{k}", "fixed_vision", k) for k in args.fixed_ks)
        conditions.extend((f"router_mass_t{threshold:g}", "router_mass", threshold)
                          for threshold in args.router_thresholds)
        conditions.extend((f"contribution_f{fraction:g}", "contribution_oracle", fraction)
                          for fraction in args.contribution_fractions)
        for policy, mode, parameter in conditions:
            mask.mode = mode
            if mode in {"schedule", "schedule_ep_refine"}:
                mask.group_k = parameter
            elif mode == "fixed_vision":
                mask.fixed_k = int(parameter)
            elif mode == "router_mass":
                mask.router_threshold = float(parameter)
            elif mode == "contribution_oracle":
                mask.target_fraction = float(parameter)
            mask.stats = []
            logits = model(**labeled, use_cache=False).logits[:, prompt_length - 1:-1].float()
            stats = list(mask.stats)
            kl = F.kl_div(logits.log_softmax(-1), reference_prob, reduction="sum") / max(targets.numel(), 1)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            mask.stats = []
            generated = model.generate(**prompt, max_new_tokens=args.generate_tokens,
                                       do_sample=False, use_cache=True)
            new = generated[:, prompt["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(new[0], skip_special_tokens=True).strip()
            record = {
                "request_id": row["id"], "dataset": row["dataset"], "policy": policy,
                "answer_tokens": int(targets.numel()), "reference_nll": reference_nll,
                "kl_per_answer_token": float(kl.clamp_min(0)), "nll_delta": float(nll - reference_nll),
                "first_answer_token_equal": bool(logits[:, 0].argmax(-1).eq(
                    reference_logits[:, 0].argmax(-1)).all()),
                "teacher_forced_greedy_equal": float(logits.argmax(-1).eq(reference_logits.argmax(-1)).float().mean()),
                "short_greedy_exact": bool(torch.equal(new, reference_new)),
                "prediction": text, "reference_prediction": reference_text,
                "benchmark_correct": benchmark_correct(text, row["answers"]),
                "reference_benchmark_correct": benchmark_correct(reference_text, row["answers"]),
                **aggregate_stats(stats),
            }
            append(output, record)
        print(json.dumps({"shard": args.shard, "request": row["id"],
                          "done": request_index + 1, "total": len(rows)}), flush=True)
    (args.output / f"completed_{args.shard}.json").write_text(json.dumps({
        "requests": len(rows), "elapsed_seconds": time.time() - started,
        "physical_gpu": 4 + args.gpu, "visible": EXPECTED_VISIBLE,
        "experts": args.experts,
        "evidence_boundary": "full-model quality only; masked expert work was still executed",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
