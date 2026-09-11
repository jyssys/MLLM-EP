#!/usr/bin/env python3
"""Offline task-conditioned spatial-group sensitivity oracle.

For each benchmark request, perturb exactly one 4x4 visual group to each
candidate integer K across all language MoE layers.  Answer-token KL/NLL is an
offline semantic oracle; it uses labels and is not deployable.  All experts are
still executed, so no performance claim is made.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from poc_semantic_first_topk.policy_core import (
    FULL_K,
    ep_same_budget_refinement,
    semantic_allocation,
)


EXPECTED_VISIBLE = "4,5,6,7"


def append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def load_records(path: Path, datasets: str, split: str, limit: int,
                 shard: int, shards: int, limit_per_dataset: int = 0) -> list[dict[str, Any]]:
    wanted = set(datasets.split(","))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row["dataset"] in wanted and row["split"] == split]
    if limit_per_dataset:
        balanced = []
        for dataset in sorted(wanted):
            part = [row for row in rows if row["dataset"] == dataset]
            random.Random(92731).shuffle(part)
            balanced.extend(part[:limit_per_dataset])
        rows = balanced
    else:
        random.Random(92731).shuffle(rows)
        rows = rows[:limit]
    return rows[shard::shards]


def make_inputs(processor: Any, row: dict[str, Any], device: torch.device,
                answer: bool, max_pixels: int) -> tuple[dict[str, torch.Tensor], int]:
    images = [Image.open(path).convert("RGB") for path in row.get("images", [])]
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": row["question"] +
                    "\nAnswer the question using a single word or phrase."})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)
    options: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
    if images:
        options.update(images=images, max_pixels=max_pixels)
    prompt_inputs = processor(**options)
    prompt_length = int(prompt_inputs.input_ids.shape[1])
    if answer:
        options["text"] = prompt + row["full_answer"] + processor.tokenizer.eos_token
    return {key: value.to(device) for key, value in processor(**options).items()}, prompt_length


class GroupMask:
    def __init__(self, model: Any, processor: Any, groups: int):
        self.model, self.processor, self.groups = model, processor, groups
        self.mode = "stock"
        self.target_group = -1
        self.target_k = 8
        self.group_k: list[int] | None = None
        self.ep_risk_slack = 0.05
        self.fixed_k = 8
        self.router_threshold = 1.0
        self.target_fraction = 0.0
        self.group_ids: torch.Tensor | None = None
        self.vision: torch.Tensor | None = None
        self.stats: list[dict[str, Any]] = []
        self.handles = [model.register_forward_pre_hook(self.model_pre, with_kwargs=True)]
        for index, layer in enumerate(model.model.language_model.layers):
            original = layer.mlp.forward

            def patched(module: Any, hidden_states: torch.Tensor, *,
                        _layer: int = index, _original=original) -> torch.Tensor:
                if self.mode == "stock" or hidden_states.shape[1] == 1:
                    return _original(hidden_states)
                batch, sequence, hidden = hidden_states.shape
                flat = hidden_states.reshape(-1, hidden)
                _, weights, ids = module.gate(flat)
                if self.group_ids is None or self.group_ids.numel() != len(flat):
                    raise RuntimeError((_layer, "group identity mismatch", sequence,
                                        None if self.group_ids is None else self.group_ids.numel()))
                k = torch.full((len(flat),), 8, dtype=torch.long, device=flat.device)
                if self.mode == "single_group":
                    k[self.group_ids == self.target_group] = self.target_k
                elif self.mode == "fixed_vision":
                    k[self.vision] = int(self.fixed_k)
                    refinement_swaps = 0
                elif self.mode == "router_mass":
                    cumulative = weights.float().cumsum(dim=-1)
                    chosen = (cumulative < float(self.router_threshold)).sum(dim=-1) + 1
                    k[self.vision] = chosen[self.vision].clamp_(1, 8)
                    refinement_swaps = 0
                elif self.mode in {"schedule", "schedule_ep_refine"}:
                    if self.group_k is None:
                        raise RuntimeError("missing group schedule")
                    for group, value in enumerate(self.group_k):
                        k[self.group_ids == group] = int(value)
                    if self.mode == "schedule_ep_refine":
                        # Diagnostic only: the CPU solver consumes the current
                        # layer's already-computed routes and router weights.
                        # It preserves the exact assignment count while moving
                        # omission budget toward critical ranks.  Its wall time
                        # is not a performance measurement.
                        modality = torch.where(
                            self.vision, torch.tensor(1, device=flat.device),
                            torch.tensor(0, device=flat.device)).cpu().numpy()
                        modality = np.where(modality == 1, "vision", "text")
                        refined = ep_same_budget_refinement(
                            ids.detach().cpu().numpy(),
                            weights.detach().float().cpu().numpy(), modality,
                            k.detach().cpu().numpy(), risk_slack=self.ep_risk_slack)
                        k = torch.as_tensor(refined.k, device=flat.device, dtype=torch.long)
                        refinement_swaps = refined.swaps
                    else:
                        refinement_swaps = 0
                else:
                    if self.mode != "contribution_oracle":
                        raise ValueError(self.mode)
                    refinement_swaps = 0
                branch_outputs = None
                if self.mode == "contribution_oracle":
                    # Offline diagnostic only.  Every selected expert is
                    # evaluated before the retained prefix K is chosen, so
                    # this cannot be used as a deployable performance path.
                    experts = module.experts
                    required = ("gate_up_proj", "down_proj", "act_fn", "num_experts")
                    if not all(hasattr(experts, name) for name in required):
                        raise RuntimeError("contribution oracle requires eager expert implementation")
                    branch_outputs = torch.zeros(
                        (len(flat), ids.shape[1], hidden), dtype=flat.dtype, device=flat.device)
                    expert_mask = torch.nn.functional.one_hot(
                        ids, num_classes=int(experts.num_experts)).permute(2, 1, 0)
                    for expert_index in torch.nonzero(
                            expert_mask.sum(dim=(-1, -2)) > 0, as_tuple=False).flatten():
                        positions, tokens = torch.where(expert_mask[expert_index])
                        current = flat[tokens]
                        gate, up = F.linear(current, experts.gate_up_proj[expert_index]).chunk(2, -1)
                        current = experts.act_fn(gate) * up
                        current = F.linear(current, experts.down_proj[expert_index])
                        branch_outputs[tokens, positions] = current
                    weighted = branch_outputs.float() * weights.float().unsqueeze(-1)
                    contribution = weighted.norm(dim=-1)
                    denominator = weighted.sum(dim=1).norm(dim=-1).clamp_min(1e-12)
                    risk = (contribution / denominator[:, None]).cpu().numpy()
                    modality = np.where(self.vision.cpu().numpy(), "vision", "text")
                    possible = int(self.vision.sum()) * ids.shape[1]
                    allocation = semantic_allocation(
                        risk, modality, round(possible * float(self.target_fraction)), FULL_K)
                    k = torch.as_tensor(allocation.k, device=flat.device, dtype=torch.long)
                keep = torch.arange(8, device=flat.device)[None, :] < k[:, None]
                before = torch.bincount((ids // 32).reshape(-1), minlength=4)
                after = torch.bincount((ids[keep] // 32).reshape(-1), minlength=4)
                self.stats.append({"layer": _layer, "dropped": int((~keep).sum()),
                                   "assignments": int(keep.numel()),
                                   "max_before": int(before.max()), "max_after": int(after.max()),
                                   "chosen_k": torch.bincount(k, minlength=9)[1:9].cpu().tolist(),
                                   "refinement_swaps": refinement_swaps if self.mode == "schedule_ep_refine" else 0})
                if branch_outputs is None:
                    output = module.experts(flat, ids, weights * keep)
                else:
                    output = (branch_outputs * (weights * keep).unsqueeze(-1)).sum(dim=1)
                return output.reshape(batch, sequence, hidden)

            layer.mlp.forward = types.MethodType(patched, layer.mlp)

    def model_pre(self, module: Any, args: tuple, kwargs: dict[str, Any]) -> None:
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            return
        flat = ids.reshape(-1)
        vision = flat == self.model.config.image_token_id
        group_ids = torch.full_like(flat, -1)
        count = int(vision.sum())
        if count:
            grid = kwargs.get("image_grid_thw")
            if grid is None or len(grid) != 1:
                raise RuntimeError("oracle currently requires exactly one image")
            height = int(grid[0, 1]) // 2
            width = int(grid[0, 2]) // 2
            if height * width != count:
                raise RuntimeError((height, width, count))
            row = torch.arange(count, device=flat.device) // width
            col = torch.arange(count, device=flat.device) % width
            side = int(round(self.groups ** .5))
            local_groups = ((row * side // height) * side + (col * side // width)).long()
            group_ids[vision] = local_groups
        self.group_ids, self.vision = group_ids, vision


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int, choices=range(4))
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--datasets", default="gqa,chartqa")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--experts", choices=["grouped_mm", "eager"], default="grouped_mm")
    parser.add_argument("--stock-repeats", type=int, default=2)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE:
        raise RuntimeError(f"illegal visibility: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if int(round(args.groups ** .5)) ** 2 != args.groups:
        raise ValueError("groups must be a square")
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation=args.experts, low_cpu_mem_usage=True
    ).to(f"cuda:{args.gpu}").eval()
    processor = AutoProcessor.from_pretrained(args.model)
    mask = GroupMask(model, processor, args.groups)
    rows = load_records(args.data, args.datasets, args.split, args.limit,
                        args.shard, args.shards, args.limit_per_dataset)
    output = args.output / f"sensitivity_{args.shard}.jsonl"
    output.write_text("")
    for request_index, row in enumerate(rows):
        inputs, prompt = make_inputs(processor, row, model.device, True, args.max_pixels)
        targets = inputs["input_ids"][:, prompt:]
        mask.mode = "stock"; mask.stats = []
        reference = model(**inputs, use_cache=False).logits[:, prompt - 1:-1].float()
        reference_prob = reference.softmax(-1)
        baseline_nll = float(F.cross_entropy(reference.reshape(-1, reference.shape[-1]),
                                             targets.reshape(-1), reduction="mean"))
        append(output, {"request_id": row["id"], "dataset": row["dataset"],
                        "group": -1, "k": 8, "answer_tokens": int(targets.numel()),
                        "kl_per_answer_token": 0.0, "nll_delta": 0.0,
                        "baseline_nll": baseline_nll})
        for repeat in range(args.stock_repeats):
            repeated = model(**inputs, use_cache=False).logits[:, prompt - 1:-1].float()
            repeat_kl = F.kl_div(repeated.log_softmax(-1), reference_prob,
                                 reduction="sum") / max(targets.numel(), 1)
            repeat_nll = F.cross_entropy(repeated.reshape(-1, repeated.shape[-1]),
                                         targets.reshape(-1), reduction="mean")
            append(output, {"request_id": row["id"], "dataset": row["dataset"],
                            "group": -2, "repeat": repeat, "k": 8,
                            "answer_tokens": int(targets.numel()),
                            "kl_per_answer_token": float(repeat_kl.clamp_min(0)),
                            "nll_delta": float(repeat_nll - baseline_nll),
                            "baseline_nll": baseline_nll})
        for group in range(args.groups):
            group_size = int((mask.group_ids == group).sum())
            for k in args.ks:
                mask.mode = "single_group"; mask.target_group = group; mask.target_k = k; mask.stats = []
                logits = model(**inputs, use_cache=False).logits[:, prompt - 1:-1].float()
                kl = F.kl_div(logits.log_softmax(-1), reference_prob,
                              reduction="sum") / max(targets.numel(), 1)
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      targets.reshape(-1), reduction="mean")
                append(output, {"request_id": row["id"], "dataset": row["dataset"],
                                "group": group, "group_size": group_size, "k": k,
                                "answer_tokens": int(targets.numel()),
                                "kl_per_answer_token": float(kl.clamp_min(0)),
                                "nll_delta": float(nll - baseline_nll),
                                "baseline_nll": baseline_nll,
                                "assignment_drop_fraction": sum(x["dropped"] for x in mask.stats) /
                                    max(sum(x["assignments"] for x in mask.stats), 1),
                                "max_rank_reduction_median": float(torch.tensor([
                                    1 - x["max_after"] / max(x["max_before"], 1) for x in mask.stats
                                ]).median())})
        print(json.dumps({"shard": args.shard, "request": row["id"],
                          "done": request_index + 1, "total": len(rows)}), flush=True)
    (args.output / f"completed_{args.shard}.json").write_text(json.dumps({
        "requests": len(rows), "elapsed_seconds": time.time() - started,
        "physical_gpu": 4 + args.gpu, "visible": EXPECTED_VISIBLE,
        "evidence_boundary": "label-conditioned full-model offline semantic oracle; all experts execute",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
