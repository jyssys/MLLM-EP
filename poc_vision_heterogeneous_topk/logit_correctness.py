#!/usr/bin/env python3
"""Full-model logit diagnostic for heterogeneous Top-K policies.

This deliberately keeps all eight expert computations and only masks their
router weights before accumulation.  It is a correctness diagnostic, not a
runtime implementation or performance measurement.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poc_vision_heterogeneous_topk.analyze_policy_oracle import budget_keep


MODEL = ("/home/esjung/.cache/huggingface/hub/"
         "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
         "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c")
EXPECTED_VISIBLE = "4,5,6,7"


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.float()
    candidate = candidate.float()
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=0)
    rel_l2 = torch.linalg.vector_norm(candidate - reference) / torch.linalg.vector_norm(reference)
    p = torch.softmax(reference, dim=0)
    log_p = torch.log_softmax(reference, dim=0)
    log_q = torch.log_softmax(candidate, dim=0)
    top_ref = torch.topk(reference, 10).indices
    top_candidate = torch.topk(candidate, 10).indices
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "logit_cosine": float(cosine),
        "logit_relative_l2": float(rel_l2),
        "kl_reference_candidate": float(torch.sum(p * (log_p - log_q))),
        "greedy_reference": int(torch.argmax(reference)),
        "greedy_candidate": int(torch.argmax(candidate)),
        "greedy_equal": bool(torch.argmax(reference) == torch.argmax(candidate)),
        "top10_overlap": len(set(top_ref.tolist()) & set(top_candidate.tolist())) / 10,
    }


def prepare(processor: Any, sample: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    from PIL import Image

    image = Image.open(sample["source_path"]).convert("RGB").resize((448, 448))
    content = [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe all visually important details, including text, numbers, and spatial relationships."},
    ]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    image_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    vision = inputs["input_ids"].eq(image_id).reshape(-1)
    if not bool(vision.any()):
        raise AssertionError((sample["sample_id"], "no visual tokens"))
    return inputs, vision


def install_policy(model: Any, state: dict[str, Any]) -> None:
    layers = model.model.language_model.layers
    for layer_index, layer in enumerate(layers):
        mlp = layer.mlp
        if hasattr(mlp, "_hetero_original_forward"):
            continue
        mlp._hetero_original_forward = mlp.forward

        def patched(self: Any, hidden_states: torch.Tensor, *, _layer: int = layer_index) -> torch.Tensor:
            batch, sequence, hidden = hidden_states.shape
            flat = hidden_states.view(-1, hidden)
            _, weights, ids = self.gate(flat)
            policy = state["policy"]
            if policy != "stock" and sequence > 1:
                vision = state["vision_mask"]
                if vision.numel() != flat.shape[0]:
                    raise AssertionError((_layer, vision.shape, flat.shape))
                keep = budget_keep(
                    ids.detach().cpu().numpy(), weights.detach().float().cpu().numpy(),
                    np.where(vision.detach().cpu().numpy(), "vision", "text"),
                    float(state["budget"]), policy, 20260911 + _layer)
                keep_gpu = torch.as_tensor(keep, device=weights.device, dtype=torch.bool)
                before = torch.bincount((ids // 32).reshape(-1), minlength=4)
                after = torch.bincount((ids[keep_gpu] // 32).reshape(-1), minlength=4)
                vision_2d = vision[:, None].expand_as(weights)
                dropped = (~keep_gpu) & vision_2d
                state["stats"].append({
                    "layer": _layer,
                    "vision_assignments": int(vision_2d.sum()),
                    "dropped": int(dropped.sum()),
                    "dropped_router_mass": float((weights * dropped).sum().float()),
                    "vision_router_mass": float((weights * vision_2d).sum().float()),
                    "max_rank_before": int(before.max()),
                    "max_rank_after": int(after.max()),
                })
                weights = weights * keep_gpu
            output = self.experts(flat, ids, weights)
            return output.reshape(batch, sequence, hidden)

        mlp.forward = types.MethodType(patched, mlp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--samples", nargs="+", default=["camera_e448", "cell_e448", "chessboard_RGB_e448"])
    parser.add_argument("--budgets", type=float, nargs="+", default=[.01, .05, .1, .2])
    parser.add_argument("--generate-tokens", type=int, default=0)
    args = parser.parse_args()

    import os
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != EXPECTED_VISIBLE:
        raise RuntimeError(f"refusing GPU execution: CUDA_VISIBLE_DEVICES={visible!r}")

    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    manifest = json.loads(args.manifest.read_text())
    sample_map = {row["sample_id"]: row for row in manifest["samples"]}
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa").to("cuda:0").eval()
    state: dict[str, Any] = {"policy": "stock", "budget": 0.0,
                             "vision_mask": None, "stats": []}
    install_policy(model, state)
    rows = []
    conditions = [("stock", 0.0)] + [
        (policy, budget) for budget in args.budgets for policy in ("semantic", "tail")]
    with torch.inference_mode():
        for sample_id in args.samples:
            inputs, vision = prepare(processor, sample_map[sample_id])
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            state["vision_mask"] = vision.to("cuda:0")
            reference = None
            reference_sequence = None
            for policy, budget in conditions:
                state.update(policy=policy, budget=budget, stats=[])
                output = model(**inputs, use_cache=False, logits_to_keep=1)
                logits = output.logits[0, -1].detach().float().cpu()
                if reference is None:
                    reference = logits
                local = metrics(reference, logits)
                stats = list(state["stats"])
                if args.generate_tokens:
                    # The generation prefill repeats the same policy. Keep its
                    # accounting separate from the one-pass logit diagnostic.
                    state["stats"] = []
                    generated = model.generate(
                        **inputs, max_new_tokens=args.generate_tokens,
                        do_sample=False, use_cache=True)
                    sequence = generated[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
                    if reference_sequence is None:
                        reference_sequence = sequence
                    local.update({
                        "generated_token_ids": sequence,
                        "short_greedy_exact": sequence == reference_sequence,
                    })
                local.update({
                    "sample": sample_id, "policy": policy, "budget": budget,
                    "prompt_tokens": int(inputs["input_ids"].numel()),
                    "vision_tokens": int(vision.sum()), "layers": len(stats),
                    "assignment_drop_fraction": (sum(x["dropped"] for x in stats) /
                                                   max(sum(x["vision_assignments"] for x in stats), 1)),
                    "router_mass_drop_fraction": (sum(x["dropped_router_mass"] for x in stats) /
                                                   max(sum(x["vision_router_mass"] for x in stats), 1e-12)),
                    "max_rank_reduction_median": (float(np.median([
                        1 - x["max_rank_after"] / max(x["max_rank_before"], 1) for x in stats
                    ])) if stats else 0.0),
                })
                rows.append(local)
                del output, logits
            del inputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = []
    for policy, budget in conditions:
        selected = [row for row in rows if row["policy"] == policy and row["budget"] == budget]
        summary.append({
            "policy": policy, "budget": budget, "samples": len(selected),
            "greedy_match_fraction": sum(row["greedy_equal"] for row in selected) / len(selected),
            "logit_relative_l2_median": float(np.median([row["logit_relative_l2"] for row in selected])),
            "logit_kl_median": float(np.median([row["kl_reference_candidate"] for row in selected])),
            "logit_cosine_median": float(np.median([row["logit_cosine"] for row in selected])),
            "top10_overlap_median": float(np.median([row["top10_overlap"] for row in selected])),
            "max_rank_reduction_median": float(np.median([row["max_rank_reduction_median"] for row in selected])),
            "short_greedy_match_fraction": (
                sum(row.get("short_greedy_exact", True) for row in selected) / len(selected)
                if args.generate_tokens else None),
        })
    args.output.with_suffix(".summary.json").write_text(json.dumps({
        "scope": "HF_FULL_MODEL_WEIGHT_MASK_CORRECTNESS_ONLY",
        "performance_claim": False,
        "visible_devices": visible,
        "model": args.model,
        "policies": summary,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
