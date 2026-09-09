"""Bounded HF quality diagnostic for deliberately relaxed branch sharing.

This is a single-GPU quality path, not EP timing evidence.  The policy is an
impossible captured-output oracle computed online inside a forward hook; it is
used only to decide whether local errors that already fail the strict branch
gate are somehow erased by later layers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from poc_flashvep.mllm_moe_transient_branch_compression.centroid_expert_oracle import expert_forward


def make_inputs(processor, row, device, answer: bool, max_pixels: int):
    images = [Image.open(path).convert("RGB") for path in row.get("images", [])]
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": row["question"] + "\nAnswer using a single word or phrase."})
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                           tokenize=False, add_generation_prompt=True)
    kwargs = {"text": prompt, "return_tensors": "pt"}
    if images:
        kwargs.update(images=images, max_pixels=max_pixels)
    prompt_inputs = processor(**kwargs)
    prompt_length = prompt_inputs.input_ids.shape[1]
    if answer:
        kwargs["text"] = prompt + row["full_answer"] + processor.tokenizer.eos_token
        return processor(**kwargs).to(device), prompt_length
    return prompt_inputs.to(device), prompt_length


class SharingObserver:
    def __init__(self, model, processor, layers: set[int], fraction: float, policy: str):
        self.model, self.processor = model, processor
        self.layers, self.fraction, self.policy = layers, fraction, policy
        self.enabled = False
        self.routes: dict[int, torch.Tensor] = {}
        self.stats: list[dict] = []
        self.image_ids = {model.config.image_token_id, model.config.video_token_id}
        self.handles = [model.register_forward_pre_hook(self.model_pre, with_kwargs=True)]
        for layer, block in enumerate(model.model.language_model.layers):
            self.handles.append(block.mlp.experts.register_forward_pre_hook(self.expert_pre(layer)))
            self.handles.append(block.mlp.experts.register_forward_hook(self.expert_post(layer)))

    def model_pre(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            self.vision = None
            return
        flat = ids.reshape(-1)
        self.vision = torch.zeros_like(flat, dtype=torch.bool)
        for token_id in self.image_ids:
            self.vision |= flat == token_id
        self.routes = {}

    def expert_pre(self, layer):
        def hook(module, args):
            hidden, ids, weights = args
            self.routes[layer] = ids.detach().cpu()
            return None
        return hook

    def expert_post(self, layer):
        @torch.inference_mode()
        def hook(module, args, output):
            if not self.enabled or layer not in self.layers or self.vision is None or not self.vision.any():
                return None
            hidden, ids, weights = args
            visual = torch.nonzero(self.vision, as_tuple=False).flatten()
            tokens = visual.repeat_interleave(ids.shape[1])
            slots = torch.arange(ids.shape[1], device=ids.device).repeat(len(visual))
            experts = ids[tokens, slots]
            branch = expert_forward(hidden[tokens].to(torch.bfloat16), experts,
                                    module.gate_up_proj, module.down_proj).to(output.dtype)
            denom = output[tokens].float().norm(dim=1).clamp_min(1e-12)
            candidates: list[tuple[float, int, int]] = []
            if self.policy == "contribution_skip":
                score = (weights[tokens, slots, None].float() * branch.float()).norm(dim=1) / denom
                candidates = [(float(score[i]), int(i), -1) for i in range(len(score))]
            else:
                # One nearest same-expert peer per branch.  Output-oracle uses
                # true branch output; spatial uses nearest sequence position.
                for i in range(len(tokens)):
                    peer = torch.nonzero((experts == experts[i]) & (tokens != tokens[i]), as_tuple=False).flatten()
                    if not len(peer):
                        continue
                    if self.policy == "spatial_contiguous":
                        distance = (tokens[peer] - tokens[i]).abs()
                        valid = peer[distance == 1]
                        if not len(valid):
                            continue
                        j = int(valid[0])
                        score = float((weights[tokens[i], slots[i]] *
                                       (branch[j] - branch[i])).float().norm() / denom[i])
                    elif self.policy == "output_oracle_share":
                        error = (branch[peer].float() - branch[i].float()).norm(dim=1)
                        local = int(torch.argmin(error))
                        j = int(peer[local])
                        score = float(weights[tokens[i], slots[i]] * error[local] / denom[i])
                    else:
                        raise ValueError(self.policy)
                    candidates.append((score, i, j))
            candidates.sort()
            target = max(1, round(len(tokens) * self.fraction))
            selected, protected = set(), set()
            changed = output.clone()
            error_norm = []
            for score, i, j in candidates:
                key = (int(tokens[i]), int(slots[i]))
                if key in selected or key in protected:
                    continue
                old = branch[i]
                if j < 0:
                    new = torch.zeros_like(old)
                else:
                    rep_key = (int(tokens[j]), int(slots[j]))
                    if rep_key in selected:
                        continue
                    protected.add(rep_key)
                    new = branch[j]
                changed[tokens[i]] += weights[tokens[i], slots[i]].to(output.dtype) * (new - old)
                selected.add(key)
                error_norm.append(score)
                if len(selected) >= target:
                    break
            relative = (changed.float() - output.float()).norm(dim=1) / output.float().norm(dim=1).clamp_min(1e-12)
            affected = relative > 0
            self.stats.append({"layer": layer, "policy": self.policy, "target_fraction": self.fraction,
                               "achieved_fraction": len(selected) / max(len(tokens), 1),
                               "affected_tokens": int(affected.sum()),
                               "affected_rel_l2_median": float(relative[affected].median()) if affected.any() else 0.0,
                               "affected_rel_l2_max": float(relative[affected].max()) if affected.any() else 0.0,
                               "oracle_score_max": max(error_norm, default=0.0)})
            return changed
        return hook


def exact_match(prediction: str, answers: list[str]) -> bool:
    norm = lambda text: " ".join(text.lower().strip().split())
    return any(norm(answer) in norm(prediction) for answer in answers)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--dataset", default="gqa")
    parser.add_argument("--layers", default="44")
    parser.add_argument("--fraction", type=float, default=.01)
    parser.add_argument("--policy", choices=["output_oracle_share", "spatial_contiguous", "contribution_skip"],
                        default="output_oracle_share")
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("This task requires CUDA_VISIBLE_DEVICES=4,5,6,7")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(7171)
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    rows = [row for row in rows if row["dataset"] == args.dataset and row["split"] == "calibration"]
    random.Random(7171).shuffle(rows)
    rows = rows[:args.limit]
    started = time.time()
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map=f"cuda:{args.gpu}").eval()
    processor = AutoProcessor.from_pretrained(args.model)
    observer = SharingObserver(model, processor, set(map(int, args.layers.split(","))),
                               args.fraction, args.policy)
    records = []
    for row in rows:
        inputs, prompt_length = make_inputs(processor, row, model.device, True, args.max_pixels)
        observer.enabled = False
        baseline = model(**inputs, use_cache=False).logits.float()
        baseline_routes = {key: value.clone() for key, value in observer.routes.items()}
        observer.stats = []
        observer.enabled = True
        modified = model(**inputs, use_cache=False).logits.float()
        modified_routes = {key: value.clone() for key, value in observer.routes.items()}
        kl = F.kl_div(modified.log_softmax(-1), baseline.softmax(-1), reduction="batchmean")
        route_rows = []
        for layer in sorted(set(baseline_routes) & set(modified_routes)):
            route_rows.append(float((baseline_routes[layer] == modified_routes[layer]).all(dim=1).float().mean()))
        observer.enabled = False
        prompt_inputs, _ = make_inputs(processor, row, model.device, False, args.max_pixels)
        base_ids = model.generate(**prompt_inputs, max_new_tokens=8, do_sample=False)
        observer.enabled = True
        mod_ids = model.generate(**prompt_inputs, max_new_tokens=8, do_sample=False)
        base_text = processor.tokenizer.decode(base_ids[0, prompt_length:], skip_special_tokens=True)
        mod_text = processor.tokenizer.decode(mod_ids[0, prompt_length:], skip_special_tokens=True)
        records.append({"request_id": row["id"], "dataset": row["dataset"], "policy": args.policy,
                        "fraction": args.fraction, "layers": args.layers,
                        "logit_rel_l2": float((modified - baseline).norm() / baseline.norm().clamp_min(1e-12)),
                        "logit_cosine": float(F.cosine_similarity(modified.flatten(), baseline.flatten(), dim=0)),
                        "logit_kl": float(kl), "first_token_equal": bool((modified[:, 0].argmax(-1) == baseline[:, 0].argmax(-1)).all()),
                        "next_route_agreement_mean": float(np.mean(route_rows)) if route_rows else None,
                        "greedy_exact": bool(torch.equal(base_ids[:, prompt_length:], mod_ids[:, prompt_length:])),
                        "baseline_text": base_text, "modified_text": mod_text,
                        "baseline_answer_match": exact_match(base_text, row["answers"]),
                        "modified_answer_match": exact_match(mod_text, row["answers"]),
                        "intervention": observer.stats})
        print(json.dumps({key: records[-1][key] for key in ("request_id", "first_token_equal", "greedy_exact",
                                                            "next_route_agreement_mean")}), flush=True)
    (args.output_dir / "quality.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (args.output_dir / "environment.json").write_text(json.dumps({"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "physical_gpu": 4 + args.gpu, "runtime": "HF_SINGLE_GPU_QUALITY_ONLY_NOT_EP_TIMING",
        "arguments": vars(args) | {"data": str(args.data), "output_dir": str(args.output_dir)},
        "elapsed_seconds": time.time() - started}, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
