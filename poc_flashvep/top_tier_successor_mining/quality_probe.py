"""Shared fresh Qwen quality/predictor/calibration diagnostics.

HF quality replicas are explicitly NOT EP serving performance evidence. The
patched boundary preserves vanilla model/attention/cache forward implementations.
"""
import argparse
import importlib.util
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from policies import libra_predict, modes_weights, sere_route, set_overlap


class Observer:
    def __init__(self, model, processor, out):
        self.model, self.processor, self.out = model, processor, out
        self.layers = model.model.language_model.layers
        self.special = torch.tensor(processor.tokenizer.all_special_ids, device=model.device)
        self.image_ids = [model.config.image_token_id, model.config.video_token_id]
        self.mode, self.record, self.capture_hidden = "vanilla", False, False
        self.defer_skip_counts = False
        self.save_hidden_layers = set()
        self.predictions, self.rows, self.hiddens = {}, [], {}
        self.context = {}
        self.similarity = None
        self.alpha = None
        self.retain, self.rho = 2, 0.5
        self.policy_phase = "all"
        self.tau_text, self.tau_vision = 0.0, 0.0
        self.ablate_layer, self.ablate_modality = -1, "text"
        self.ablation_semantics = "paper_zero"
        self.handles = [model.register_forward_pre_hook(self.model_pre, with_kwargs=True)]
        for l, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.experts.register_forward_pre_hook(self.expert_pre(l)))
            self.handles.append(layer.mlp.experts.register_forward_hook(self.expert_post(l)))

    def model_pre(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise RuntimeError("Quality observer requires input_ids to identify modalities")
        sequence_length = ids.shape[-1]
        ids = ids.reshape(-1)
        self.valid_mask = ids != self.processor.tokenizer.pad_token_id
        self.vision_mask = (ids == self.image_ids[0]) | (ids == self.image_ids[1])
        self.text_mask = ~torch.isin(ids, self.special)
        self.phase = "decode" if kwargs.get("past_key_values") is not None and sequence_length == 1 else "prefill"
        self.predictions = {}

    def expert_pre(self, layer):
        def hook(module, args):
            hidden, ids, weights = args
            if hidden.shape[0] != self.text_mask.numel():
                raise RuntimeError("Token identity mismatch at expert boundary")
            if self.capture_hidden:
                self.hiddens[layer] = hidden.detach().cpu()
            if self.record:
                pred_prev = self.predictions.pop(layer, None)
                next_ids = next_values = None
                if layer + 1 < len(self.layers):
                    next_ids, next_values = libra_predict(hidden, self.layers[layer+1].mlp.gate.weight)
                    self.predictions[layer+1] = next_ids
                router_probs = F.linear(hidden, self.layers[layer].mlp.gate.weight).float().softmax(-1)
                overlap = set_overlap(pred_prev, ids) if pred_prev is not None else None
                for modality, mask in [("text", self.text_mask), ("vision", self.vision_mask)]:
                    if not mask.any():
                        continue
                    chosen = ids[mask]
                    hist = torch.bincount(chosen.reshape(-1), minlength=128)
                    self.rows.append({**self.context, "phase": self.phase, "layer": layer,
                                      "modality": modality, "tokens": int(mask.sum()),
                                      "libra_topk_recall": float(overlap[mask].mean()) if overlap is not None else None,
                                      "entropy": float((-(router_probs[mask].clamp_min(1e-12).log()*router_probs[mask]).sum(-1)).mean()),
                                      "top1_mass": float(weights[mask, 0].float().mean()),
                                      "primary2_experts": int(chosen[:, :2].unique().numel()),
                                      "active_experts": int((hist > 0).sum()),
                                      "expert_histogram": hist.cpu().tolist()})
                if self.context.get("save_routes"):
                    path = self.out / "routes" / f"{self.context['request_id']}_{self.phase}_l{layer:02d}.pt"
                    path.parent.mkdir(exist_ok=True)
                    route={"ids": ids.cpu(), "weights": weights.cpu(),
                                "vision": self.vision_mask.cpu(), "text": self.text_mask.cpu(),
                                "next_prediction": next_ids.cpu() if next_ids is not None else None}
                    if layer in self.save_hidden_layers:
                        route["hidden"]=hidden.detach().cpu()
                    torch.save(route, path)
            if self.mode == "sere":
                if self.similarity is None:
                    raise RuntimeError("SERE requires calibrated similarity")
                if self.policy_phase != "all" and self.policy_phase != self.phase:
                    return None
                changed = sere_route(ids, self.similarity[layer], self.retain, self.rho, self.valid_mask)
                if hasattr(self, "skip_counts"):
                    self.skip_counts[0] += int((ids != changed).sum())
                    self.skip_counts[1] += ids.numel()
                return hidden, changed, weights
            if self.mode == "modes":
                if self.alpha is None:
                    raise RuntimeError("MoDES requires calibrated importance")
                at, av = self.alpha[layer]
                new_weights, drop = modes_weights(weights, self.text_mask, self.vision_mask,
                                                  at, av, self.tau_text, self.tau_vision)
                self.skip_counts[0] += drop.sum() if self.defer_skip_counts else int(drop.sum())
                self.skip_counts[1] += drop.numel()
                # Keep valid IDs for HF quality path. Zero weights implement exact
                # baseline contribution math; timing is not claimed as fast skipping.
                return hidden, ids, new_weights
            return None
        return hook

    def expert_post(self, layer):
        def hook(module, args, output):
            if layer != self.ablate_layer:
                return None
            mask = self.text_mask if self.ablate_modality == "text" else self.vision_mask
            out = output.clone()
            if self.ablation_semantics == "official_input":
                out[mask] = args[0][mask]
            elif self.ablation_semantics == "paper_zero":
                out[mask] = 0
            else:
                raise ValueError(self.ablation_semantics)
            return out
        return hook


def load_records(path, dataset, split, limit, shard, shards):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows = [r for r in rows if r["dataset"] == dataset and r["split"] == split]
    random.Random(7317).shuffle(rows)
    return rows[:limit][shard::shards]


def make_inputs(processor, row, device, answer=False, max_pixels=512*512, instruction=True):
    images = [Image.open(p).convert("RGB") for p in row.get("images", [])]
    content = [{"type": "image", "image": image} for image in images]
    question = row["question"] + ("\nAnswer the question using a single word or phrase." if instruction else "")
    content.append({"type": "text", "text": question})
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                            tokenize=False, add_generation_prompt=True)
    options = dict(text=prompt, return_tensors="pt")
    if images:
        options.update(images=images, max_pixels=max_pixels)
    prompt_inputs = processor(**options)
    prompt_length = prompt_inputs.input_ids.shape[1]
    if answer:
        options["text"] = prompt + row["full_answer"] + processor.tokenizer.eos_token
        inputs = processor(**options)
    else:
        inputs = prompt_inputs
    return inputs.to(device), prompt_length


def append_jsonl(path, rows):
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job", choices=["smoke", "capture", "calibrate_modes"], default="smoke")
    ap.add_argument("--gpu", type=int, default=0, choices=range(4))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--dataset", default="gqa")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--max-pixels", type=int, default=512*512)
    ap.add_argument("--experts", default="grouped_mm", choices=["eager", "grouped_mm"])
    ap.add_argument("--layers", default="0,1,2,3,12,24,36,47")
    ap.add_argument("--ablation-semantics", choices=["paper_zero", "official_input"], default="official_input")
    ap.add_argument("--logit-mask", choices=["official", "answer_prediction"], default="official")
    ap.add_argument("--route-limit", type=int, default=32)
    ap.add_argument("--hidden-layers",default="",help="Bounded exact-input replay capture")
    ap.add_argument("--calibration-prompt", choices=["official_raw", "short_answer"], default="official_raw")
    args = ap.parse_args()
    allowed_devices()
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.manual_seed(7317)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation=args.experts, device_map=f"cuda:{args.gpu}").eval()
    processor = AutoProcessor.from_pretrained(args.model)
    observer = Observer(model, processor, args.out)
    observer.save_hidden_layers={int(x) for x in args.hidden_layers.split(",") if x}
    observer.ablation_semantics = args.ablation_semantics
    metadata = {"job": args.job, "physical_gpu": physical_gpu(args.gpu),
                "started_utc": started_utc, "arguments": {k: str(v) for k,v in vars(args).items()},
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "quality_runtime": "HF_SINGLE_GPU_REPLICA_NOT_EP_TIMING",
                "model": args.model, "experts": args.experts,
                "load_seconds": time.time()-start,
                "allocated_gib": torch.cuda.memory_allocated()/2**30}
    (args.out / f"environment_{args.shard}.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    records = load_records(args.data, args.dataset, args.split, args.limit, args.shard, args.shards)
    layers = [int(x) for x in args.layers.split(",")]
    for idx, row in enumerate(records):
        observer.context = {"request_id": row["id"], "dataset": row["dataset"],
                            "image_id": row["image_id"], "save_routes": args.job in ("capture", "calibrate_modes") and idx < args.route_limit}
        observer.record = True
        inputs, prompt_length = make_inputs(processor, row, model.device, answer=True, max_pixels=args.max_pixels,
                                           instruction=args.job != "calibrate_modes" or args.calibration_prompt == "short_answer")
        # Public MoDES selects logits at answer positions (including EOS), not
        # shifted next-token prediction positions. Keep that as the faithful
        # default; the shifted mask is a separately labeled calibration control.
        if args.logit_mask == "official":
            last_header = int((inputs.input_ids[0] == 151644).nonzero()[-1])
            logit_slice = slice(last_header + 3, None)
        else:
            logit_slice = slice(prompt_length - 1, -1)
        observer.ablate_layer = -1
        t0 = time.time()
        baseline = model(**inputs, use_cache=False).logits[:, logit_slice].float()
        teacher = baseline.softmax(-1)
        record = {"request_id": row["id"], "tokens": inputs.input_ids.numel(),
                  "answer_tokens": baseline.shape[1], "job": args.job,
                  "vanilla_seconds": time.time()-t0, "logit_mask": args.logit_mask}
        observer.record = False
        if args.job == "smoke":
            # tau=0 must be an exact no-op in our boundary port.
            observer.alpha = torch.ones(48, 2, device=model.device)/48
            observer.mode, observer.skip_counts = "modes", [0, 0]
            repeat = model(**inputs, use_cache=False).logits[:, logit_slice].float()
            record.update(noop_max_abs=float((baseline-repeat).abs().max()),
                          noop_cosine=float(F.cosine_similarity(baseline.flatten(),repeat.flatten(),dim=0)),
                          noop_first_token_equal=bool((baseline[:,0].argmax(-1)==repeat[:,0].argmax(-1)).all()),
                          skipped=observer.skip_counts[0])
            observer.mode = "vanilla"
            prompt_inputs, _ = make_inputs(processor, row, model.device, max_pixels=args.max_pixels)
            generated = model.generate(**prompt_inputs, max_new_tokens=32, do_sample=False)
            record["generated"] = processor.tokenizer.decode(generated[0,prompt_length:], skip_special_tokens=True)
            record["answers"] = row["answers"]
        elif args.job == "calibrate_modes":
            scores = {}
            for layer in layers:
                for modality in ["text", "vision"]:
                    observer.ablate_layer, observer.ablate_modality = layer, modality
                    altered = model(**inputs, use_cache=False).logits[:, logit_slice].float()
                    kl = F.kl_div(altered.log_softmax(-1), teacher, reduction="sum").clamp_min(0)
                    scores[f"{layer}_{modality}"] = float(kl)
            record["kl_sums"] = scores
            record["ablation_semantics"] = args.ablation_semantics
        append_jsonl(args.out/f"quality_{args.shard}.jsonl", [record])
        append_jsonl(args.out/f"predictor_{args.shard}.jsonl", observer.rows)
        observer.rows.clear()
        print(json.dumps({k:v for k,v in record.items() if k != "kl_sums"}), flush=True)
    torch.cuda.synchronize()
    (args.out/f"completed_{args.shard}.json").write_text(json.dumps({"elapsed_seconds": time.time()-start,
                                                                  "started_utc": started_utc,
                                                                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                                                                  "requests": len(records)}))


if __name__ == "__main__":
    main()
