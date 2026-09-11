#!/usr/bin/env python3
"""Run official SDAR or TEAM decoding with clean or structural measurement.

This imports the released decoding loop rather than reimplementing it.  The
baseline and TEAM model views differ only in `modeling_sdar_moe.py`.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPECTED_UUID = {
    "6": "GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28",
    "7": "GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b",
}


def validate_stage_a_gpu() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in EXPECTED_UUID:
        raise RuntimeError(
            f"Stage A requires exactly physical GPU 6 or 7; got CUDA_VISIBLE_DEVICES={visible!r}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")


def load_official_generator(path: Path):
    """Load only the released generation functions from an OpenCompass wrapper.

    Importing `opencompass.models` eagerly imports every optional backend.  The
    released repository does not include its requirements directory, so this
    AST loader executes the unchanged generation helpers without importing the
    unrelated OpenCompass registry/model classes.  The function bodies and
    constants remain byte-for-byte sourced from the official file.
    """
    required = {
        "add_gumbel_noise",
        "top_k_logits",
        "top_p_logits",
        "sample_with_temperature_topk_topp",
        "sample_with_greedy",
        "get_num_transfer_tokens",
        "block_diffusion_generate",
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in required
    ]
    names = {node.name for node in definitions}
    if "block_diffusion_generate" not in names:
        raise RuntimeError(f"official generator not found in {path}")
    module = ast.Module(body=definitions, type_ignores=[])
    namespace = {
        "torch": torch,
        "np": np,
        "F": F,
        "DynamicCache": DynamicCache,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["block_diffusion_generate"]


class WorkTracer:
    def __init__(self, model: torch.nn.Module):
        self.model_forwards = 0
        self.mlp_calls: list[dict[str, Any]] = []
        self._current: dict[int, dict[str, Any]] = {}
        self._handles = []

        self._handles.append(model.register_forward_pre_hook(self._model_pre))
        layers = model.model.layers
        for layer_id, layer in enumerate(layers):
            mlp = layer.mlp
            self._handles.append(
                mlp.register_forward_pre_hook(
                    lambda module, args, kwargs, lid=layer_id: self._mlp_pre(lid, args, kwargs),
                    with_kwargs=True,
                )
            )
            self._handles.append(
                mlp.register_forward_hook(
                    lambda module, args, kwargs, output, lid=layer_id: self._mlp_post(lid),
                    with_kwargs=True,
                )
            )
            for expert_id, expert in enumerate(mlp.experts):
                self._handles.append(
                    expert.register_forward_pre_hook(
                        lambda module, args, lid=layer_id, eid=expert_id: self._expert_pre(lid, eid, args)
                    )
                )

    def _model_pre(self, _module, _args):
        self.model_forwards += 1

    def _mlp_pre(self, layer_id, args, kwargs):
        hidden = args[0]
        decoded = kwargs.get("decoded_index", args[2] if len(args) > 2 else None)
        physical_rows = hidden.numel() // hidden.shape[-1]
        requested_rows = physical_rows if decoded is None else int((~decoded).sum().item())
        limited = kwargs.get("expert_limit_index", args[3] if len(args) > 3 else None)
        self._current[layer_id] = {
            "layer": layer_id,
            "physical_rows": physical_rows,
            "requested_rows": requested_rows,
            "expert_limited_rows": 0 if limited is None else int(limited.sum().item()),
            "token_expert_pairs": 0,
            "active_experts": set(),
            "expert_rows": {},
        }

    def _expert_pre(self, layer_id, expert_id, args):
        if layer_id not in self._current or not args:
            return
        hidden = args[0]
        rows = 0 if hidden.ndim == 0 else int(hidden.shape[0])
        current = self._current[layer_id]
        current["token_expert_pairs"] += rows
        current["expert_rows"][expert_id] = rows
        if rows:
            current["active_experts"].add(expert_id)

    def _mlp_post(self, layer_id):
        current = self._current.pop(layer_id)
        current["active_expert_count"] = len(current.pop("active_experts"))
        self.mlp_calls.append(current)

    def close(self):
        for handle in self._handles:
            handle.remove()

    def summary(self) -> dict[str, Any]:
        pairs = sum(row["token_expert_pairs"] for row in self.mlp_calls)
        requested = sum(row["requested_rows"] for row in self.mlp_calls)
        physical = sum(row["physical_rows"] for row in self.mlp_calls)
        return {
            "model_forwards": self.model_forwards,
            "moe_layer_calls": len(self.mlp_calls),
            "physical_moe_rows": physical,
            "requested_moe_rows": requested,
            "token_expert_pairs": pairs,
            "active_expert_events": sum(row["active_expert_count"] for row in self.mlp_calls),
            "mean_active_experts_per_layer_call": (
                sum(row["active_expert_count"] for row in self.mlp_calls) / len(self.mlp_calls)
                if self.mlp_calls
                else 0.0
            ),
            "expert_limited_rows": sum(row["expert_limited_rows"] for row in self.mlp_calls),
        }


def tokenize(tokenizer, prompt: str, device: torch.device):
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.batch_encode_plus(
        [rendered], return_tensors="pt", padding=True, truncation=True,
        add_special_tokens=False, max_length=32768,
    )
    return {key: value.to(device) for key, value in tokens.items()}


def stop_ids(tokenizer) -> list[int]:
    ids = []
    candidates = tokenizer.eos_token_id
    if isinstance(candidates, int):
        ids.append(candidates)
    elif candidates:
        ids.extend(candidates)
    return sorted(set(ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "team"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--team-repo", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trace-work", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ids", help="comma-separated sample IDs, preserving dataset order")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    validate_stage_a_gpu()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wrapper_name = (
        "huggingface_bd3_original.py"
        if args.mode == "baseline"
        else "huggingface_bd3_decoded_jump_expert_limit_speculative.py"
    )
    generator = load_official_generator(
        args.team_repo / "evaluation/opencompass/opencompass/models" / wrapper_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    load_peak = torch.cuda.max_memory_allocated()

    tracer = WorkTracer(model) if args.trace_work else None
    records = []
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    if args.ids:
        requested_ids = set(args.ids.split(","))
        prompts = [row for row in prompts if row.get("id") in requested_ids]
        missing = requested_ids - {row.get("id") for row in prompts}
        if missing:
            raise ValueError(f"unknown prompt IDs: {sorted(missing)}")
    if args.max_samples is not None:
        prompts = prompts[: args.max_samples]

    def generate_once(item):
        tokens = tokenize(tokenizer, item["prompt"], model.device)
        kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            prompt=tokens,
            mask_id=151669,
            gen_length=args.gen_length,
            block_length=args.block_length,
            denoising_steps=args.denoising_steps,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            remasking="low_confidence",
            threshold=args.threshold,
            stopping_criteria_idx=stop_ids(tokenizer),
        )
        return generator(**kwargs), tokens["input_ids"].shape[1]

    for warm_index in range(args.warmup):
        generate_once(prompts[warm_index % len(prompts)])
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    for item in prompts:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        before_work = tracer.summary() if tracer else None
        start = time.perf_counter()
        output, prompt_length = generate_once(item)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = output[:, prompt_length:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
        decoded = decoded.replace("<|MASK|>", "")
        after_work = tracer.summary() if tracer else None
        if tracer:
            work = {key: after_work[key] - before_work[key] for key in after_work}
        else:
            work = None
        records.append({
            "id": item.get("id"),
            "task": item.get("task"),
            "prompt": item["prompt"],
            "reference": item.get("reference"),
            "elapsed_s": elapsed,
            "generated_token_slots": int(generated.numel()),
            "output": decoded,
            "work": work,
        })
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    payload = {
        "mode": args.mode,
        "model_dir": str(args.model_dir.resolve()),
        "wrapper": str(wrapper_name),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "dtype": "float16",
        "generation": {
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "denoising_steps": args.denoising_steps,
            "threshold": args.threshold,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
        "trace_work": args.trace_work,
        "model_load_peak_bytes": load_peak,
        "measurement_peak_bytes": torch.cuda.max_memory_allocated(),
        "records": records,
        "trace_summary": tracer.summary() if tracer else None,
        "trace_rows": tracer.mlp_calls if tracer else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if tracer:
        tracer.close()


if __name__ == "__main__":
    main()
