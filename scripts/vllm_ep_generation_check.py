"""End-to-end generation check for vLLM Qwen3-VL-MoE expert parallelism."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def _gpu_memory() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.strip().splitlines():
        index, used, total = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
            }
        )
    return rows


def _image_from_parquet_cell(value: Any) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError(f"unsupported parquet image cell: {type(value)!r}")


def _build_samples() -> list[dict[str, Any]]:
    chartqa = pd.read_parquet("data/benchmarks/ChartQA/data/test-00000-of-00001.parquet").iloc[0]
    textvqa = pd.read_parquet("data/benchmarks/TextVQA/data/validation-00000-of-00003.parquet").iloc[0]
    mmmu = pd.read_parquet("data/benchmarks/MMMU/data/validation-00000-of-00001.parquet").iloc[0]
    return [
        {
            "id": "chartqa_0",
            "dataset": "ChartQA",
            "image": _image_from_parquet_cell(chartqa["image"]),
            "question": f"Answer the chart question briefly: {chartqa['question']}",
            "reference": str(chartqa["answer"]),
        },
        {
            "id": "textvqa_0",
            "dataset": "TextVQA",
            "image": _image_from_parquet_cell(textvqa["image"]),
            "question": f"Answer using the visible text in the image: {textvqa['question']}",
            "reference": ", ".join(map(str, textvqa["answers"][:5])),
        },
        {
            "id": "mmmu_0",
            "dataset": "MMMU",
            "image": _image_from_parquet_cell(mmmu["image_1"]),
            "question": (
                "Solve the visual multiple-choice question. "
                f"Question: {mmmu['question']} Options: {mmmu['options']} "
                "Return the option letter and a short reason."
            ),
            "reference": str(mmmu["answer"]),
        },
    ]


def _prompt(processor: AutoProcessor, image: Image.Image, question: str) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--output", default="outputs/vllm_ep_generation_check.json")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    samples = _build_samples()
    prompts = [_prompt(processor, sample["image"], sample["question"]) for sample in samples]

    result: dict[str, Any] = {
        "model_path": args.model_path,
        "settings": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_expert_parallel": True,
            "expert_placement_strategy": "linear",
            "all2all_backend": "allgather_reducescatter",
            "enable_return_routed_experts": True,
            "enable_ep_weight_filter": True,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
        },
        "memory_before": _gpu_memory(),
    }

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        expert_placement_strategy="linear",
        all2all_backend="allgather_reducescatter",
        enable_return_routed_experts=True,
        enable_ep_weight_filter=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        disable_log_stats=False,
    )
    result["memory_after_load"] = _gpu_memory()

    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
        use_tqdm=False,
    )
    result["memory_after_generate"] = _gpu_memory()

    rows = []
    for sample, request in zip(samples, outputs, strict=True):
        completion = request.outputs[0]
        routed = completion.routed_experts
        rows.append(
            {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "question": sample["question"],
                "reference": sample["reference"],
                "prompt_token_count": len(request.prompt_token_ids or []),
                "output_text": completion.text.strip(),
                "output_token_ids": list(completion.token_ids),
                "routed_experts_shape": None if routed is None else list(routed.shape),
                "routed_experts_min": None if routed is None else int(routed.min()),
                "routed_experts_max": None if routed is None else int(routed.max()),
            }
        )
    result["samples"] = rows

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
