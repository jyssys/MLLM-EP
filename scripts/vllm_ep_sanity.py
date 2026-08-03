"""Sanity-check vLLM native expert parallelism for Qwen3-VL-MoE."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

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


def _dummy_prompt(model_path: str) -> dict[str, Any]:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--output", default="outputs/vllm_ep_sanity.json")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--skip-mm-profiling", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--disable-return-routed-experts", action="store_true")
    args = parser.parse_args()
    enable_return_routed_experts = not args.disable_return_routed_experts

    result: dict[str, Any] = {
        "model_path": args.model_path,
        "vllm_settings": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_expert_parallel": True,
            "expert_placement_strategy": "linear",
            "all2all_backend": "allgather_reducescatter",
            "enable_return_routed_experts": enable_return_routed_experts,
            "enable_eplb": False,
            "enable_ep_weight_filter": True,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "skip_mm_profiling": args.skip_mm_profiling,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "moe_backend": args.moe_backend,
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
        enable_return_routed_experts=enable_return_routed_experts,
        enable_ep_weight_filter=True,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        skip_mm_profiling=args.skip_mm_profiling,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        moe_backend=args.moe_backend,
        enforce_eager=True,
        disable_log_stats=False,
    )
    result["memory_after_load"] = _gpu_memory()

    sampling = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate([_dummy_prompt(args.model_path)], sampling, use_tqdm=False)
    result["memory_after_generate"] = _gpu_memory()

    request = outputs[0]
    completion = request.outputs[0]
    routed = completion.routed_experts
    prompt_token_ids = request.prompt_token_ids or []
    image_token_id = 151655
    result["generation"] = {
        "prompt_token_count": len(prompt_token_ids),
        "image_token_count": sum(1 for token_id in prompt_token_ids if token_id == image_token_id),
        "output_text": completion.text,
        "output_token_ids": list(completion.token_ids),
        "routed_experts_shape": None if routed is None else list(routed.shape),
        "routed_experts_dtype": None if routed is None else str(routed.dtype),
        "routed_experts_min": None if routed is None else int(routed.min()),
        "routed_experts_max": None if routed is None else int(routed.max()),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
