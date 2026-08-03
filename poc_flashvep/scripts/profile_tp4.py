"""Run a fixed-input TP4/EP4 FlashVEP baseline profiling workload."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def _prompt(model_path: str, image_size: int) -> tuple[dict[str, Any], str, int]:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    image = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_token = str(processor.image_token)
    image_token_id = int(processor.tokenizer.convert_tokens_to_ids(image_token))
    return (
        {"prompt": text, "multi_modal_data": {"image": image}},
        image_token,
        image_token_id,
    )


def _request(llm: LLM, prompt: dict[str, Any], sampling: SamplingParams) -> Any:
    return llm.generate([prompt], sampling, use_tqdm=False)[0]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--moe-backend", default="triton")
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    prompt, image_token, image_token_id = _prompt(args.model_path, args.image_size)
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
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        skip_mm_profiling=True,
        enable_prefix_caching=False,
        moe_backend=args.moe_backend,
        enforce_eager=True,
        disable_log_stats=False,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0)

    warmup_outputs: list[int] = []
    for _ in range(args.warmups):
        request = _request(llm, prompt, sampling)
        warmup_outputs.append(int(request.outputs[0].token_ids[0]))

    rows: list[dict[str, Any]] = []
    reference_prompt_ids: list[int] | None = None
    for iteration in range(args.iterations):
        start_ns = time.perf_counter_ns()
        request = _request(llm, prompt, sampling)
        end_ns = time.perf_counter_ns()
        completion = request.outputs[0]
        prompt_ids = [int(value) for value in (request.prompt_token_ids or [])]
        if reference_prompt_ids is None:
            reference_prompt_ids = prompt_ids
        elif prompt_ids != reference_prompt_ids:
            raise AssertionError("fixed prompt tokenization changed between iterations")
        routed = completion.routed_experts
        if routed is None:
            raise RuntimeError("routed expert capture is unavailable")
        rows.append(
            {
                "iteration_id": iteration,
                "wall_ms": (end_ns - start_ns) / 1_000_000,
                "prompt_token_count": len(prompt_ids),
                "image_token_count": sum(value == image_token_id for value in prompt_ids),
                "text_and_special_token_count": sum(
                    value != image_token_id for value in prompt_ids
                ),
                "output_token_ids": [int(value) for value in completion.token_ids],
                "output_text": completion.text,
                "routed_experts_shape": list(routed.shape),
                "routed_experts_min": int(routed.min()),
                "routed_experts_max": int(routed.max()),
            }
        )

    token_outputs = [row["output_token_ids"] for row in rows]
    if any(value != token_outputs[0] for value in token_outputs[1:]):
        raise AssertionError("greedy output token changed across measured iterations")
    wall = [float(row["wall_ms"]) for row in rows]
    result = {
        "run_id": output.parent.name,
        "settings": {
            "physical_gpus": [4, 5, 6, 7],
            "tensor_parallel_size": args.tensor_parallel_size,
            "effective_expert_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "dtype": "bfloat16",
            "moe_backend": args.moe_backend,
            "all2all_backend": "allgather_reducescatter",
            "dispatch_collective_present": False,
            "combine_collective": "tensor_parallel_all_reduce",
            "prefix_caching": False,
            "skip_mm_profiling": True,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "input_image_size": [args.image_size, args.image_size],
            "image_token": image_token,
            "image_token_id": image_token_id,
        },
        "warmup_output_token_ids": warmup_outputs,
        "request_wall_ms": {
            "mean": statistics.fmean(wall),
            "median": statistics.median(wall),
            "p90": _percentile(wall, 0.9),
            "min": min(wall),
            "max": max(wall),
            "stdev": statistics.stdev(wall) if len(wall) > 1 else 0.0,
        },
        "prompt_token_ids": reference_prompt_ids,
        "iterations": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["request_wall_ms"], indent=2))


if __name__ == "__main__":
    main()
