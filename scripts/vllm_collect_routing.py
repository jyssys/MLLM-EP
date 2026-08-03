"""Collect routed-expert ids from vLLM for EP/single-GPU validation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
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
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def _count_loads(
    routed: np.ndarray,
    *,
    num_experts: int,
    ep_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    expert_load = np.bincount(routed.reshape(-1), minlength=num_experts).astype(np.int64)
    experts_per_rank = num_experts // ep_degree
    rank_ids = routed.reshape(-1) // experts_per_rank
    rank_load = np.bincount(rank_ids, minlength=ep_degree).astype(np.int64)
    return expert_load, rank_load


def collect(args: argparse.Namespace) -> None:
    llm_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_return_routed_experts": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "disable_log_stats": False,
    }
    if args.enable_expert_parallel:
        llm_kwargs.update(
            {
                "enable_expert_parallel": True,
                "expert_placement_strategy": "linear",
                "all2all_backend": "allgather_reducescatter",
                "enable_ep_weight_filter": True,
            }
        )

    result: dict[str, Any] = {
        "mode": "ep" if args.enable_expert_parallel else "single",
        "model_path": args.model_path,
        "settings": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_expert_parallel": bool(args.enable_expert_parallel),
            "expert_placement_strategy": "linear" if args.enable_expert_parallel else None,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
        "memory_before": _gpu_memory(),
    }

    llm = LLM(**llm_kwargs)
    result["memory_after_load"] = _gpu_memory()
    outputs = llm.generate(
        [_dummy_prompt(args.model_path)],
        SamplingParams(max_tokens=1, temperature=0.0),
        use_tqdm=False,
    )
    result["memory_after_generate"] = _gpu_memory()

    request = outputs[0]
    completion = request.outputs[0]
    routed = completion.routed_experts
    if routed is None:
        raise RuntimeError("vLLM did not return routed_experts")

    expert_load, rank_load = _count_loads(
        routed,
        num_experts=args.num_experts,
        ep_degree=args.ep_degree,
    )
    result["generation"] = {
        "prompt_token_count": len(request.prompt_token_ids or []),
        "output_text": completion.text,
        "output_token_ids": list(completion.token_ids),
        "routed_experts_shape": list(routed.shape),
        "routed_experts_min": int(routed.min()),
        "routed_experts_max": int(routed.max()),
        "expert_load_sum": int(expert_load.sum()),
        "rank_load_sum": int(rank_load.sum()),
    }

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        routed_experts=routed.astype(np.int32),
        expert_load=expert_load,
        rank_load=rank_load,
    )
    output_prefix.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def compare(args: argparse.Namespace) -> None:
    ep = np.load(args.ep_npz)
    single = np.load(args.single_npz)
    ep_routed = ep["routed_experts"]
    single_routed = single["routed_experts"]
    same_shape = ep_routed.shape == single_routed.shape
    if same_shape:
        mismatch_count = int(np.count_nonzero(ep_routed != single_routed))
        exact_match = mismatch_count == 0
    else:
        mismatch_count = None
        exact_match = False

    ep_expert = ep["expert_load"]
    single_expert = single["expert_load"]
    ep_rank = ep["rank_load"]
    single_rank = single["rank_load"]
    report = {
        "ep_npz": args.ep_npz,
        "single_npz": args.single_npz,
        "same_shape": same_shape,
        "exact_routing_match": exact_match,
        "routing_mismatch_count": mismatch_count,
        "routing_total_entries": int(ep_routed.size) if same_shape else None,
        "expert_load_equal": bool(np.array_equal(ep_expert, single_expert)),
        "rank_load_equal": bool(np.array_equal(ep_rank, single_rank)),
        "expert_load_l1": int(np.abs(ep_expert - single_expert).sum()),
        "rank_load_l1": int(np.abs(ep_rank - single_rank).sum()),
        "ep_rank_load": ep_rank.astype(int).tolist(),
        "single_rank_load": single_rank.astype(int).tolist(),
        "ep_top_experts": np.argsort(-ep_expert)[:10].astype(int).tolist(),
        "single_top_experts": np.argsort(-single_expert)[:10].astype(int).tolist(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--model-path", default="models/Qwen3-VL-30B-A3B-Instruct")
    collect_parser.add_argument("--output-prefix", required=True)
    collect_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    collect_parser.add_argument("--enable-expert-parallel", action="store_true")
    collect_parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    collect_parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    collect_parser.add_argument("--max-model-len", type=int, default=512)
    collect_parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    collect_parser.add_argument("--num-experts", type=int, default=128)
    collect_parser.add_argument("--ep-degree", type=int, default=8)
    collect_parser.set_defaults(func=collect)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--ep-npz", required=True)
    compare_parser.add_argument("--single-npz", required=True)
    compare_parser.add_argument("--output", default="outputs/ep_sim_validation/compare.json")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
