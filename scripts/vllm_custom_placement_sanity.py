"""Sanity-check vLLM EP with layer-wise custom expert placement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _read_audit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _reconstruct_layer_map(records: list[dict[str, Any]], layer: int) -> list[int]:
    layer_records = [row for row in records if int(row["layer"]) == layer]
    if not layer_records:
        return []
    num_experts = int(layer_records[0]["global_num_experts"])
    owner = [-1 for _ in range(num_experts)]
    for row in layer_records:
        rank = int(row["ep_rank"])
        for expert in row["local_global_experts"]:
            owner[int(expert)] = rank
    return owner


def _audit_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_rank = {
        (int(row["layer"]), int(row["ep_rank"])): row for row in records
    }
    layers = sorted({layer for layer, _ in by_layer_rank})
    ranks = sorted({rank for _, rank in by_layer_rank})
    invalid_counts = [
        {"layer": layer, "rank": rank, "count": len(row["local_global_experts"])}
        for (layer, rank), row in sorted(by_layer_rank.items())
        if len(row["local_global_experts"]) != 16
    ]
    layer9 = _reconstruct_layer_map(records, 9)
    layer20 = _reconstruct_layer_map(records, 20)
    return {
        "num_records": len(records),
        "num_layers_seen": len(layers),
        "layers_seen": layers,
        "ranks_seen": ranks,
        "invalid_local_expert_counts": invalid_counts,
        "layer_9_map_available": bool(layer9),
        "layer_20_map_available": bool(layer20),
        "layer_9_differs_from_layer_20": bool(layer9 and layer20 and layer9 != layer20),
        "layer_9_rank0_experts": [
            expert for expert, rank in enumerate(layer9) if rank == 0
        ],
        "layer_20_rank0_experts": [
            expert for expert, rank in enumerate(layer20) if rank == 0
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument(
        "--placement-map",
        default="outputs/placement/modality_balanced_map_perlayer.json",
    )
    parser.add_argument(
        "--output", default="outputs/placement/vllm_custom_placement_sanity.json"
    )
    parser.add_argument(
        "--audit-jsonl", default="outputs/placement/vllm_custom_placement_audit.jsonl"
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--moe-backend", default="auto")
    args = parser.parse_args()

    placement_map = Path(args.placement_map).resolve()
    audit_path = Path(args.audit_jsonl).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists():
        audit_path.unlink()

    os.environ["VLLM_MOE_EXPERT_MAP_JSON"] = str(placement_map)
    os.environ["VLLM_MOE_EXPERT_MAP_AUDIT_JSONL"] = str(audit_path)

    from vllm_custom_placement import apply_vllm_custom_placement_patch

    patch_applied = apply_vllm_custom_placement_patch()

    from vllm import LLM, SamplingParams

    result: dict[str, Any] = {
        "note": "Layer-wise custom expert_map sanity; fused kernels are not modified.",
        "model_path": args.model_path,
        "placement_map": str(placement_map),
        "audit_jsonl": str(audit_path),
        "patch_applied": patch_applied,
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
            "moe_backend": args.moe_backend,
        },
        "memory_before": _gpu_memory(),
    }

    llm_kwargs: dict[str, Any] = dict(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        expert_placement_strategy="linear",
        all2all_backend="allgather_reducescatter",
        enable_return_routed_experts=True,
        enable_ep_weight_filter=True,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        disable_log_stats=False,
    )
    if args.moe_backend != "auto":
        llm_kwargs["moe_backend"] = args.moe_backend
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
    result["generation"] = {
        "prompt_token_count": len(request.prompt_token_ids or []),
        "output_text": completion.text,
        "output_token_ids": list(completion.token_ids),
        "routed_experts_shape": None if routed is None else list(routed.shape),
        "routed_experts_min": None if routed is None else int(routed.min()),
        "routed_experts_max": None if routed is None else int(routed.max()),
    }

    audit_records = _read_audit(audit_path)
    result["audit"] = _audit_summary(audit_records)
    result["gate_pass"] = bool(
        patch_applied
        and routed is not None
        and result["audit"]["num_layers_seen"] >= 48
        and result["audit"]["ranks_seen"] == list(range(args.tensor_parallel_size))
        and not result["audit"]["invalid_local_expert_counts"]
        and result["audit"]["layer_9_differs_from_layer_20"]
    )
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
