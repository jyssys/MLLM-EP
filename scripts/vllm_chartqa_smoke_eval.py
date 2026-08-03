"""Small ChartQA smoke accuracy run for vLLM Qwen3-VL-MoE EP."""

from __future__ import annotations

import argparse
import io
import json
import re
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


def _image_from_cell(value: Any) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError(f"unsupported image cell: {type(value)!r}")


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.%$ -]", "", text)
    return text.strip()


def _first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if match is None:
        return None
    return float(match.group(0))


def _is_correct(prediction: str, answer: str) -> bool:
    pred_norm = _normalize_text(prediction)
    answer_norm = _normalize_text(answer)
    if pred_norm == answer_norm:
        return True

    pred_num = _first_number(pred_norm)
    answer_num = _first_number(answer_norm)
    if pred_num is None or answer_num is None:
        return answer_norm in pred_norm

    if re.fullmatch(r"\d{4}", answer_norm):
        return pred_norm == answer_norm

    tolerance = max(1e-3, abs(answer_num) * 1e-3)
    return abs(pred_num - answer_num) <= tolerance


def _build_prompts(
    processor: AutoProcessor,
    rows: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = []
    metadata = []
    for idx, row in rows.iterrows():
        image = _image_from_cell(row["image"])
        question = str(row["question"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Answer the chart question with only the final answer. "
                            f"Question: {question}"
                        ),
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        metadata.append(
            {
                "row_index": int(idx),
                "type": str(row.get("type", "")),
                "question": question,
                "answer": str(row["answer"]),
            }
        )
    return prompts, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument(
        "--chartqa-path",
        default="data/benchmarks/ChartQA/data/test-00000-of-00001.parquet",
    )
    parser.add_argument("--output", default="outputs/accuracy_smoke/chartqa20_vllm_ep.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_parquet(args.chartqa_path).head(args.limit)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    prompts, metadata = _build_prompts(processor, df)

    result: dict[str, Any] = {
        "note": "Small smoke accuracy only; not an official benchmark.",
        "model_path": args.model_path,
        "dataset": args.chartqa_path,
        "limit": args.limit,
        "settings": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_expert_parallel": True,
            "expert_placement_strategy": "linear",
            "all2all_backend": "allgather_reducescatter",
            "enable_ep_weight_filter": True,
            "enable_return_routed_experts": False,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_tokens": args.max_tokens,
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
        use_tqdm=True,
    )
    result["memory_after_generate"] = _gpu_memory()

    examples = []
    correct = 0
    for item, request in zip(metadata, outputs, strict=True):
        completion = request.outputs[0]
        prediction = completion.text.strip()
        ok = _is_correct(prediction, item["answer"])
        correct += int(ok)
        examples.append(
            {
                **item,
                "prompt_token_count": len(request.prompt_token_ids or []),
                "prediction": prediction,
                "correct": ok,
            }
        )

    result["summary"] = {
        "num_examples": len(examples),
        "num_correct": correct,
        "accuracy": correct / max(1, len(examples)),
    }
    result["examples"] = examples

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
