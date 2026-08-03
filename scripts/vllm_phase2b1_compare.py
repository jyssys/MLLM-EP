"""Phase 2-B1 As-Is vs To-Be vLLM EP placement measurement."""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
NUM_LAYERS = 48
NUM_EXPERTS = 128
EP_DEGREE = 8


@dataclass
class EvalSample:
    dataset: str
    sample_id: str
    images: list[Image.Image]
    question: str
    answer: str
    answer_type: str
    metadata: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


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


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def _image_from_cell(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    return None


def _image_area(value: Any) -> int:
    image = _image_from_cell(value)
    if image is None:
        return 0
    width, height = image.size
    return int(width * height)


def _load_chartqa(limit: int) -> list[EvalSample]:
    df = pd.read_parquet("data/benchmarks/ChartQA/data/test-00000-of-00001.parquet")
    df = df.assign(_area=df["image"].map(_image_area))
    df = df.sort_values("_area", ascending=False).head(limit)
    samples = []
    for idx, row in df.iterrows():
        image = _image_from_cell(row["image"])
        if image is None:
            continue
        samples.append(
            EvalSample(
                dataset="ChartQA",
                sample_id=f"chartqa_{idx}",
                images=[image],
                question=(
                    "Answer the chart question with only the final answer. "
                    f"Question: {row['question']}"
                ),
                answer=_safe_text(row["answer"]),
                answer_type="freeform",
                metadata={"area": int(row["_area"]), "type": _safe_text(row["type"])},
            )
        )
    return samples


def _load_mmmu(limit: int) -> list[EvalSample]:
    df = pd.read_parquet("data/benchmarks/MMMU/data/validation-00000-of-00001.parquet")
    rows: list[tuple[int, int, int, Any]] = []
    for idx, row in df.iterrows():
        image_count = 0
        total_area = 0
        for i in range(1, 8):
            area = _image_area(row.get(f"image_{i}"))
            if area:
                image_count += 1
                total_area += area
        rows.append((image_count, total_area, int(idx), row))
    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

    samples = []
    for image_count, total_area, idx, row in rows[:limit]:
        images = [
            image
            for i in range(1, 8)
            if (image := _image_from_cell(row.get(f"image_{i}"))) is not None
        ]
        if not images:
            continue
        question_type = _safe_text(row["question_type"])
        if question_type == "multiple-choice":
            options = _parse_mmmu_options(row["options"])
            prompt = (
                f"Question: {row['question']}\n"
                "Options:\n"
                f"{_format_mmmu_options(options)}\n"
                "Answer with the option letter only."
            )
            answer_type = "choice"
        else:
            prompt = (
                f"Question: {row['question']}\n"
                "Please answer the question directly."
            )
            answer_type = "freeform"
        samples.append(
            EvalSample(
                dataset="MMMU",
                sample_id=f"mmmu_{row['id']}",
                images=images,
                question=prompt,
                answer=_safe_text(row["answer"]),
                answer_type=answer_type,
                metadata={
                    "row_index": idx,
                    "image_count": image_count,
                    "area": total_area,
                    "subfield": _safe_text(row["subfield"]),
                    "question_type": question_type,
                    "options": _parse_mmmu_options(row["options"]),
                },
            )
        )
    return samples


def _load_mmstar(limit: int) -> list[EvalSample]:
    df = pd.read_parquet("data/benchmarks/MMStar/mmstar.parquet")
    df = df.assign(_area=df["image"].map(_image_area))
    df = df.sort_values("_area", ascending=False).head(limit)
    samples = []
    for _, row in df.iterrows():
        image = _image_from_cell(row["image"])
        if image is None:
            continue
        samples.append(
            EvalSample(
                dataset="MMStar",
                sample_id=f"mmstar_{row['index']}",
                images=[image],
                question=(
                    "Answer the multiple-choice visual question. "
                    f"{row['question']} Return only the answer letter."
                ),
                answer=_safe_text(row["answer"]),
                answer_type="choice",
                metadata={
                    "area": int(row["_area"]),
                    "category": _safe_text(row["category"]),
                    "l2_category": _safe_text(row["l2_category"]),
                },
            )
        )
    return samples


def _load_sharegpt4v(limit: int) -> list[EvalSample]:
    manifest = Path("data/sharegpt4v_512/manifest.jsonl")
    samples: list[EvalSample] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if len(samples) >= limit:
                break
            row = json.loads(line)
            image_path = Path(row["local_image"])
            if not image_path.exists():
                continue
            conversations = row.get("conversations", [])
            human = next(
                (
                    turn.get("value", "")
                    for turn in conversations
                    if turn.get("from") == "human"
                ),
                "Describe the image.",
            )
            answer = next(
                (
                    turn.get("value", "")
                    for turn in conversations
                    if turn.get("from") == "gpt"
                ),
                "",
            )
            question = human.replace("<image>", "").strip() or "Describe the image."
            samples.append(
                EvalSample(
                    dataset="ShareGPT4V",
                    sample_id=f"sharegpt4v_{row['id']}",
                    images=[Image.open(image_path).convert("RGB")],
                    question=question,
                    answer=_safe_text(answer),
                    answer_type="freeform",
                    metadata={"image": row.get("image", ""), "local_image": str(image_path)},
                )
            )
    return samples


def _load_samples(limit_per_dataset: int, datasets: list[str]) -> list[EvalSample]:
    loaders = {
        "ChartQA": _load_chartqa,
        "MMMU": _load_mmmu,
        "MMStar": _load_mmstar,
        "ShareGPT4V": _load_sharegpt4v,
    }
    samples: list[EvalSample] = []
    for dataset in datasets:
        samples.extend(loaders[dataset](limit_per_dataset))
    return samples


def _build_prompt(
    processor: AutoProcessor,
    sample: EvalSample,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image} for image in sample.images
    ]
    content.append({"type": "text", "text": sample.question})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    images: Image.Image | list[Image.Image]
    images = sample.images[0] if len(sample.images) == 1 else sample.images
    return {"prompt": prompt, "multi_modal_data": {"image": images}}


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.%$ -]", "", text)
    return text.strip()


def _parse_mmmu_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError(f"MMMU options must parse to a list: {value!r}")
        return [str(item) for item in parsed]
    raise ValueError(f"unsupported MMMU options value: {type(value)!r}")


def _format_mmmu_options(options: list[str]) -> str:
    return "\n".join(
        f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(options)
    )


def _first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return float(match.group(0)) if match else None


def _choice_prediction(text: str) -> str:
    match = re.search(r"\b([A-H])\b", text.upper())
    if match:
        return match.group(1)
    stripped = text.strip().upper()
    return stripped[:1] if stripped[:1] in set("ABCDEFGH") else ""


def _parse_mmmu_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
) -> str:
    """Parse MMMU multiple-choice output following lmms-eval/MMMU logic."""

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = f" {response} "

    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if not candidates:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if not candidates and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if not candidates:
        return random.choice(all_choices)
    if len(candidates) == 1:
        return candidates[0]

    start_indexes = []
    if index_ans:
        if ans_with_brack:
            start_indexes = [response.rfind(f"({can})") for can in candidates]
        else:
            start_indexes = [response.rfind(f" {can} ") for can in candidates]
    else:
        start_indexes = [
            response.lower().rfind(index2ans[can].lower()) for can in candidates
        ]
    return candidates[int(np.argmax(start_indexes))]


def _is_correct(
    prediction: str,
    answer: str,
    answer_type: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    if answer_type == "choice":
        if metadata and metadata.get("options"):
            options = [str(option) for option in metadata["options"]]
            all_choices = [chr(ord("A") + idx) for idx in range(len(options))]
            index2ans = dict(zip(all_choices, options, strict=True))
            return (
                _parse_mmmu_choice_response(prediction, all_choices, index2ans)
                == answer.strip().upper()[:1]
            )
        return _choice_prediction(prediction) == answer.strip().upper()[:1]

    pred_norm = _normalize_text(prediction)
    answer_norm = _normalize_text(answer)
    if pred_norm == answer_norm:
        return True
    pred_num = _first_number(pred_norm)
    answer_num = _first_number(answer_norm)
    if pred_num is None or answer_num is None:
        return bool(answer_norm and answer_norm in pred_norm)
    if re.fullmatch(r"\d{4}", answer_norm):
        return pred_norm == answer_norm
    tolerance = max(1e-3, abs(answer_num) * 1e-3)
    return abs(pred_num - answer_num) <= tolerance


def _placement_array(path: str) -> np.ndarray:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = sorted(int(layer) for layer in raw)
    mapping = np.empty((len(layers), NUM_EXPERTS), dtype=np.int64)
    for layer in layers:
        for expert, rank in raw[str(layer)].items():
            mapping[layer, int(expert)] = int(rank)
    return mapping


def _prefill_routing(request: Any) -> tuple[np.ndarray, np.ndarray, int]:
    completion = request.outputs[0]
    routed = completion.routed_experts
    if routed is None:
        raise RuntimeError("vLLM did not return routed_experts")
    token_ids = np.asarray(request.prompt_token_ids or [], dtype=np.int64)
    n = min(len(token_ids), routed.shape[0])
    routed_arr = np.asarray(routed[:n], dtype=np.int32)
    vision_mask = np.isin(token_ids[:n], [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID])
    return routed_arr, vision_mask, n


def _rank_load_by_layer(routed: np.ndarray, placement: np.ndarray) -> np.ndarray:
    rank_load = np.zeros((routed.shape[1], EP_DEGREE), dtype=np.int64)
    for layer in range(routed.shape[1]):
        ranks = placement[layer, routed[:, layer, :].reshape(-1)]
        rank_load[layer] = np.bincount(ranks, minlength=EP_DEGREE)[:EP_DEGREE]
    return rank_load


def _expert_load_by_layer(routed: np.ndarray) -> np.ndarray:
    expert_load = np.zeros((routed.shape[1], NUM_EXPERTS), dtype=np.int64)
    for layer in range(routed.shape[1]):
        experts = routed[:, layer, :].reshape(-1)
        expert_load[layer] = np.bincount(experts, minlength=NUM_EXPERTS)[:NUM_EXPERTS]
    return expert_load


def _accumulate_load_profile(
    routed: np.ndarray,
    vision_mask: np.ndarray,
    placement: np.ndarray,
    expert_load: np.ndarray,
    expert_load_vision: np.ndarray,
    expert_load_text: np.ndarray,
    rank_load: np.ndarray,
    rank_load_vision: np.ndarray,
    rank_load_text: np.ndarray,
) -> None:
    token_modality = np.repeat(vision_mask.astype(bool), routed.shape[2])
    for layer in range(routed.shape[1]):
        experts = routed[:, layer, :].reshape(-1)
        ranks = placement[layer, experts]

        expert_counts = np.bincount(experts, minlength=NUM_EXPERTS)[:NUM_EXPERTS]
        expert_vis = np.bincount(
            experts[token_modality], minlength=NUM_EXPERTS
        )[:NUM_EXPERTS]
        expert_txt = expert_counts - expert_vis

        rank_counts = np.bincount(ranks, minlength=EP_DEGREE)[:EP_DEGREE]
        rank_vis = np.bincount(ranks[token_modality], minlength=EP_DEGREE)[:EP_DEGREE]
        rank_txt = rank_counts - rank_vis

        expert_load[layer] += expert_counts
        expert_load_vision[layer] += expert_vis
        expert_load_text[layer] += expert_txt
        rank_load[layer] += rank_counts
        rank_load_vision[layer] += rank_vis
        rank_load_text[layer] += rank_txt


def _load_stats(load: np.ndarray) -> dict[str, float]:
    values = np.asarray(load, dtype=float)
    mean = float(values.mean()) if values.size else 0.0
    max_value = float(values.max()) if values.size else 0.0
    min_value = float(values.min()) if values.size else 0.0
    return {
        "mean": mean,
        "min": min_value,
        "max": max_value,
        "max_over_mean": max_value / mean if mean > 0 else 0.0,
        "min_over_max": min_value / max_value if max_value > 0 else 0.0,
    }


def _profile_load_summary(
    expert_load: np.ndarray,
    expert_load_vision: np.ndarray,
    expert_load_text: np.ndarray,
    rank_load: np.ndarray,
    rank_load_vision: np.ndarray,
    rank_load_text: np.ndarray,
) -> dict[str, Any]:
    rank_imbalance_by_layer = _imbalance(rank_load)
    expert_imbalance_by_layer = _imbalance(expert_load)
    total_rank_load = rank_load.sum(axis=0)
    total_expert_load = expert_load.sum(axis=0)
    total_rank_vision = rank_load_vision.sum(axis=0)
    total_expert_vision = expert_load_vision.sum(axis=0)

    hot_rank = int(total_rank_load.argmax()) if total_rank_load.size else 0
    hot_expert = int(total_expert_load.argmax()) if total_expert_load.size else 0
    hot_layer = int(rank_imbalance_by_layer.argmax()) if rank_imbalance_by_layer.size else 0

    return {
        "note": (
            "Expert load is determined by router choice and is placement-invariant; "
            "rank load applies the active layer-wise expert->rank map."
        ),
        "rank_load_by_layer": rank_load,
        "rank_load_vision_by_layer": rank_load_vision,
        "rank_load_text_by_layer": rank_load_text,
        "expert_load_by_layer": expert_load,
        "expert_load_vision_by_layer": expert_load_vision,
        "expert_load_text_by_layer": expert_load_text,
        "total_rank_load": total_rank_load,
        "total_rank_vision_load": total_rank_vision,
        "total_rank_text_load": rank_load_text.sum(axis=0),
        "total_expert_load": total_expert_load,
        "total_expert_vision_load": total_expert_vision,
        "total_expert_text_load": expert_load_text.sum(axis=0),
        "rank_total_stats": _load_stats(total_rank_load),
        "expert_total_stats": _load_stats(total_expert_load),
        "rank_layer_imbalance": {
            "mean_max_over_mean": float(rank_imbalance_by_layer.mean())
            if rank_imbalance_by_layer.size
            else 0.0,
            "p95_max_over_mean": float(np.quantile(rank_imbalance_by_layer, 0.95))
            if rank_imbalance_by_layer.size
            else 0.0,
            "max_max_over_mean": float(rank_imbalance_by_layer.max())
            if rank_imbalance_by_layer.size
            else 0.0,
        },
        "expert_layer_imbalance": {
            "mean_max_over_mean": float(expert_imbalance_by_layer.mean())
            if expert_imbalance_by_layer.size
            else 0.0,
            "p95_max_over_mean": float(np.quantile(expert_imbalance_by_layer, 0.95))
            if expert_imbalance_by_layer.size
            else 0.0,
            "max_max_over_mean": float(expert_imbalance_by_layer.max())
            if expert_imbalance_by_layer.size
            else 0.0,
        },
        "hot_total_rank": {
            "rank": hot_rank,
            "load": int(total_rank_load[hot_rank]) if total_rank_load.size else 0,
            "vision_ratio": float(
                total_rank_vision[hot_rank] / max(total_rank_load[hot_rank], 1)
            )
            if total_rank_load.size
            else 0.0,
        },
        "hot_total_expert": {
            "expert": hot_expert,
            "load": int(total_expert_load[hot_expert]) if total_expert_load.size else 0,
            "vision_ratio": float(
                total_expert_vision[hot_expert] / max(total_expert_load[hot_expert], 1)
            )
            if total_expert_load.size
            else 0.0,
        },
        "hot_layer_by_rank_imbalance": {
            "layer": hot_layer,
            "rank_load": rank_load[hot_layer] if rank_load.size else [],
            "rank_vision_load": rank_load_vision[hot_layer] if rank_load.size else [],
            "rank_text_load": rank_load_text[hot_layer] if rank_load.size else [],
            "max_over_mean": float(rank_imbalance_by_layer[hot_layer])
            if rank_imbalance_by_layer.size
            else 0.0,
        },
    }


def _imbalance(load: np.ndarray) -> np.ndarray:
    mean = load.mean(axis=-1)
    max_load = load.max(axis=-1)
    return np.divide(max_load, mean, out=np.zeros_like(mean, dtype=float), where=mean > 0)


def _finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "sum": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def _positive_delta(end: float, start: float) -> float:
    if end <= 0 or start <= 0:
        return float("nan")
    return max(0.0, end - start)


def _request_timing(request: Any) -> dict[str, Any]:
    metrics = getattr(request, "metrics", None)
    if metrics is None:
        return {
            "available": False,
            "num_generation_tokens": 0,
            "queue_s": float("nan"),
            "ttft_s": float("nan"),
            "scheduled_to_first_token_s": float("nan"),
            "decode_s": float("nan"),
            "e2e_s": float("nan"),
        }

    arrival = float(getattr(metrics, "arrival_time", 0.0) or 0.0)
    scheduled = float(getattr(metrics, "scheduled_ts", 0.0) or 0.0)
    first = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    ttft = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    if ttft <= 0:
        ttft = _positive_delta(first, arrival)

    return {
        "available": True,
        "num_generation_tokens": int(getattr(metrics, "num_generation_tokens", 0) or 0),
        "queue_s": _positive_delta(scheduled, arrival),
        "ttft_s": ttft,
        "scheduled_to_first_token_s": _positive_delta(first, scheduled),
        "decode_s": _positive_delta(last, first),
        "e2e_s": _positive_delta(last, arrival),
        "is_corrupted": bool(getattr(metrics, "is_corrupted", False)),
    }


def _summarize_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_requests": len(rows),
        "num_available": int(sum(bool(row.get("available")) for row in rows)),
        "num_generation_tokens": int(
            sum(int(row.get("num_generation_tokens", 0) or 0) for row in rows)
        ),
        "queue_s": _finite_stats([float(row.get("queue_s", float("nan"))) for row in rows]),
        "ttft_s": _finite_stats([float(row.get("ttft_s", float("nan"))) for row in rows]),
        "scheduled_to_first_token_s": _finite_stats(
            [
                float(row.get("scheduled_to_first_token_s", float("nan")))
                for row in rows
            ]
        ),
        "decode_s": _finite_stats([float(row.get("decode_s", float("nan"))) for row in rows]),
        "e2e_s": _finite_stats([float(row.get("e2e_s", float("nan"))) for row in rows]),
    }


def _chunked(values: list[Any], chunk_size: int) -> list[list[Any]]:
    return [values[start : start + chunk_size] for start in range(0, len(values), chunk_size)]


def run_variant(args: argparse.Namespace) -> None:
    placement_map = Path(args.placement_map).resolve()
    audit_path = Path(args.output_dir) / f"{args.variant}_audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    os.environ["VLLM_MOE_EXPERT_MAP_JSON"] = str(placement_map)
    os.environ["VLLM_MOE_EXPERT_MAP_AUDIT_JSONL"] = str(audit_path.resolve())
    if args.moe_cuda_timing_jsonl:
        timing_jsonl = Path(args.moe_cuda_timing_jsonl).resolve()
        if timing_jsonl.exists():
            timing_jsonl.unlink()
        timing_jsonl.parent.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_MOE_TIMING_JSONL"] = str(timing_jsonl)
        os.environ["VLLM_MOE_TIMING_FLUSH_EVERY"] = str(
            args.moe_cuda_timing_flush_every
        )
    else:
        os.environ.pop("VLLM_MOE_TIMING_JSONL", None)

    from vllm_custom_placement import apply_vllm_custom_placement_patch

    patch_applied = apply_vllm_custom_placement_patch()
    moe_timing_patch_applied = False
    if args.moe_cuda_timing_jsonl:
        from vllm_moe_timing import apply_vllm_moe_timing_patch

        moe_timing_patch_applied = apply_vllm_moe_timing_patch()

    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    samples = _load_samples(args.samples_per_dataset, args.datasets)
    prompts = [_build_prompt(processor, sample) for sample in samples]
    placement = _placement_array(str(placement_map))

    result: dict[str, Any] = {
        "variant": args.variant,
        "placement_map": str(placement_map),
        "patch_applied": patch_applied,
        "moe_timing_patch_applied": moe_timing_patch_applied,
        "datasets": args.datasets,
        "samples_per_dataset": args.samples_per_dataset,
        "num_samples": len(samples),
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
            "accuracy_max_tokens": args.accuracy_max_tokens,
            "profile_max_tokens": 1,
            "moe_backend": args.moe_backend,
            "profile_only": args.profile_only,
            "save_batch_expert_loads": args.save_batch_expert_loads,
            "enable_prefix_caching": not args.disable_prefix_caching,
            "moe_cuda_timing_jsonl": args.moe_cuda_timing_jsonl,
            "moe_cuda_timing_flush_every": args.moe_cuda_timing_flush_every,
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
        enable_prefix_caching=not args.disable_prefix_caching,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
    )
    if args.moe_backend != "auto":
        llm_kwargs["moe_backend"] = args.moe_backend
    llm = LLM(**llm_kwargs)
    result["memory_after_load"] = _gpu_memory()

    # Warmup is excluded from timing and accuracy.
    llm.generate(
        [prompts[0]],
        SamplingParams(max_tokens=1, temperature=0.0),
        use_tqdm=False,
    )

    accuracy_rows = []
    correct_by_dataset: dict[str, int] = {dataset: 0 for dataset in args.datasets}
    total_by_dataset: dict[str, int] = {dataset: 0 for dataset in args.datasets}
    accuracy_elapsed = 0.0
    if not args.profile_only:
        for batch_indices in _chunked(list(range(len(prompts))), args.batch_size):
            batch_prompts = [prompts[idx] for idx in batch_indices]
            t0 = time.perf_counter()
            outputs = llm.generate(
                batch_prompts,
                SamplingParams(max_tokens=args.accuracy_max_tokens, temperature=0.0),
                use_tqdm=False,
            )
            accuracy_elapsed += time.perf_counter() - t0
            for idx, request in zip(batch_indices, outputs, strict=True):
                sample = samples[idx]
                prediction = request.outputs[0].text.strip()
                correct = _is_correct(
                    prediction,
                    sample.answer,
                    sample.answer_type,
                    sample.metadata,
                )
                correct_by_dataset[sample.dataset] += int(correct)
                total_by_dataset[sample.dataset] += 1
                accuracy_rows.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "answer": sample.answer,
                        "prediction": prediction,
                        "correct": correct,
                        "prompt_token_count": len(request.prompt_token_ids or []),
                    }
                )

    profile_batches = []
    batch_layer_imbalances = []
    total_prefill_tokens = 0
    profile_elapsed_total = 0.0
    aggregate_expert_load = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    aggregate_expert_load_vision = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    aggregate_expert_load_text = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    aggregate_rank_load = np.zeros((NUM_LAYERS, EP_DEGREE), dtype=np.int64)
    aggregate_rank_load_vision = np.zeros((NUM_LAYERS, EP_DEGREE), dtype=np.int64)
    aggregate_rank_load_text = np.zeros((NUM_LAYERS, EP_DEGREE), dtype=np.int64)
    saved_batch_expert_loads: list[np.ndarray] = []
    saved_batch_meta: list[dict[str, Any]] = []
    all_timing_rows: list[dict[str, Any]] = []
    timing_batches: list[dict[str, Any]] = []
    for batch_no, batch_indices in enumerate(
        _chunked(list(range(len(prompts))), args.batch_size)
    ):
        batch_prompts = [prompts[idx] for idx in batch_indices]
        t0 = time.perf_counter()
        outputs = llm.generate(
            batch_prompts,
            SamplingParams(max_tokens=1, temperature=0.0),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - t0
        profile_elapsed_total += elapsed
        batch_timing_rows = [_request_timing(request) for request in outputs]
        all_timing_rows.extend(batch_timing_rows)

        batch_rank_load = np.zeros((NUM_LAYERS, EP_DEGREE), dtype=np.int64)
        batch_expert_load = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
        batch_prefill_tokens = 0
        dataset_names = sorted({samples[idx].dataset for idx in batch_indices})
        for request in outputs:
            routed, _vision_mask, prefill_tokens = _prefill_routing(request)
            batch_expert_load += _expert_load_by_layer(routed)
            _accumulate_load_profile(
                routed,
                _vision_mask,
                placement,
                aggregate_expert_load,
                aggregate_expert_load_vision,
                aggregate_expert_load_text,
                aggregate_rank_load,
                aggregate_rank_load_vision,
                aggregate_rank_load_text,
            )
            batch_rank_load += _rank_load_by_layer(routed, placement)
            batch_prefill_tokens += prefill_tokens
        if args.save_batch_expert_loads:
            saved_batch_expert_loads.append(batch_expert_load.copy())
            saved_batch_meta.append(
                {
                    "batch": batch_no,
                    "datasets": dataset_names,
                    "sample_ids": [samples[idx].sample_id for idx in batch_indices],
                    "prefill_tokens": int(batch_prefill_tokens),
                }
            )
        total_prefill_tokens += batch_prefill_tokens
        imbalance = _imbalance(batch_rank_load)
        batch_layer_imbalances.extend(imbalance.tolist())
        hot_layer = int(imbalance.argmax())
        hot_rank = int(batch_rank_load[hot_layer].argmax())
        batch_timing = _summarize_timing(batch_timing_rows)
        timing_batches.append(
            {
                "batch": batch_no,
                "datasets": dataset_names,
                "num_samples": len(batch_indices),
                "wall_s": elapsed,
                "request_timing": batch_timing,
            }
        )
        profile_batches.append(
            {
                "batch": batch_no,
                "datasets": dataset_names,
                "num_samples": len(batch_indices),
                "elapsed_s": elapsed,
                "prefill_tokens": batch_prefill_tokens,
                "prefill_tokens_per_s": batch_prefill_tokens / elapsed
                if elapsed > 0
                else 0.0,
                "mean_layer_imbalance": float(imbalance.mean()),
                "max_layer_imbalance": float(imbalance.max()),
                "p95_layer_imbalance": float(np.quantile(imbalance, 0.95)),
                "hot_layer": hot_layer,
                "hot_rank": hot_rank,
                "hot_layer_rank_load": batch_rank_load[hot_layer].tolist(),
            }
        )

    imbalances = np.asarray(batch_layer_imbalances, dtype=float)
    result["memory_after_generate"] = _gpu_memory()
    result["accuracy"] = {
        "elapsed_s": accuracy_elapsed,
        "profile_only": args.profile_only,
        "overall": {
            "correct": int(sum(correct_by_dataset.values())),
            "total": int(sum(total_by_dataset.values())),
            "accuracy": float(
                sum(correct_by_dataset.values()) / max(1, sum(total_by_dataset.values()))
            ),
        },
        "by_dataset": {
            dataset: {
                "correct": int(correct_by_dataset[dataset]),
                "total": int(total_by_dataset[dataset]),
                "accuracy": float(
                    correct_by_dataset[dataset] / max(1, total_by_dataset[dataset])
                ),
            }
            for dataset in args.datasets
        },
        "examples": accuracy_rows,
    }
    result["latency"] = {
        "note": "Profile pass uses max_tokens=1; latency is prefill-dominant and includes one decode token.",
        "total_elapsed_s": profile_elapsed_total,
        "num_batches": len(profile_batches),
        "mean_batch_elapsed_s": float(
            np.mean([row["elapsed_s"] for row in profile_batches])
        ),
        "std_batch_elapsed_s": float(
            np.std([row["elapsed_s"] for row in profile_batches])
        ),
        "total_prefill_tokens": int(total_prefill_tokens),
        "prefill_tokens_per_s": float(total_prefill_tokens / profile_elapsed_total)
        if profile_elapsed_total > 0
        else 0.0,
        "batches": profile_batches,
    }
    result["timing"] = {
        "note": (
            "Request metrics from vLLM. With max_tokens=1, "
            "scheduled_to_first_token_s is the closest available prefill proxy; "
            "decode_s should be near zero and E2E includes queue/scheduling."
        ),
        "profile_max_tokens": 1,
        "batch_wall_s": _finite_stats([row["elapsed_s"] for row in profile_batches]),
        "request": _summarize_timing(all_timing_rows),
        "batches": timing_batches,
    }
    result["straggler"] = {
        "note": "Batch-layer rank max/mean imbalance computed from routed_experts and the active layer-wise expert->rank map.",
        "num_batch_layers": int(imbalances.size),
        "mean_max_over_mean": float(imbalances.mean()) if imbalances.size else 0.0,
        "median_max_over_mean": float(np.median(imbalances)) if imbalances.size else 0.0,
        "p90_max_over_mean": float(np.quantile(imbalances, 0.90))
        if imbalances.size
        else 0.0,
        "p95_max_over_mean": float(np.quantile(imbalances, 0.95))
        if imbalances.size
        else 0.0,
        "max_max_over_mean": float(imbalances.max()) if imbalances.size else 0.0,
        "all_batch_layer_max_over_mean": imbalances,
    }
    result["load_profile"] = _profile_load_summary(
        aggregate_expert_load,
        aggregate_expert_load_vision,
        aggregate_expert_load_text,
        aggregate_rank_load,
        aggregate_rank_load_vision,
        aggregate_rank_load_text,
    )
    if args.save_batch_expert_loads:
        counts_path = Path(args.save_batch_expert_loads)
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        batch_counts = np.stack(saved_batch_expert_loads, axis=0)
        np.savez_compressed(
            counts_path,
            batch_expert_loads=batch_counts,
            num_layers=np.asarray([NUM_LAYERS], dtype=np.int64),
            num_experts=np.asarray([NUM_EXPERTS], dtype=np.int64),
            ep_degree=np.asarray([EP_DEGREE], dtype=np.int64),
        )
        meta_path = counts_path.with_suffix(counts_path.suffix + ".meta.json")
        _write_json(
            meta_path,
            {
                "variant": args.variant,
                "datasets": args.datasets,
                "samples_per_dataset": args.samples_per_dataset,
                "batch_size": args.batch_size,
                "shape": list(batch_counts.shape),
                "batches": saved_batch_meta,
            },
        )
        result["batch_expert_loads"] = {
            "path": str(counts_path),
            "meta_path": str(meta_path),
            "shape": list(batch_counts.shape),
        }

    output = Path(args.output_dir) / f"{args.variant}.json"
    _write_json(output, result)
    print(json.dumps({k: result[k] for k in ["variant", "accuracy", "latency", "straggler"]}, indent=2, default=_json_default))


def _load_variant(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_accuracy(asis: dict[str, Any], tobe: dict[str, Any], output: Path) -> None:
    datasets = list(asis["accuracy"]["by_dataset"].keys())
    x = np.arange(len(datasets) + 1)
    labels = datasets + ["Overall"]
    asis_values = [
        asis["accuracy"]["by_dataset"][dataset]["accuracy"] for dataset in datasets
    ] + [asis["accuracy"]["overall"]["accuracy"]]
    tobe_values = [
        tobe["accuracy"]["by_dataset"][dataset]["accuracy"] for dataset in datasets
    ] + [tobe["accuracy"]["overall"]["accuracy"]]
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, asis_values, width, label="As-Is linear", color="#4C78A8")
    ax.bar(x + width / 2, tobe_values, width, label="To-Be layer-wise", color="#59A14F")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, labels)
    ax.set_title("Accuracy: As-Is vs To-Be")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for idx, (a, b) in enumerate(zip(asis_values, tobe_values, strict=True)):
        ax.text(idx - width / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(idx + width / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_latency(asis: dict[str, Any], tobe: dict[str, Any], output: Path) -> None:
    names = ["As-Is linear", "To-Be layer-wise"]
    latency = [
        asis["latency"]["mean_batch_elapsed_s"],
        tobe["latency"]["mean_batch_elapsed_s"],
    ]
    latency_std = [
        asis["latency"]["std_batch_elapsed_s"],
        tobe["latency"]["std_batch_elapsed_s"],
    ]
    throughput = [
        asis["latency"]["prefill_tokens_per_s"],
        tobe["latency"]["prefill_tokens_per_s"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    axes[0].bar(names, latency, yerr=latency_std, color=["#4C78A8", "#59A14F"], capsize=5)
    axes[0].set_title("Prefill-Dominant Batch Latency")
    axes[0].set_ylabel("Seconds / batch")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(names, throughput, color=["#4C78A8", "#59A14F"])
    axes[1].set_title("Prefill Throughput")
    axes[1].set_ylabel("Prompt tokens / second")
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(15)
            label.set_ha("right")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_straggler(asis: dict[str, Any], tobe: dict[str, Any], output: Path) -> None:
    asis_values = np.asarray(
        asis["straggler"]["all_batch_layer_max_over_mean"], dtype=float
    )
    tobe_values = np.asarray(
        tobe["straggler"]["all_batch_layer_max_over_mean"], dtype=float
    )
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.boxplot(
        [asis_values, tobe_values],
        tick_labels=["As-Is linear", "To-Be layer-wise"],
        showfliers=False,
    )
    ax.scatter(
        [1, 2],
        [asis["straggler"]["p95_max_over_mean"], tobe["straggler"]["p95_max_over_mean"]],
        color="#D62728",
        label="p95",
        zorder=3,
    )
    ax.set_title("Batch-Layer EP Rank Imbalance")
    ax.set_ylabel("max rank load / mean rank load")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _profile_array(result: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(result["load_profile"][key], dtype=np.int64)


def _plot_rank_load_profile(
    asis: dict[str, Any], tobe: dict[str, Any], output: Path
) -> None:
    names = ["As-Is linear", "To-Be layer-wise"]
    profiles = [asis["load_profile"], tobe["load_profile"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, name, profile in zip(axes, names, profiles, strict=True):
        vision = np.asarray(profile["total_rank_vision_load"], dtype=float)
        text = np.asarray(profile["total_rank_text_load"], dtype=float)
        total = vision + text
        x = np.arange(EP_DEGREE)
        ax.bar(x, vision, label="vision", color="#4C78A8")
        ax.bar(x, text, bottom=vision, label="text", color="#F28E2B")
        ax.axhline(total.mean(), color="#D62728", linestyle="--", linewidth=1.5, label="ideal")
        ax.set_title(name)
        ax.set_xlabel("EP rank")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25)
        for rank, value in enumerate(total):
            ratio = vision[rank] / value if value > 0 else 0.0
            ax.text(rank, value * 1.01, f"{ratio:.0%}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Routed token-expert assignments")
    axes[0].legend(loc="upper right")
    fig.suptitle("Total EP Rank Load by Placement")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_hot_layer_rank_load(
    asis: dict[str, Any], tobe: dict[str, Any], output: Path
) -> None:
    layer = int(asis["load_profile"]["hot_layer_by_rank_imbalance"]["layer"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, result, title in zip(
        axes,
        [asis, tobe],
        ["As-Is linear", "To-Be layer-wise"],
        strict=True,
    ):
        rank_vis = _profile_array(result, "rank_load_vision_by_layer")[layer].astype(float)
        rank_txt = _profile_array(result, "rank_load_text_by_layer")[layer].astype(float)
        total = rank_vis + rank_txt
        x = np.arange(EP_DEGREE)
        ax.bar(x, rank_vis, label="vision", color="#4C78A8")
        ax.bar(x, rank_txt, bottom=rank_vis, label="text", color="#F28E2B")
        ax.axhline(total.mean(), color="#D62728", linestyle="--", linewidth=1.5, label="ideal")
        ax.set_title(f"{title} layer {layer}")
        ax.set_xlabel("EP rank")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25)
        for rank, value in enumerate(total):
            ratio = rank_vis[rank] / value if value > 0 else 0.0
            ax.text(rank, value * 1.01, f"{ratio:.0%}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Routed token-expert assignments")
    axes[0].legend(loc="upper right")
    fig.suptitle("Hot-Layer EP Rank Load")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_expert_load_profile(
    asis: dict[str, Any], tobe: dict[str, Any], output: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.2), sharex=True, constrained_layout=True)
    for ax, result, title in zip(
        axes,
        [asis, tobe],
        ["As-Is routed expert load", "To-Be routed expert load"],
        strict=True,
    ):
        vision = np.asarray(result["load_profile"]["total_expert_vision_load"], dtype=float)
        text = np.asarray(result["load_profile"]["total_expert_text_load"], dtype=float)
        total = vision + text
        x = np.arange(NUM_EXPERTS)
        ax.bar(x, vision, width=0.9, label="vision", color="#4C78A8")
        ax.bar(x, text, bottom=vision, width=0.9, label="text", color="#F28E2B")
        ax.axhline(total.mean(), color="#D62728", linestyle="--", linewidth=1.2, label="ideal")
        hot = int(total.argmax()) if total.size else 0
        if total.size:
            ratio = vision[hot] / total[hot] if total[hot] > 0 else 0.0
            ax.text(
                hot,
                total[hot] * 1.02,
                f"E{hot}\n{ratio:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_ylabel("Assignments")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Global expert id")
    axes[0].legend(loc="upper right")
    fig.suptitle("Router Expert Load, Placement-Invariant")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_rank_imbalance_by_layer(
    asis: dict[str, Any], tobe: dict[str, Any], output: Path
) -> None:
    asis_load = _profile_array(asis, "rank_load_by_layer")
    tobe_load = _profile_array(tobe, "rank_load_by_layer")
    asis_imb = _imbalance(asis_load)
    tobe_imb = _imbalance(tobe_load)
    layers = np.arange(len(asis_imb))
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(layers, asis_imb, marker="o", linewidth=1.5, markersize=3, label="As-Is linear")
    ax.plot(layers, tobe_imb, marker="o", linewidth=1.5, markersize=3, label="To-Be layer-wise")
    ax.set_title("Layer-Wise EP Rank Imbalance")
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("max rank load / mean rank load")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_summary(asis: dict[str, Any], tobe: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    colors = ["#4C78A8", "#59A14F"]
    names = ["As-Is", "To-Be"]

    acc = [asis["accuracy"]["overall"]["accuracy"], tobe["accuracy"]["overall"]["accuracy"]]
    axes[0].bar(names, acc, color=colors)
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")

    lat = [
        asis["latency"]["mean_batch_elapsed_s"],
        tobe["latency"]["mean_batch_elapsed_s"],
    ]
    axes[1].bar(names, lat, color=colors)
    axes[1].set_title("Latency")
    axes[1].set_ylabel("Seconds / batch")

    imb = [
        asis["straggler"]["p95_max_over_mean"],
        tobe["straggler"]["p95_max_over_mean"],
    ]
    axes[2].bar(names, imb, color=colors)
    axes[2].set_title("Straggler p95")
    axes[2].set_ylabel("max / mean")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    asis = _load_variant(Path(args.asis_json))
    tobe = _load_variant(Path(args.tobe_json))

    accuracy = {
        "asis": asis["accuracy"],
        "tobe": tobe["accuracy"],
        "delta_overall": tobe["accuracy"]["overall"]["accuracy"]
        - asis["accuracy"]["overall"]["accuracy"],
    }
    latency = {
        "asis": asis["latency"],
        "tobe": tobe["latency"],
        "mean_batch_latency_speedup": asis["latency"]["mean_batch_elapsed_s"]
        / max(tobe["latency"]["mean_batch_elapsed_s"], 1e-12),
        "throughput_ratio": tobe["latency"]["prefill_tokens_per_s"]
        / max(asis["latency"]["prefill_tokens_per_s"], 1e-12),
    }
    straggler = {
        "asis": asis["straggler"],
        "tobe": tobe["straggler"],
        "p95_reduction": asis["straggler"]["p95_max_over_mean"]
        - tobe["straggler"]["p95_max_over_mean"],
        "mean_reduction": asis["straggler"]["mean_max_over_mean"]
        - tobe["straggler"]["mean_max_over_mean"],
    }
    _write_json(output_dir / "accuracy.json", accuracy)
    _write_json(output_dir / "latency.json", latency)
    _write_json(output_dir / "straggler.json", straggler)

    _plot_accuracy(asis, tobe, output_dir / "accuracy.png")
    _plot_latency(asis, tobe, output_dir / "latency.png")
    _plot_straggler(asis, tobe, output_dir / "straggler.png")
    _plot_summary(asis, tobe, output_dir / "summary.png")

    load_profile: dict[str, Any] | None = None
    load_figures: dict[str, str] = {}
    if "load_profile" in asis and "load_profile" in tobe:
        load_profile = {
            "asis": asis["load_profile"],
            "tobe": tobe["load_profile"],
            "rank_total_max_over_mean_delta": (
                tobe["load_profile"]["rank_total_stats"]["max_over_mean"]
                - asis["load_profile"]["rank_total_stats"]["max_over_mean"]
            ),
            "rank_layer_mean_max_over_mean_delta": (
                tobe["load_profile"]["rank_layer_imbalance"]["mean_max_over_mean"]
                - asis["load_profile"]["rank_layer_imbalance"]["mean_max_over_mean"]
            ),
            "rank_layer_p95_max_over_mean_delta": (
                tobe["load_profile"]["rank_layer_imbalance"]["p95_max_over_mean"]
                - asis["load_profile"]["rank_layer_imbalance"]["p95_max_over_mean"]
            ),
        }
        _write_json(output_dir / "load_profile.json", load_profile)
        load_figures = {
            "rank_load_profile": str(output_dir / "rank_load_profile.png"),
            "hot_layer_rank_load": str(output_dir / "hot_layer_rank_load.png"),
            "expert_load_profile": str(output_dir / "expert_load_profile.png"),
            "rank_imbalance_by_layer": str(output_dir / "rank_imbalance_by_layer.png"),
        }
        _plot_rank_load_profile(asis, tobe, output_dir / "rank_load_profile.png")
        _plot_hot_layer_rank_load(asis, tobe, output_dir / "hot_layer_rank_load.png")
        _plot_expert_load_profile(asis, tobe, output_dir / "expert_load_profile.png")
        _plot_rank_imbalance_by_layer(asis, tobe, output_dir / "rank_imbalance_by_layer.png")

    summary = {
        "accuracy_delta_overall": accuracy["delta_overall"],
        "latency_speedup_mean_batch": latency["mean_batch_latency_speedup"],
        "throughput_ratio": latency["throughput_ratio"],
        "straggler_p95_reduction": straggler["p95_reduction"],
        "straggler_mean_reduction": straggler["mean_reduction"],
        "figures": {
            "accuracy": str(output_dir / "accuracy.png"),
            "latency": str(output_dir / "latency.png"),
            "straggler": str(output_dir / "straggler.png"),
            "summary": str(output_dir / "summary.png"),
            **load_figures,
        },
    }
    if load_profile is not None:
        summary["rank_total_max_over_mean_delta"] = load_profile[
            "rank_total_max_over_mean_delta"
        ]
        summary["rank_layer_mean_max_over_mean_delta"] = load_profile[
            "rank_layer_mean_max_over_mean_delta"
        ]
        summary["rank_layer_p95_max_over_mean_delta"] = load_profile[
            "rank_layer_p95_max_over_mean_delta"
        ]
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def summarize_load(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    asis = _load_variant(Path(args.asis_json))
    tobe = _load_variant(Path(args.tobe_json))
    if "load_profile" not in asis or "load_profile" not in tobe:
        raise ValueError("both inputs must contain load_profile; run with the patched profiler")

    load_profile = {
        "asis": asis["load_profile"],
        "tobe": tobe["load_profile"],
        "rank_total_max_over_mean_delta": (
            tobe["load_profile"]["rank_total_stats"]["max_over_mean"]
            - asis["load_profile"]["rank_total_stats"]["max_over_mean"]
        ),
        "rank_layer_mean_max_over_mean_delta": (
            tobe["load_profile"]["rank_layer_imbalance"]["mean_max_over_mean"]
            - asis["load_profile"]["rank_layer_imbalance"]["mean_max_over_mean"]
        ),
        "rank_layer_p95_max_over_mean_delta": (
            tobe["load_profile"]["rank_layer_imbalance"]["p95_max_over_mean"]
            - asis["load_profile"]["rank_layer_imbalance"]["p95_max_over_mean"]
        ),
        "figures": {
            "rank_load_profile": str(output_dir / "rank_load_profile.png"),
            "hot_layer_rank_load": str(output_dir / "hot_layer_rank_load.png"),
            "expert_load_profile": str(output_dir / "expert_load_profile.png"),
            "rank_imbalance_by_layer": str(output_dir / "rank_imbalance_by_layer.png"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "load_profile.json", load_profile)
    _plot_rank_load_profile(asis, tobe, output_dir / "rank_load_profile.png")
    _plot_hot_layer_rank_load(asis, tobe, output_dir / "hot_layer_rank_load.png")
    _plot_expert_load_profile(asis, tobe, output_dir / "expert_load_profile.png")
    _plot_rank_imbalance_by_layer(asis, tobe, output_dir / "rank_imbalance_by_layer.png")
    print(json.dumps(load_profile, indent=2, default=_json_default))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--variant", choices=["asis", "tobe"], required=True)
    run.add_argument("--placement-map", required=True)
    run.add_argument("--output-dir", default="outputs/asis_tobe")
    run.add_argument("--model-path", default="models/Qwen3-VL-30B-A3B-Instruct")
    run.add_argument(
        "--datasets",
        nargs="+",
        default=["ChartQA", "MMMU", "MMStar"],
        choices=["ChartQA", "MMMU", "MMStar", "ShareGPT4V"],
    )
    run.add_argument("--samples-per-dataset", type=int, default=64)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--accuracy-max-tokens", type=int, default=16)
    run.add_argument("--tensor-parallel-size", type=int, default=8)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument("--max-num-batched-tokens", type=int, default=8192)
    run.add_argument("--max-num-seqs", type=int, default=8)
    run.add_argument("--max-images-per-prompt", type=int, default=8)
    run.add_argument("--moe-backend", default="auto")
    run.add_argument("--profile-only", action="store_true")
    run.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching for cleaner prefill timing comparisons.",
    )
    run.add_argument(
        "--save-batch-expert-loads",
        default="",
        help="Optional .npz path for batch x layer x expert routed counts.",
    )
    run.add_argument(
        "--moe-cuda-timing-jsonl",
        default="",
        help=(
            "Optional JSONL path for CUDA-event FusedMoE.forward timings. "
            "This is a measurement-only runtime patch."
        ),
    )
    run.add_argument(
        "--moe-cuda-timing-flush-every",
        type=int,
        default=48,
        help="Flush CUDA timing records every N MoE forward calls per worker.",
    )
    run.set_defaults(func=run_variant)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--asis-json", default="outputs/asis_tobe/asis.json")
    summary.add_argument("--tobe-json", default="outputs/asis_tobe/tobe.json")
    summary.add_argument("--output-dir", default="outputs/asis_tobe")
    summary.set_defaults(func=summarize)

    summary_load = subparsers.add_parser("summarize-load")
    summary_load.add_argument("--asis-json", required=True)
    summary_load.add_argument("--tobe-json", required=True)
    summary_load.add_argument("--output-dir", default="outputs/asis_tobe_load")
    summary_load.set_defaults(func=summarize_load)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
