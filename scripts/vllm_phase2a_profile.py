"""Phase 2-A vLLM EP motivation and calibration profiling.

This script only observes vanilla vLLM expert parallel routing. It does not
modify placement, dispatch, experts, de-RoPE, merge, cap, or generation logic.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
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
from vllm import LLM, SamplingParams

from measure.ep_load import count_expert_load, load_imbalance


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
NUM_LAYERS = 48
NUM_EXPERTS = 128
TOPK = 8
EP_DEGREE = 8


@dataclass
class ProfileSample:
    dataset: str
    sample_id: str
    images: list[Image.Image]
    question: str
    metadata: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=_json_default),
        encoding="utf-8",
    )


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
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    return None


def _image_area_from_cell(value: Any) -> int:
    image = _image_from_cell(value)
    if image is None:
        return 0
    width, height = image.size
    return int(width * height)


def _load_chartqa(limit: int) -> list[ProfileSample]:
    path = "data/benchmarks/ChartQA/data/test-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    df = df.assign(_area=df["image"].map(_image_area_from_cell))
    df = df.sort_values("_area", ascending=False).head(limit)
    samples = []
    for idx, row in df.iterrows():
        image = _image_from_cell(row["image"])
        if image is None:
            continue
        samples.append(
            ProfileSample(
                dataset="ChartQA",
                sample_id=f"chartqa_{idx}",
                images=[image],
                question=f"Answer the chart question briefly: {row['question']}",
                metadata={"area": int(row["_area"]), "answer": _safe_text(row["answer"])},
            )
        )
    return samples


def _load_textvqa(limit: int) -> list[ProfileSample]:
    path = "data/benchmarks/TextVQA/data/validation-00000-of-00003.parquet"
    df = pd.read_parquet(path)
    df = df.assign(_area=(df["image_width"].astype(int) * df["image_height"].astype(int)))
    df = df.sort_values("_area", ascending=False).head(limit)
    samples = []
    for _, row in df.iterrows():
        image = _image_from_cell(row["image"])
        if image is None:
            continue
        samples.append(
            ProfileSample(
                dataset="TextVQA",
                sample_id=f"textvqa_{row['question_id']}",
                images=[image],
                question=f"Answer using the visible text in the image: {row['question']}",
                metadata={"area": int(row["_area"]), "image_id": _safe_text(row["image_id"])},
            )
        )
    return samples


def _load_mmmu(limit: int) -> list[ProfileSample]:
    path = "data/benchmarks/MMMU/data/validation-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    rows: list[tuple[int, int, int, Any]] = []
    for idx, row in df.iterrows():
        image_count = 0
        total_area = 0
        for i in range(1, 8):
            cell = row.get(f"image_{i}")
            area = _image_area_from_cell(cell)
            if area:
                image_count += 1
                total_area += area
        rows.append((image_count, total_area, int(idx), row))
    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

    samples = []
    for image_count, total_area, idx, row in rows[:limit]:
        images = []
        for i in range(1, 8):
            image = _image_from_cell(row.get(f"image_{i}"))
            if image is not None:
                images.append(image)
        if not images:
            continue
        prompt = (
            "Solve the visual multiple-choice question. "
            f"Question: {row['question']} Options: {row['options']} "
            "Return only the answer letter or short answer."
        )
        samples.append(
            ProfileSample(
                dataset="MMMU",
                sample_id=f"mmmu_{row['id']}",
                images=images,
                question=prompt,
                metadata={
                    "image_count": image_count,
                    "area": total_area,
                    "answer": _safe_text(row["answer"]),
                },
            )
        )
    return samples


def _load_mmbench(limit: int) -> list[ProfileSample]:
    path = "data/benchmarks/MMBench/en/dev-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    df = df.assign(_area=df["image"].map(_image_area_from_cell))
    df = df.sort_values("_area", ascending=False).head(limit)
    samples = []
    for _, row in df.iterrows():
        image = _image_from_cell(row["image"])
        if image is None:
            continue
        options = []
        for letter in ["A", "B", "C", "D"]:
            text = _safe_text(row.get(letter))
            if text:
                options.append(f"{letter}. {text}")
        hint = _safe_text(row.get("hint"))
        prompt_parts = [
            "Answer the multiple-choice visual question.",
            f"Question: {row['question']}",
        ]
        if hint:
            prompt_parts.append(f"Context: {hint}")
        if options:
            prompt_parts.append("Options: " + " ".join(options))
        prompt_parts.append("Return only the answer letter.")
        samples.append(
            ProfileSample(
                dataset="MMBench",
                sample_id=f"mmbench_{row['index']}",
                images=[image],
                question="\n".join(prompt_parts),
                metadata={"area": int(row["_area"]), "answer": _safe_text(row["answer"])},
            )
        )
    return samples


def _load_motivation_samples(limit_per_dataset: int) -> list[ProfileSample]:
    samples: list[ProfileSample] = []
    samples.extend(_load_mmmu(limit_per_dataset))
    samples.extend(_load_chartqa(limit_per_dataset))
    samples.extend(_load_textvqa(limit_per_dataset))
    samples.extend(_load_mmbench(limit_per_dataset))
    return samples


def _load_sharegpt4v(limit: int) -> list[ProfileSample]:
    manifest = Path("data/sharegpt4v_512/manifest.jsonl")
    samples = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if len(samples) >= limit:
                break
            row = json.loads(line)
            image_path = Path(row["local_image"])
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            question = ""
            for turn in row.get("conversations", []):
                if turn.get("from") == "human":
                    question = str(turn.get("value", ""))
                    break
            question = re.sub(r"<image>", "", question).strip()
            if not question:
                question = "Describe the image in detail."
            samples.append(
                ProfileSample(
                    dataset="ShareGPT4V",
                    sample_id=str(row["id"]),
                    images=[image],
                    question=question,
                    metadata={
                        "image": row.get("image", ""),
                        "image_url": row.get("image_url", ""),
                    },
                )
            )
    return samples


def _build_prompt(
    processor: AutoProcessor,
    sample: ProfileSample,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image} for image in sample.images
    ]
    content.append({"type": "text", "text": sample.question})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images: Image.Image | list[Image.Image]
    images = sample.images[0] if len(sample.images) == 1 else sample.images
    return {"prompt": prompt, "multi_modal_data": {"image": images}}


def _init_llm(args: argparse.Namespace) -> LLM:
    return LLM(
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
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
    )


def _prefill_routing(
    request: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    completion = request.outputs[0]
    routed = completion.routed_experts
    if routed is None:
        raise RuntimeError("vLLM did not return routed_experts")
    token_ids = np.asarray(request.prompt_token_ids or [], dtype=np.int64)
    n = min(len(token_ids), routed.shape[0])
    if n <= 0:
        raise RuntimeError("empty prompt routing capture")
    token_ids = token_ids[:n]
    routed = np.asarray(routed[:n], dtype=np.int32)
    vision_mask = np.isin(token_ids, [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID])
    info = {
        "prompt_token_count": int(len(request.prompt_token_ids or [])),
        "prefill_token_count": int(n),
        "routed_seq_len": int(completion.routed_experts.shape[0]),
        "vision_token_count": int(vision_mask.sum()),
        "text_token_count": int((~vision_mask).sum()),
    }
    return routed, vision_mask, info


def _empty_counts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64),
        np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64),
        np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64),
    )


def _accumulate_layer_counts(
    routed: np.ndarray,
    vision_mask: np.ndarray,
    trace: np.ndarray,
    vision: np.ndarray,
    text: np.ndarray,
) -> None:
    layers = min(routed.shape[1], trace.shape[0])
    for layer in range(layers):
        layer_ids = routed[:, layer, :]
        trace[layer] += np.bincount(
            layer_ids.reshape(-1), minlength=NUM_EXPERTS
        )[:NUM_EXPERTS]
        if vision_mask.any():
            vision[layer] += np.bincount(
                layer_ids[vision_mask].reshape(-1), minlength=NUM_EXPERTS
            )[:NUM_EXPERTS]
        if (~vision_mask).any():
            text[layer] += np.bincount(
                layer_ids[~vision_mask].reshape(-1), minlength=NUM_EXPERTS
            )[:NUM_EXPERTS]


def _plot_modality_ratio(samples: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets = sorted({sample["dataset"] for sample in samples})
    ratios = np.asarray([sample["vision_ratio"] for sample in samples]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].hist(ratios, bins=np.linspace(0, 100, 21), color="#4C78A8", alpha=0.85)
    axes[0].axvline(ratios.mean(), color="#D62728", linestyle="--", linewidth=2)
    axes[0].set_title("Input Vision Token Ratio")
    axes[0].set_xlabel("Vision tokens / prefill tokens (%)")
    axes[0].set_ylabel("Number of inputs")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(
        ratios.mean(),
        axes[0].get_ylim()[1] * 0.92,
        f"mean {ratios.mean():.1f}%",
        color="#D62728",
        ha="left",
        va="top",
    )

    grouped = [
        np.asarray(
            [sample["vision_ratio"] * 100.0 for sample in samples if sample["dataset"] == ds]
        )
        for ds in datasets
    ]
    axes[1].boxplot(grouped, tick_labels=datasets, showfliers=False)
    rng = np.random.default_rng(0)
    for idx, values in enumerate(grouped, start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        axes[1].scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=14,
            alpha=0.45,
            color="#59A14F",
            edgecolors="none",
        )
    axes[1].set_title("Vision Ratio by Dataset")
    axes[1].set_ylabel("Vision tokens / prefill tokens (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25)
    for label in axes[1].get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_stacked_rank(
    vision: np.ndarray,
    text: np.ndarray,
    output: Path,
    *,
    title: str = "EP Rank Load by Token Modality",
) -> None:
    total = vision + text
    ideal = total.sum() / len(total)
    ratio = np.divide(vision, total, out=np.zeros_like(vision, dtype=float), where=total > 0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(total))
    ax.bar(x, vision, label="vision routed assignments", color="#4C78A8")
    ax.bar(x, text, bottom=vision, label="text/control routed assignments", color="#F28E2B")
    ax.axhline(ideal, color="#222222", linestyle="--", linewidth=1.6, label="ideal load")
    for idx, value in enumerate(total):
        ax.text(
            idx,
            value + max(total) * 0.015,
            f"{ratio[idx] * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_xlabel("EP rank (linear expert placement)")
    ax.set_ylabel("Routed assignments (tokens x layers x top-k)")
    ax.set_xticks(x)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_stacked_expert(
    vision: np.ndarray,
    text: np.ndarray,
    output: Path,
    *,
    title: str = "Expert Load by Token Modality",
) -> None:
    total = vision + text
    ideal = total.sum() / len(total)
    ratio = np.divide(vision, total, out=np.zeros_like(vision, dtype=float), where=total > 0)
    hot = np.argsort(-total)[:8]

    fig, ax = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    x = np.arange(len(total))
    ax.bar(x, vision, label="vision routed assignments", color="#4C78A8", width=0.85)
    ax.bar(x, text, bottom=vision, label="text/control routed assignments", color="#F28E2B", width=0.85)
    ax.axhline(ideal, color="#222222", linestyle="--", linewidth=1.6, label="ideal load")
    for expert in hot[:5]:
        ax.text(
            expert,
            total[expert] + max(total) * 0.02,
            f"E{expert}\n{ratio[expert] * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(title)
    ax.set_xlabel("Expert id")
    ax.set_ylabel("Routed assignments (tokens x layers x top-k)")
    ax.set_xlim(-1, len(total))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_heatmap(
    matrix: np.ndarray,
    output: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    kwargs: dict[str, Any] = {"aspect": "auto", "interpolation": "nearest", "cmap": cmap}
    if center_zero:
        bound = float(np.max(np.abs(matrix)))
        kwargs.update({"vmin": -bound, "vmax": bound})
    im = ax.imshow(matrix, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Expert id")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(0, matrix.shape[1], 16))
    ax.set_yticks(np.arange(0, matrix.shape[0], 4))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _summarize_by_dataset(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in sorted({sample["dataset"] for sample in samples}):
        rows = [sample for sample in samples if sample["dataset"] == dataset]
        ratios = np.asarray([row["vision_ratio"] for row in rows], dtype=float)
        vision = int(sum(row["vision_token_count"] for row in rows))
        text = int(sum(row["text_token_count"] for row in rows))
        out[dataset] = {
            "num_samples": len(rows),
            "vision_tokens": vision,
            "text_tokens": text,
            "mean_vision_ratio": float(ratios.mean()) if ratios.size else 0.0,
            "median_vision_ratio": float(np.median(ratios)) if ratios.size else 0.0,
            "p90_vision_ratio": float(np.quantile(ratios, 0.9)) if ratios.size else 0.0,
        }
    return out


def _load_summary(vision: np.ndarray, text: np.ndarray) -> dict[str, Any]:
    total = vision + text
    ratio = np.divide(vision, total, out=np.zeros_like(vision, dtype=float), where=total > 0)
    hot = int(total.argmax()) if total.size else -1
    summary = load_imbalance(total)
    summary.update(
        {
            "total_load": int(total.sum()),
            "vision_load": int(vision.sum()),
            "text_load": int(text.sum()),
            "mean_vision_ratio": float(vision.sum() / max(1, total.sum())),
            "hot_vision_ratio": float(ratio[hot]) if hot >= 0 else 0.0,
            "hot_load": int(total[hot]) if hot >= 0 else 0,
        }
    )
    return summary


def _rank_by_layer(expert_by_layer: np.ndarray) -> np.ndarray:
    return expert_by_layer.reshape(NUM_LAYERS, EP_DEGREE, NUM_EXPERTS // EP_DEGREE).sum(axis=2)


def _select_hot_straggler_scope(
    sample_rows: list[dict[str, Any]],
    sample_vision_layers: list[np.ndarray],
    sample_text_layers: list[np.ndarray],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Select the batch/layer with the highest rank max/mean imbalance."""

    best: dict[str, Any] | None = None
    n = len(sample_rows)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        vision_layer = np.sum(sample_vision_layers[start:end], axis=0)
        text_layer = np.sum(sample_text_layers[start:end], axis=0)
        rank_vision_layer = _rank_by_layer(vision_layer)
        rank_text_layer = _rank_by_layer(text_layer)
        rank_total_layer = rank_vision_layer + rank_text_layer
        means = rank_total_layer.mean(axis=1)
        maxes = rank_total_layer.max(axis=1)
        imbalance = np.divide(maxes, means, out=np.zeros_like(maxes, dtype=float), where=means > 0)
        layer = int(imbalance.argmax())
        score = float(imbalance[layer])
        if best is None or score > best["rank_summary"]["max_over_mean"]:
            datasets = sorted({row["dataset"] for row in sample_rows[start:end]})
            best = {
                "batch_start": start,
                "batch_end": end,
                "batch_size": end - start,
                "datasets": datasets,
                "layer": layer,
                "rank_vision": rank_vision_layer[layer].astype(np.int64),
                "rank_text": rank_text_layer[layer].astype(np.int64),
                "expert_vision": vision_layer[layer].astype(np.int64),
                "expert_text": text_layer[layer].astype(np.int64),
                "rank_summary": _load_summary(
                    rank_vision_layer[layer], rank_text_layer[layer]
                ),
                "expert_summary": _load_summary(vision_layer[layer], text_layer[layer]),
            }

    if best is None:
        raise RuntimeError("no samples available for straggler scope selection")
    return best


def run_motivation(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    samples = _load_motivation_samples(args.samples_per_dataset)
    prompts = [_build_prompt(processor, sample) for sample in samples]

    result: dict[str, Any] = {
        "note": "Phase 2-A measurement only; vanilla vLLM EP routing, prefill tokens only.",
        "model_path": args.model_path,
        "settings": _settings_dict(args),
        "num_samples_requested_per_dataset": args.samples_per_dataset,
        "num_prompts": len(prompts),
        "memory_before": _gpu_memory(),
    }
    llm = _init_llm(args)
    result["memory_after_load"] = _gpu_memory()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=1, temperature=0.0),
        use_tqdm=True,
    )
    result["memory_after_generate"] = _gpu_memory()

    trace, vision_layer, text_layer = _empty_counts()
    rank_vision = np.zeros(EP_DEGREE, dtype=np.int64)
    rank_text = np.zeros(EP_DEGREE, dtype=np.int64)
    expert_vision = np.zeros(NUM_EXPERTS, dtype=np.int64)
    expert_text = np.zeros(NUM_EXPERTS, dtype=np.int64)
    sample_rows = []
    hot_rank_rows = []
    sample_vision_layers: list[np.ndarray] = []
    sample_text_layers: list[np.ndarray] = []

    for sample, request in zip(samples, outputs, strict=True):
        routed, vision_mask, info = _prefill_routing(request)
        breakdown = count_expert_load(
            routed,
            vision_mask,
            num_experts=NUM_EXPERTS,
            ep_degree=EP_DEGREE,
        )
        rank_vision += breakdown.rank_vision
        rank_text += breakdown.rank_text
        expert_vision += breakdown.expert_vision
        expert_text += breakdown.expert_text
        local_trace, local_vision_layer, local_text_layer = _empty_counts()
        _accumulate_layer_counts(
            routed,
            vision_mask,
            local_trace,
            local_vision_layer,
            local_text_layer,
        )
        trace += local_trace
        vision_layer += local_vision_layer
        text_layer += local_text_layer
        sample_vision_layers.append(local_vision_layer)
        sample_text_layers.append(local_text_layer)

        vision_tokens = info["vision_token_count"]
        text_tokens = info["text_token_count"]
        total_tokens = max(1, vision_tokens + text_tokens)
        rank_total = breakdown.rank_total
        hot_rank = int(rank_total.argmax())
        sample_rows.append(
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "image_count": len(sample.images),
                "prompt_token_count": info["prompt_token_count"],
                "prefill_token_count": info["prefill_token_count"],
                "routed_seq_len": info["routed_seq_len"],
                "vision_token_count": vision_tokens,
                "text_token_count": text_tokens,
                "vision_ratio": float(vision_tokens / total_tokens),
                "metadata": sample.metadata,
            }
        )
        hot_rank_rows.append(
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "hot_rank": hot_rank,
                "hot_rank_load": int(rank_total[hot_rank]),
                "rank_load": rank_total.tolist(),
            }
        )

    total_vision = int(sum(row["vision_token_count"] for row in sample_rows))
    total_text = int(sum(row["text_token_count"] for row in sample_rows))
    ratios = np.asarray([row["vision_ratio"] for row in sample_rows], dtype=float)
    result["summary"] = {
        "num_samples": len(sample_rows),
        "vision_tokens": total_vision,
        "text_tokens": total_text,
        "mean_vision_ratio": float(ratios.mean()),
        "median_vision_ratio": float(np.median(ratios)),
        "p90_vision_ratio": float(np.quantile(ratios, 0.9)),
        "by_dataset": _summarize_by_dataset(sample_rows),
    }
    result["samples"] = sample_rows
    result["hot_rank_by_sample"] = hot_rank_rows

    _write_json(output_dir / "token_modality_ratio.json", result)
    _plot_modality_ratio(sample_rows, output_dir / "token_modality_ratio.png")

    hot_scope = _select_hot_straggler_scope(
        sample_rows,
        sample_vision_layers,
        sample_text_layers,
        batch_size=args.straggler_batch_size,
    )
    hot_rank_vision = hot_scope["rank_vision"]
    hot_rank_text = hot_scope["rank_text"]
    hot_expert_vision = hot_scope["expert_vision"]
    hot_expert_text = hot_scope["expert_text"]

    rank_report = {
        "note": "Linear EP placement, rank = expert_id // 16. Counts include top-k multiplicity. The plotted scope is the batch/layer with the highest per-rank max/mean load.",
        "plotted_scope": {
            "batch_start": hot_scope["batch_start"],
            "batch_end": hot_scope["batch_end"],
            "batch_size": hot_scope["batch_size"],
            "datasets": hot_scope["datasets"],
            "layer": hot_scope["layer"],
        },
        "rank_vision": hot_rank_vision.tolist(),
        "rank_text": hot_rank_text.tolist(),
        "rank_total": (hot_rank_vision + hot_rank_text).tolist(),
        "rank_vision_ratio": np.divide(
            hot_rank_vision,
            hot_rank_vision + hot_rank_text,
            out=np.zeros(EP_DEGREE, dtype=float),
            where=(hot_rank_vision + hot_rank_text) > 0,
        ).tolist(),
        "summary": hot_scope["rank_summary"],
        "aggregate_all_layers": {
            "rank_vision": rank_vision.tolist(),
            "rank_text": rank_text.tolist(),
            "rank_total": (rank_vision + rank_text).tolist(),
            "summary": _load_summary(rank_vision, rank_text),
        },
        "hot_rank_by_sample": hot_rank_rows,
    }
    expert_report = {
        "note": "Counts include top-k multiplicity. The plotted scope matches ep_straggler_rank.json.",
        "plotted_scope": {
            "batch_start": hot_scope["batch_start"],
            "batch_end": hot_scope["batch_end"],
            "batch_size": hot_scope["batch_size"],
            "datasets": hot_scope["datasets"],
            "layer": hot_scope["layer"],
        },
        "expert_vision": hot_expert_vision.tolist(),
        "expert_text": hot_expert_text.tolist(),
        "expert_total": (hot_expert_vision + hot_expert_text).tolist(),
        "expert_vision_ratio": np.divide(
            hot_expert_vision,
            hot_expert_vision + hot_expert_text,
            out=np.zeros(NUM_EXPERTS, dtype=float),
            where=(hot_expert_vision + hot_expert_text) > 0,
        ).tolist(),
        "summary": hot_scope["expert_summary"],
        "top_experts": np.argsort(-(hot_expert_vision + hot_expert_text))[:16].astype(int).tolist(),
        "aggregate_all_layers": {
            "expert_vision": expert_vision.tolist(),
            "expert_text": expert_text.tolist(),
            "expert_total": (expert_vision + expert_text).tolist(),
            "summary": _load_summary(expert_vision, expert_text),
        },
    }
    _write_json(output_dir / "ep_straggler_rank.json", rank_report)
    _write_json(output_dir / "ep_straggler_expert.json", expert_report)
    scope_title = (
        f"EP Rank Load by Token Modality "
        f"(batch {hot_scope['batch_start']}-{hot_scope['batch_end'] - 1}, "
        f"layer {hot_scope['layer']})"
    )
    _plot_stacked_rank(
        hot_rank_vision,
        hot_rank_text,
        output_dir / "ep_straggler_rank.png",
        title=scope_title,
    )
    _plot_stacked_expert(
        hot_expert_vision,
        hot_expert_text,
        output_dir / "ep_straggler_expert.png",
        title=(
            f"Expert Load by Token Modality "
            f"(batch {hot_scope['batch_start']}-{hot_scope['batch_end'] - 1}, "
            f"layer {hot_scope['layer']})"
        ),
    )

    print(json.dumps({"M1": result["summary"], "M2_rank": rank_report["summary"], "M2_expert": expert_report["summary"]}, indent=2))


def _settings_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
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
        "max_images_per_prompt": args.max_images_per_prompt,
    }


def _normalize_layer_distribution(counts: np.ndarray) -> np.ndarray:
    denom = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, denom, out=np.zeros_like(counts, dtype=float), where=denom > 0)


def _write_dist_table(path: Path, dist_vis: np.ndarray, dist_txt: np.ndarray) -> None:
    tv = 0.5 * np.abs(dist_vis - dist_txt).sum(axis=1)
    selected = sorted(set([0, 8, 16, 24, 32, 40, 47] + np.argsort(-tv)[:5].astype(int).tolist()))
    lines = [
        "# Vision/Text Expert Preference Table",
        "",
        "Probabilities are normalized over experts within each layer.",
        "",
        "| layer | TV distance | top vision experts | top text experts |",
        "| ---: | ---: | --- | --- |",
    ]
    for layer in selected:
        top_v = np.argsort(-dist_vis[layer])[:5]
        top_t = np.argsort(-dist_txt[layer])[:5]
        v_text = ", ".join(f"E{e}:{dist_vis[layer, e]:.3f}" for e in top_v)
        t_text = ", ".join(f"E{e}:{dist_txt[layer, e]:.3f}" for e in top_t)
        lines.append(f"| {layer} | {tv[layer]:.3f} | {v_text} | {t_text} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gating_todo(output_dir: Path) -> None:
    text = """# P2 Gating Score Status

Status: TODO(Phase2B or deeper vLLM hook).

vLLM 0.20 exposes `CompletionOutput.routed_experts` when
`enable_return_routed_experts=True`, but that path stores only top-k expert ids.
The local source path is:

- `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`
- `RoutedExpertsCapturer.capture(layer_id, topk_ids)`
- output field: `CompletionOutput.routed_experts`

`topk_weights`/router softmax values exist inside the fused MoE routing path, but
are not exported by the public routed-expert capture API. This phase does not
patch vLLM fused kernels or alter the model, so P2 gating heatmaps are left as
TODO while P1/P3 use actual vLLM EP routed expert ids.
"""
    (output_dir / "gating_score_todo.md").write_text(text, encoding="utf-8")
    _write_json(
        output_dir / "gating_score_todo.json",
        {
            "status": "TODO",
            "reason": "vLLM 0.20 routed-expert capture stores topk_ids only, not topk_weights/router softmax.",
            "source_evidence": [
                "RoutedExpertsCapturer.capture(layer_id, topk_ids)",
                "CompletionOutput.routed_experts shape [seq_len, layer_num, topk]",
            ],
        },
    )


def run_calibration(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    samples = _load_sharegpt4v(args.limit)
    prompts = [_build_prompt(processor, sample) for sample in samples]

    result: dict[str, Any] = {
        "note": "ShareGPT4V calibration profiling; vanilla vLLM EP routing, prefill tokens only.",
        "model_path": args.model_path,
        "settings": _settings_dict(args),
        "num_prompts": len(prompts),
        "memory_before": _gpu_memory(),
    }
    llm = _init_llm(args)
    result["memory_after_load"] = _gpu_memory()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=1, temperature=0.0),
        use_tqdm=True,
    )
    result["memory_after_generate"] = _gpu_memory()

    trace, vision_layer, text_layer = _empty_counts()
    sample_rows = []
    for sample, request in zip(samples, outputs, strict=True):
        routed, vision_mask, info = _prefill_routing(request)
        _accumulate_layer_counts(routed, vision_mask, trace, vision_layer, text_layer)
        vision_tokens = info["vision_token_count"]
        text_tokens = info["text_token_count"]
        total_tokens = max(1, vision_tokens + text_tokens)
        sample_rows.append(
            {
                "sample_id": sample.sample_id,
                "prompt_token_count": info["prompt_token_count"],
                "prefill_token_count": info["prefill_token_count"],
                "routed_seq_len": info["routed_seq_len"],
                "vision_token_count": vision_tokens,
                "text_token_count": text_tokens,
                "vision_ratio": float(vision_tokens / total_tokens),
            }
        )

    dist_vis = _normalize_layer_distribution(vision_layer)
    dist_txt = _normalize_layer_distribution(text_layer)
    diff = dist_vis - dist_txt
    tv = 0.5 * np.abs(diff).sum(axis=1)

    np.save(output_dir / "trace_freq.npy", trace)
    np.save(output_dir / "dist_vision.npy", dist_vis)
    np.save(output_dir / "dist_text.npy", dist_txt)
    np.save(output_dir / "dist_diff.npy", diff)

    _plot_heatmap(
        np.log1p(trace),
        output_dir / "trace_freq.png",
        title="Expert Selection Frequency by Layer (log1p)",
        colorbar_label="log(1 + routed assignment count)",
    )
    _plot_heatmap(
        dist_vis,
        output_dir / "dist_vision.png",
        title="P_vis: Vision Token Expert Distribution",
        colorbar_label="Probability within layer",
    )
    _plot_heatmap(
        dist_txt,
        output_dir / "dist_text.png",
        title="P_txt: Text Token Expert Distribution",
        colorbar_label="Probability within layer",
    )
    _plot_heatmap(
        diff,
        output_dir / "dist_diff.png",
        title="P_vis - P_txt Expert Preference Difference",
        colorbar_label="Probability difference",
        cmap="coolwarm",
        center_zero=True,
    )
    _write_dist_table(output_dir / "dist_table.md", dist_vis, dist_txt)
    _write_gating_todo(output_dir)

    ratios = np.asarray([row["vision_ratio"] for row in sample_rows], dtype=float)
    result["summary"] = {
        "num_samples": len(sample_rows),
        "vision_tokens": int(sum(row["vision_token_count"] for row in sample_rows)),
        "text_tokens": int(sum(row["text_token_count"] for row in sample_rows)),
        "mean_vision_ratio": float(ratios.mean()) if ratios.size else 0.0,
        "trace_total_assignments": int(trace.sum()),
        "dist_mean_abs_diff": float(np.abs(diff).mean()),
        "dist_mean_total_variation": float(tv.mean()),
        "dist_max_total_variation": float(tv.max()),
        "dist_top_tv_layers": np.argsort(-tv)[:8].astype(int).tolist(),
        "p2_gating_score_status": "TODO: vLLM routed-expert API exposes ids, not gating scores.",
    }
    result["samples"] = sample_rows
    _write_json(output_dir / "calibration_summary.json", result)

    print(json.dumps(result["summary"], indent=2))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-images-per-prompt", type=int, default=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    motivation = subparsers.add_parser("motivation")
    _add_common_args(motivation)
    motivation.add_argument("--samples-per-dataset", type=int, default=64)
    motivation.add_argument("--straggler-batch-size", type=int, default=8)
    motivation.add_argument("--output-dir", default="outputs/motivation")
    motivation.set_defaults(func=run_motivation)

    calibration = subparsers.add_parser("calibration")
    _add_common_args(calibration)
    calibration.add_argument("--limit", type=int, default=512)
    calibration.add_argument("--output-dir", default="outputs/calibration")
    calibration.set_defaults(func=run_calibration)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
