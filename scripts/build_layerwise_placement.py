"""Build layer-wise modality-balanced expert placement maps.

Inputs are Phase 2-A P3 distributions:

- ``dist_vision.npy``: P_vis[layer, expert]
- ``dist_text.npy``: P_txt[layer, expert]

Outputs are vLLM-ready JSON maps shaped as ``{layer: {expert: rank}}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method1.placement import compute_loads, contiguous_placement, lpt_placement


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


def _map_to_json(mapping: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        str(layer): {
            str(expert): int(mapping[layer, expert])
            for expert in range(mapping.shape[1])
        }
        for layer in range(mapping.shape[0])
    }


def _validate(mapping: np.ndarray, *, num_ranks: int, experts_per_rank: int) -> None:
    if mapping.ndim != 2:
        raise ValueError("mapping must have shape [layer, expert]")
    for layer in range(mapping.shape[0]):
        counts = np.bincount(mapping[layer], minlength=num_ranks)
        if len(counts) != num_ranks or not np.all(counts == experts_per_rank):
            raise ValueError(
                f"layer {layer} must assign {experts_per_rank} experts/rank; "
                f"got {counts.tolist()}"
            )


def _imbalance(load: np.ndarray) -> dict[str, float | int]:
    mean = float(load.mean())
    max_load = float(load.max())
    min_load = float(load.min())
    return {
        "mean": mean,
        "max": max_load,
        "min": min_load,
        "max_over_mean": float(max_load / mean) if mean > 0 else 0.0,
        "hot_rank": int(load.argmax()),
    }


def _classify(vis: np.ndarray, txt: np.ndarray, delta: float) -> np.ndarray:
    rel = (vis - txt) / np.maximum(vis + txt, 1e-12)
    labels = np.full(vis.shape, "shared", dtype=object)
    labels[rel >= delta] = "vision"
    labels[rel <= -delta] = "text"
    return labels


def build(args: argparse.Namespace) -> None:
    dist_vision = np.load(args.dist_vision).astype(np.float64)
    dist_text = np.load(args.dist_text).astype(np.float64)
    if dist_vision.shape != dist_text.shape:
        raise ValueError("dist_vision and dist_text must have the same shape")

    num_layers, num_experts = dist_vision.shape
    if num_experts % args.num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")
    experts_per_rank = num_experts // args.num_ranks

    balanced = np.empty((num_layers, num_experts), dtype=np.int64)
    linear = np.empty_like(balanced)
    labels = _classify(dist_vision, dist_text, args.delta)
    layer_summaries: list[dict[str, Any]] = []

    for layer in range(num_layers):
        weights = torch.from_numpy(dist_vision[layer].astype(np.float32))
        placement = lpt_placement(
            weights,
            num_gpus=args.num_ranks,
            max_experts_per_gpu=experts_per_rank,
        )
        balanced[layer] = placement.expert_to_gpu.numpy()

        linear_map = contiguous_placement(num_experts, args.num_ranks)
        linear[layer] = linear_map.numpy()

        balanced_load = compute_loads(
            weights, placement.expert_to_gpu, args.num_ranks
        ).numpy()
        linear_load = compute_loads(weights, linear_map, args.num_ranks).numpy()
        label_values, label_counts = np.unique(labels[layer], return_counts=True)
        layer_summaries.append(
            {
                "layer": layer,
                "classification_counts": {
                    str(label): int(count)
                    for label, count in zip(label_values, label_counts, strict=True)
                },
                "linear_vision_load": linear_load,
                "balanced_vision_load": balanced_load,
                "linear_imbalance": _imbalance(linear_load),
                "balanced_imbalance": _imbalance(balanced_load),
                "top_vision_experts": np.argsort(-dist_vision[layer])[:10],
                "top_text_experts": np.argsort(-dist_text[layer])[:10],
            }
        )

    _validate(balanced, num_ranks=args.num_ranks, experts_per_rank=experts_per_rank)
    _validate(linear, num_ranks=args.num_ranks, experts_per_rank=experts_per_rank)

    output_dir = Path(args.output_dir)
    _write_json(
        output_dir / "modality_balanced_map_perlayer.json",
        _map_to_json(balanced),
    )
    _write_json(output_dir / "linear_map_perlayer.json", _map_to_json(linear))
    _write_json(
        output_dir / "expert_modality_perlayer.json",
        {
            str(layer): {
                str(expert): str(labels[layer, expert])
                for expert in range(num_experts)
            }
            for layer in range(num_layers)
        },
    )

    summary = {
        "note": (
            "Layer-wise LPT placement using P_vis[layer, expert] as expected "
            "vision routing load. Each layer/rank owns exactly 16 experts."
        ),
        "dist_vision": args.dist_vision,
        "dist_text": args.dist_text,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_ranks": args.num_ranks,
        "experts_per_rank": experts_per_rank,
        "delta_relative_modality_score": args.delta,
        "layer_9_differs_from_layer_20": bool(
            not np.array_equal(balanced[9], balanced[20])
            if num_layers > 20
            else False
        ),
        "mean_linear_max_over_mean": float(
            np.mean([row["linear_imbalance"]["max_over_mean"] for row in layer_summaries])
        ),
        "mean_balanced_max_over_mean": float(
            np.mean(
                [row["balanced_imbalance"]["max_over_mean"] for row in layer_summaries]
            )
        ),
        "max_linear_max_over_mean": float(
            np.max([row["linear_imbalance"]["max_over_mean"] for row in layer_summaries])
        ),
        "max_balanced_max_over_mean": float(
            np.max(
                [row["balanced_imbalance"]["max_over_mean"] for row in layer_summaries]
            )
        ),
        "layers": layer_summaries,
    }
    _write_json(output_dir / "placement_summary.json", summary)

    np.save(output_dir / "modality_balanced_map_perlayer.npy", balanced)
    np.save(output_dir / "linear_map_perlayer.npy", linear)
    print(json.dumps(summary, indent=2, default=_json_default))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-vision", default="outputs/calibration/dist_vision.npy")
    parser.add_argument("--dist-text", default="outputs/calibration/dist_text.npy")
    parser.add_argument("--output-dir", default="outputs/placement")
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.1)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
