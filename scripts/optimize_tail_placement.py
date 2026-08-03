"""Build layer-wise expert placement by minimizing batch-tail rank imbalance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

NUM_LAYERS = 48
NUM_EXPERTS = 128
EP_DEGREE = 8
EXPERTS_PER_RANK = NUM_EXPERTS // EP_DEGREE


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


def linear_mapping() -> np.ndarray:
    mapping = np.empty((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    base = np.arange(NUM_EXPERTS, dtype=np.int64) // EXPERTS_PER_RANK
    for layer in range(NUM_LAYERS):
        mapping[layer] = base
    return mapping


def mapping_to_json(mapping: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        str(layer): {str(expert): int(mapping[layer, expert]) for expert in range(NUM_EXPERTS)}
        for layer in range(mapping.shape[0])
    }


def rank_load(counts_layer: np.ndarray, mapping_layer: np.ndarray) -> np.ndarray:
    loads = np.zeros((counts_layer.shape[0], EP_DEGREE), dtype=np.float64)
    for rank in range(EP_DEGREE):
        experts = np.flatnonzero(mapping_layer == rank)
        loads[:, rank] = counts_layer[:, experts].sum(axis=1)
    return loads


def imbalance_from_rank_load(loads: np.ndarray) -> np.ndarray:
    mean = loads.mean(axis=1)
    return np.divide(loads.max(axis=1), mean, out=np.zeros_like(mean), where=mean > 0)


def objective_from_rank_load(
    loads: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    imbalance = imbalance_from_rank_load(loads)
    total = loads.sum(axis=0)
    total_imbalance = total.max() / total.mean() if total.mean() > 0 else 0.0
    return (
        float(imbalance.mean())
        + alpha * float(np.quantile(imbalance, 0.95))
        + beta * float(imbalance.max())
        + gamma * float(total_imbalance)
    )


def init_lpt_by_mean(counts_layer: np.ndarray) -> np.ndarray:
    weights = counts_layer.mean(axis=0)
    order = sorted(range(NUM_EXPERTS), key=lambda expert: (-float(weights[expert]), expert))
    loads = np.zeros(EP_DEGREE, dtype=np.float64)
    counts = np.zeros(EP_DEGREE, dtype=np.int64)
    mapping = np.full(NUM_EXPERTS, -1, dtype=np.int64)
    for expert in order:
        candidates = [rank for rank in range(EP_DEGREE) if counts[rank] < EXPERTS_PER_RANK]
        rank = min(candidates, key=lambda value: (loads[value], counts[value], value))
        mapping[expert] = rank
        counts[rank] += 1
        loads[rank] += weights[expert]
    return mapping


def optimize_layer(
    counts_layer: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    max_passes: int,
    candidate_pool: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mapping = init_lpt_by_mean(counts_layer)
    loads = rank_load(counts_layer, mapping)
    best = objective_from_rank_load(loads, alpha, beta, gamma)
    initial = best
    accepted = 0
    scores = counts_layer.mean(axis=0) + counts_layer.std(axis=0)
    order = sorted(range(NUM_EXPERTS), key=lambda expert: (-float(scores[expert]), expert))
    if candidate_pool > 0:
        order = order[: min(NUM_EXPERTS, candidate_pool)]

    # Pair-swap local search. Swaps preserve exactly 16 experts per rank.
    for _pass in range(max_passes):
        improved = False
        for pos, expert_a in enumerate(order):
            rank_a = int(mapping[expert_a])
            for expert_b in order[pos + 1 :]:
                rank_b = int(mapping[expert_b])
                if rank_a == rank_b:
                    continue
                candidate_loads = loads.copy()
                delta = counts_layer[:, expert_a] - counts_layer[:, expert_b]
                candidate_loads[:, rank_a] -= delta
                candidate_loads[:, rank_b] += delta
                candidate = objective_from_rank_load(candidate_loads, alpha, beta, gamma)
                if candidate + 1e-12 < best:
                    mapping[expert_a], mapping[expert_b] = rank_b, rank_a
                    loads = candidate_loads
                    best = candidate
                    accepted += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    final_imbalance = imbalance_from_rank_load(loads)
    return mapping, {
        "initial_objective": initial,
        "final_objective": best,
        "accepted_swaps": accepted,
        "candidate_pool": len(order),
        "mean_max_over_mean": float(final_imbalance.mean()),
        "p95_max_over_mean": float(np.quantile(final_imbalance, 0.95)),
        "max_max_over_mean": float(final_imbalance.max()),
        "std_max_over_mean": float(final_imbalance.std()),
        "rank_total_max_over_mean": float(loads.sum(axis=0).max() / loads.sum(axis=0).mean()),
    }


def summarize_mapping(counts: np.ndarray, mapping: np.ndarray) -> dict[str, Any]:
    per_layer = []
    all_imbalance = []
    total_rank = np.zeros(EP_DEGREE, dtype=np.float64)
    for layer in range(counts.shape[1]):
        loads = rank_load(counts[:, layer, :], mapping[layer])
        imbalance = imbalance_from_rank_load(loads)
        all_imbalance.extend(imbalance.tolist())
        total_rank += loads.sum(axis=0)
        per_layer.append(
            {
                "layer": layer,
                "mean_max_over_mean": float(imbalance.mean()),
                "p95_max_over_mean": float(np.quantile(imbalance, 0.95)),
                "max_max_over_mean": float(imbalance.max()),
                "std_max_over_mean": float(imbalance.std()),
            }
        )
    all_arr = np.asarray(all_imbalance, dtype=np.float64)
    return {
        "batch_layer": {
            "mean_max_over_mean": float(all_arr.mean()),
            "std_max_over_mean": float(all_arr.std()),
            "p90_max_over_mean": float(np.quantile(all_arr, 0.90)),
            "p95_max_over_mean": float(np.quantile(all_arr, 0.95)),
            "max_max_over_mean": float(all_arr.max()),
        },
        "rank_total": {
            "loads": total_rank,
            "max_over_mean": float(total_rank.max() / total_rank.mean()),
            "min_over_max": float(total_rank.min() / total_rank.max()),
        },
        "per_layer": per_layer,
    }


def plot_layer_imbalance(
    linear_summary: dict[str, Any],
    optimized_summary: dict[str, Any],
    output: Path,
) -> None:
    layers = np.arange(NUM_LAYERS)
    linear = [row["mean_max_over_mean"] for row in linear_summary["per_layer"]]
    optimized = [row["mean_max_over_mean"] for row in optimized_summary["per_layer"]]
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(layers, linear, marker="o", markersize=3, label="As-Is linear")
    ax.plot(layers, optimized, marker="o", markersize=3, label="Tail-aware To-Be")
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1)
    ax.set_title("Offline Calibration Layer-Wise Rank Imbalance")
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("mean batch max rank load / mean rank load")
    ax.grid(alpha=0.25)
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_distribution(
    linear_summary: dict[str, Any],
    optimized_summary: dict[str, Any],
    output: Path,
) -> None:
    linear = [
        row["mean_max_over_mean"]
        for row in linear_summary["per_layer"]
    ]
    optimized = [
        row["mean_max_over_mean"]
        for row in optimized_summary["per_layer"]
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.boxplot([linear, optimized], tick_labels=["As-Is linear", "Tail-aware To-Be"], showfliers=True)
    ax.set_title("Offline Layer Imbalance Distribution")
    ax.set_ylabel("mean batch max / mean")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-counts", required=True)
    parser.add_argument("--output-dir", default="outputs/placement_tail")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--max-passes", type=int, default=250)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=96,
        help="Limit pair-swap search to the highest mean+std experts per layer; <=0 searches all experts.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw = np.load(args.batch_counts)
    counts = raw["batch_expert_loads"].astype(np.float64)
    if counts.shape[1:] != (NUM_LAYERS, NUM_EXPERTS):
        raise ValueError(f"unexpected counts shape: {counts.shape}")

    linear = linear_mapping()
    optimized = np.empty_like(linear)
    layer_stats = []
    for layer in range(NUM_LAYERS):
        optimized[layer], stats = optimize_layer(
            counts[:, layer, :],
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            max_passes=args.max_passes,
            candidate_pool=args.candidate_pool,
        )
        stats["layer"] = layer
        layer_stats.append(stats)
        print(
            "layer "
            f"{layer:02d}: objective {stats['initial_objective']:.6f}"
            f" -> {stats['final_objective']:.6f},"
            f" swaps={stats['accepted_swaps']},"
            f" mean={stats['mean_max_over_mean']:.6f},"
            f" p95={stats['p95_max_over_mean']:.6f},"
            f" max={stats['max_max_over_mean']:.6f}",
            flush=True,
        )

    linear_summary = summarize_mapping(counts, linear)
    optimized_summary = summarize_mapping(counts, optimized)
    _write_json(output_dir / "linear_map_perlayer.json", mapping_to_json(linear))
    _write_json(output_dir / "tail_optimized_map_perlayer.json", mapping_to_json(optimized))
    _write_json(
        output_dir / "tail_optimization_summary.json",
        {
            "batch_counts": args.batch_counts,
            "shape": list(counts.shape),
            "objective": {
                "mean": 1.0,
                "p95_alpha": args.alpha,
                "max_beta": args.beta,
                "rank_total_gamma": args.gamma,
                "max_passes": args.max_passes,
                "candidate_pool": args.candidate_pool,
            },
            "linear": linear_summary,
            "tail_optimized": optimized_summary,
            "layer_optimization": layer_stats,
        },
    )
    plot_layer_imbalance(
        linear_summary,
        optimized_summary,
        output_dir / "offline_layer_imbalance.png",
    )
    plot_distribution(
        linear_summary,
        optimized_summary,
        output_dir / "offline_layer_imbalance_box.png",
    )
    print(
        json.dumps(
            {
                "linear": linear_summary["batch_layer"],
                "tail_optimized": optimized_summary["batch_layer"],
                "outputs": {
                    "linear_map": str(output_dir / "linear_map_perlayer.json"),
                    "tail_map": str(output_dir / "tail_optimized_map_perlayer.json"),
                    "summary": str(output_dir / "tail_optimization_summary.json"),
                },
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
