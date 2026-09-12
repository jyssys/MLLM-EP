#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASES = ["early", "middle", "late"]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def median(values):
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else float("nan")


def write_csv(path: Path, rows: list[dict]):
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ordered_by_layer(rows):
    grouped = defaultdict(list)
    for row in rows:
        if str(row.get("request_id", "")).startswith("measured"):
            grouped[int(row["layer"])].append(row)
    return grouped


def main():
    aggregate_stability = []
    aggregate_latency = []
    for dataset_dir in sorted((ROOT / "results/raw/stability").glob("*/r*")):
        dataset = dataset_dir.parent.name
        repeat = dataset_dir.name
        stability = [
            row
            for row in read_jsonl(dataset_dir / "layer_stability_rank0.jsonl")
            if str(row.get("request_id", "")).startswith("measured")
            and row.get("phase") in PHASES
        ]
        stability_by_layer = ordered_by_layer(stability)
        aligned = {}
        for layer, rows in stability_by_layer.items():
            for order, row in enumerate(rows):
                aligned[(layer, order)] = row

        block_rank = {}
        ep_rank = {}
        for rank in range(4):
            for kind, destination in (("block", block_rank), ("ep", ep_rank)):
                grouped = ordered_by_layer(
                    read_jsonl(dataset_dir / f"{kind}_rank{rank}.jsonl")
                )
                for layer, rows in grouped.items():
                    for order, row in enumerate(rows):
                        destination[(layer, order, rank)] = row

        shape_rank = {}
        for rank in range(4):
            grouped = ordered_by_layer(
                read_jsonl(dataset_dir / f"shape_rank{rank}.jsonl")
            )
            for layer, rows in grouped.items():
                for order, row in enumerate(rows):
                    shape_rank[(layer, order, rank)] = row

        stability_metrics = [
            "input_hidden_norm",
            "output_hidden_norm",
            "residual_update_norm",
            "relative_l2_update",
            "output_cosine",
            "attention_update_norm",
            "moe_update_norm",
            "moe_to_total_update_norm_ratio",
            "routed_output_norm",
            "shared_output_norm",
            "moe_output_norm",
        ]
        latency_metrics = [
            "prepare_attn_ms",
            "attention_ms",
            "prepare_mlp_ms",
            "mlp_ms",
            "postprocess_ms",
            "router_ms",
            "dispatch_ms",
            "expert_ms",
            "combine_ms",
            "shared_ms",
            "gather_ms",
        ]

        cells = defaultdict(list)
        for key, row in aligned.items():
            cells[(int(row["layer"]), row["phase"])].append((key, row))
        for (layer, phase), entries in sorted(cells.items()):
            out = {
                "dataset": dataset,
                "repeat": repeat,
                "layer": layer,
                "phase": phase,
                "waves": len(entries),
                "live_ratio_median": median(row["live_ratio"] for _, row in entries),
            }
            for metric in stability_metrics:
                out[f"{metric}_median"] = median(row.get(metric) for _, row in entries)
            aggregate_stability.append(out)

            latency = {
                "dataset": dataset,
                "repeat": repeat,
                "layer": layer,
                "phase": phase,
                "waves": len(entries),
            }
            for metric in latency_metrics:
                critical = []
                for (entry_layer, order), _ in entries:
                    source = block_rank if metric in {
                        "prepare_attn_ms", "attention_ms", "prepare_mlp_ms", "mlp_ms", "postprocess_ms"
                    } else ep_rank
                    values = [
                        source[(entry_layer, order, rank)].get(metric)
                        for rank in range(4)
                        if (entry_layer, order, rank) in source
                    ]
                    if values:
                        critical.append(max(float(value) for value in values))
                latency[f"{metric}_critical_median_ms"] = median(critical)
                latency[f"{metric}_critical_sum_ms"] = sum(critical)
            router_entropy = []
            topk_mass = []
            for (entry_layer, order), _ in entries:
                for rank in range(4):
                    shape = shape_rank.get((entry_layer, order, rank))
                    if shape:
                        router_entropy.append(shape.get("router_entropy_mean"))
                        topk_mass.append(shape.get("topk_probability_mass_mean"))
            latency["router_entropy_mean"] = median(router_entropy)
            latency["topk_mass_mean"] = median(topk_mass)
            aggregate_latency.append(latency)

    if not aggregate_stability:
        raise SystemExit("no stability traces found")
    write_csv(ROOT / "LAYER_PHASE_STABILITY.csv", aggregate_stability)
    write_csv(ROOT / "LAYER_PHASE_LATENCY.csv", aggregate_latency)

    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)

    def heatmap(rows, field, title, filename, cmap="viridis"):
        datasets = sorted({row["dataset"] for row in rows})
        figure, axes = plt.subplots(1, len(datasets), figsize=(5.8 * len(datasets), 7), squeeze=False)
        for axis, dataset in zip(axes[0], datasets):
            matrix = np.full((32, 3), np.nan)
            for row in rows:
                if row["dataset"] == dataset:
                    matrix[int(row["layer"]), PHASES.index(row["phase"])] = float(row[field])
            image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
            axis.set_title(dataset)
            axis.set_xticks(range(3), PHASES)
            axis.set_xlabel("refinement phase")
            axis.set_ylabel("layer")
            figure.colorbar(image, ax=axis, shrink=0.75)
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=180)
        plt.close(figure)

    heatmap(
        aggregate_stability,
        "relative_l2_update_median",
        "Layer output relative-L2 update",
        "01_layer_phase_hidden_update.png",
    )
    heatmap(
        aggregate_stability,
        "moe_update_norm_median",
        "MoE update norm",
        "02_layer_phase_moe_update.png",
    )
    heatmap(
        aggregate_stability,
        "attention_update_norm_median",
        "Attention update norm",
        "03_layer_phase_attention_update.png",
    )
    heatmap(
        aggregate_stability,
        "output_cosine_median",
        "Layer input/output cosine",
        "04_layer_phase_output_cosine.png",
        cmap="magma",
    )
    heatmap(
        aggregate_latency,
        "mlp_ms_critical_median_ms",
        "Observer-heavy critical-rank MLP latency (ms)",
        "05_layer_phase_mlp_latency.png",
    )
    heatmap(
        aggregate_latency,
        "attention_ms_critical_median_ms",
        "Observer-heavy critical-rank attention latency (ms)",
        "06_layer_phase_attention_latency.png",
    )
    print(json.dumps({"stability_cells": len(aggregate_stability), "latency_cells": len(aggregate_latency)}))


if __name__ == "__main__":
    main()
