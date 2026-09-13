#!/usr/bin/env python3
"""Build PP ceiling, temporal-boundary, and quality-conditioned Pareto tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TIMING_FIELDS = ("prepare_attn_ms", "attention_ms", "prepare_mlp_ms", "mlp_ms")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def log_seconds(path: Path) -> float:
    match = re.findall(r"Forward:\s*\d+, Time:\s*([0-9.]+)", path.read_text())
    if not match:
        raise RuntimeError(f"missing Forward line: {path}")
    return float(match[-1])


def flowshop(stage_times: np.ndarray) -> float:
    completion = np.zeros_like(stage_times)
    for wave in range(stage_times.shape[0]):
        for stage in range(stage_times.shape[1]):
            completion[wave, stage] = stage_times[wave, stage] + max(
                completion[wave - 1, stage] if wave else 0.0,
                completion[wave, stage - 1] if stage else 0.0,
            )
    return float(completion[-1, -1])


def stage_matrix(root: Path, dataset: str) -> tuple[list[int], np.ndarray]:
    directory = root / f"trackB/boundary_trace/{dataset}/r1"
    ranks = []
    for rank in range(4):
        rows = [
            row
            for row in jsonl(directory / f"block_rank{rank}.jsonl")
            if row["request_id"].startswith("measured") and row["wave"] >= 0
        ]
        ranks.append({(row["wave"], row["layer"]): row for row in rows})
    waves = sorted({key[0] for key in ranks[0]})
    matrix = []
    for wave in waves:
        layer_winners = []
        for layer in range(32):
            # Exclude postprocess timing: boundary tracing executes inside that
            # interval. The omitted production postprocess is sub-0.01 ms.
            layer_winners.append(
                max(
                    sum(float(ranks[rank][wave, layer][field]) for field in TIMING_FIELDS)
                    for rank in range(4)
                )
            )
        matrix.append(layer_winners)
    return waves, np.asarray(matrix)


def split_stages(layer_times: np.ndarray, boundaries: list[int]) -> np.ndarray:
    starts = [0] + boundaries
    ends = boundaries + [32]
    return np.column_stack(
        [layer_times[:, start:end].sum(axis=1) for start, end in zip(starts, ends)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    boundary_rows = []
    ceiling_rows = []
    stage_cache = {}
    for dataset in ("gsm8k", "humaneval"):
        directory = root / f"trackB/boundary_trace/{dataset}/r1"
        boundaries = jsonl(directory / "pp_boundary_rank0.jsonl")
        for boundary in (8, 16, 24, 32):
            for phase in ("all", "early", "middle", "late"):
                selected = [
                    row for row in boundaries
                    if row["boundary"] == boundary
                    and row["matched_rows"]
                    and (phase == "all" or row["phase"] == phase)
                ]
                if not selected:
                    continue
                boundary_rows.append(
                    {
                        "dataset": dataset,
                        "boundary": boundary,
                        "phase": phase,
                        "records": len(selected),
                        "matched_row_fraction": sum(row["matched_rows"] for row in selected) / sum(row["rows"] for row in selected),
                        "cosine_p50_median": float(np.median([row["cosine_p50"] for row in selected])),
                        "rel_l2_p50_median": float(np.median([row["rel_l2_p50"] for row in selected])),
                        "rel_l2_p90_median": float(np.median([row["rel_l2_p90"] for row in selected])),
                        "live_rel_l2_mean_median": float(np.median([row["live_rel_l2_mean"] for row in selected if row["live_rel_l2_mean"] is not None])),
                    }
                )
        waves, layers = stage_matrix(root, dataset)
        stage_cache[dataset] = layers
        trace_wall_ms = 1000.0 * log_seconds(
            root / f"logs/trackB_boundary_{dataset}_ep4_b32_mini32_r1_g32.log"
        )
        stage_proxy_ms = float(layers.sum())
        backbone_fraction = min(stage_proxy_ms / trace_wall_ms, 0.95)
        for name, split in (("PP2_boundary16", [16]), ("PP4_equal8", [8, 16, 24])):
            stages = split_stages(layers, split)
            sequential = float(stages.sum())
            pipeline = flowshop(stages)
            ceiling_rows.append(
                {
                    "dataset": dataset,
                    "topology": name,
                    "waves": len(waves),
                    "stage_proxy_sequential_ms": sequential,
                    "ideal_pipeline_makespan_ms": pipeline,
                    "ideal_backbone_speedup": sequential / pipeline,
                    "observer_backbone_fraction": backbone_fraction,
                    "amdahl_request_speedup_ceiling": 1.0 / ((1.0 - backbone_fraction) + backbone_fraction * pipeline / sequential),
                    "request_latency_reduction_ceiling_pct": 100.0 * backbone_fraction * (1.0 - pipeline / sequential),
                    "evidence_boundary": "measured TP4+EP4 per-layer replay proxy; timestep dependencies hypothetically removed; not live PP",
                }
            )
    write_csv(analysis / "trackB_boundary_similarity.csv", boundary_rows)
    write_csv(analysis / "trackB_ideal_pipeline_ceilings.csv", ceiling_rows)

    pareto = []
    for dataset in ("gsm8k", "humaneval"):
        quality_path = analysis / f"trackB_{dataset}_policy_quality.csv"
        if not quality_path.exists():
            continue
        candidate_runs = sorted((root / f"trackB/policy_sweep/{dataset}").glob("r*"))
        if not candidate_runs:
            continue
        policy_dir = max(
            candidate_runs,
            key=lambda path: len(list(path.glob("policy_*.json"))),
        )
        scope_path = policy_dir / "async_policy_scope_rank0.jsonl"
        quality = list(csv.DictReader(quality_path.open()))
        scope = jsonl(scope_path) if scope_path.exists() else []
        scope_by_policy = defaultdict(list)
        for row in scope:
            scope_by_policy[row["policy"]].append(row)
        baseline = next(row for row in quality if row["policy"] == "baseline")
        baseline_nfe = float(baseline["nfe"])
        trace_wall_ms = 1000.0 * log_seconds(
            root / f"logs/trackB_boundary_{dataset}_ep4_b32_mini32_r1_g32.log"
        )
        layers = stage_cache[dataset]
        backbone_fraction = min(float(layers.sum()) / trace_wall_ms, 0.95)
        for row in quality:
            name = row["policy"]
            payload = json.loads((policy_dir / f"policy_{name}.json").read_text())
            policy = payload["policy"]
            selected_layers = [int(item) for item in str(policy.get("async_boundaries", "")).split(",") if item]
            if not selected_layers:
                predicted_ratio = 1.0
                ideal_backbone_speedup = 1.0
                stale_fraction = 0.0
            else:
                boundaries = [layer + 1 for layer in selected_layers]
                stages = split_stages(layers, boundaries)
                ideal_backbone_speedup = float(stages.sum()) / flowshop(stages)
                records = scope_by_policy.get(name, [])
                stale_fraction = (
                    sum(item["stale_rows"] for item in records) / sum(item["rows"] for item in records)
                    if records else 0.0
                )
                nfe_ratio = float(row["nfe"]) / baseline_nfe
                backbone_ratio = nfe_ratio * (
                    (1.0 - stale_fraction) + stale_fraction / ideal_backbone_speedup
                )
                predicted_ratio = (1.0 - backbone_fraction) + backbone_fraction * backbone_ratio
            pareto.append(
                {
                    **row,
                    "stale_row_fraction": stale_fraction,
                    "ideal_backbone_speedup_for_cut": ideal_backbone_speedup,
                    "predicted_request_speedup_upper": 1.0 / predicted_ratio,
                    "predicted_request_latency_reduction_pct": 100.0 * (1.0 - predicted_ratio),
                    "evidence_boundary": "quality measured on sequential stale-boundary emulator; speed is stage-trace analytical upper, not live PP",
                }
            )
    if pareto:
        write_csv(analysis / "trackB_speed_quality_pareto.csv", pareto)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for dataset in ("gsm8k", "humaneval"):
        selected = [row for row in boundary_rows if row["dataset"] == dataset and row["phase"] == "all"]
        axes[0].plot([row["boundary"] for row in selected], [row["rel_l2_p50_median"] for row in selected], marker="o", label=dataset)
    axes[0].set_xlabel("pipeline boundary after layer")
    axes[0].set_ylabel("lag-1 activation rel-L2 (median)")
    axes[0].legend()
    if pareto:
        for dataset in ("gsm8k", "humaneval"):
            selected = [row for row in pareto if row["task"] == dataset]
            axes[1].scatter([float(row["relative_quality_pct"]) for row in selected], [float(row["predicted_request_speedup_upper"]) for row in selected], label=dataset, alpha=0.75)
        axes[1].axvline(99, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("relative bounded benchmark quality (%)")
        axes[1].set_ylabel("analytical request speedup upper")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(analysis / "trackB_boundary_and_pareto.png", dpi=180)


if __name__ == "__main__":
    main()
