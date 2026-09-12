#!/usr/bin/env python3
"""Align prior low-overhead EP4 timings with the fresh phase trace.

The stability run records tensor norms and therefore perturbs the distributed
timeline. It is representation evidence, not cost evidence. The immediately
preceding discovery sprint used the identical model, request set and
best-static runtime with only CUDA-event stage tracing. This script aligns
those records by deterministic per-layer invocation order and maps each cell
to the three-restart clean request median.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIOR = Path("/home/esjung/MLLM-EP-discovery/poc_dllm_ep_discovery")
PHASES = ["early", "middle", "late"]
STAGES = ["router", "dispatch", "expert", "combine", "shared", "gather"]
BLOCK_STAGES = ["prepare_attn", "attention", "prepare_mlp", "mlp", "postprocess"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def by_layer(rows: list[dict], measured_prefix: str = "measured"):
    result = defaultdict(list)
    for row in rows:
        if str(row.get("request_id", "")).startswith(measured_prefix):
            result[int(row["layer"])].append(row)
    return result


def clean_medians() -> dict[str, float]:
    values = defaultdict(list)
    with (PRIOR / "BASELINE_CLEAN.csv").open() as handle:
        for row in csv.DictReader(handle):
            values[row["dataset"]].append(float(row["wall_seconds"]) * 1000.0)
    return {dataset: statistics.median(samples) for dataset, samples in values.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    clean = clean_medians()
    output = []
    for dataset in ("gsm8k", "humaneval"):
        prior_dir = PRIOR / f"results/raw/stage/{dataset}/r2"
        fresh_dir = ROOT / f"results/raw/stability/{dataset}/r1"

        phase_rows = by_layer(read_jsonl(fresh_dir / "layer_stability_rank0.jsonl"))
        phases = {
            layer: [row["phase"] for row in rows if row.get("phase") in PHASES]
            for layer, rows in phase_rows.items()
        }

        stage_by_rank = []
        block_by_rank = []
        for rank in range(4):
            stage_by_rank.append(by_layer(read_jsonl(prior_dir / f"ep_rank{rank}.jsonl")))
            block_by_rank.append(by_layer(read_jsonl(prior_dir / f"block_rank{rank}.jsonl")))

        for layer in range(32):
            layer_phases = phases[layer]
            # Prior records contain one measured prefill before refinement.
            block_rows = [rank_rows[layer][1:] for rank_rows in block_by_rank]
            if not all(len(rows) == len(layer_phases) for rows in block_rows):
                raise RuntimeError(
                    f"{dataset} layer {layer}: phase/block alignment mismatch "
                    f"{len(layer_phases)} vs {[len(rows) for rows in block_rows]}"
                )
            ep_rows = None
            if layer > 0:
                ep_rows = [rank_rows[layer][1:] for rank_rows in stage_by_rank]
                if not all(len(rows) == len(layer_phases) for rows in ep_rows):
                    raise RuntimeError(
                        f"{dataset} layer {layer}: phase/EP alignment mismatch"
                    )

            for phase in PHASES:
                indices = [index for index, value in enumerate(layer_phases) if value == phase]
                row = {
                    "dataset": dataset,
                    "layer": layer,
                    "phase": phase,
                    "waves": len(indices),
                    "clean_request_median_ms": clean[dataset],
                }
                for stage in BLOCK_STAGES:
                    critical = [
                        max(
                            float(block_rows[rank][index][f"{stage}_ms"])
                            for rank in range(4)
                        )
                        for index in indices
                    ]
                    row[f"{stage}_sum_ms"] = sum(critical)
                for stage in STAGES:
                    if ep_rows is None:
                        row[f"{stage}_sum_ms"] = 0.0
                    else:
                        critical = [
                            max(
                                float(ep_rows[rank][index][f"{stage}_ms"])
                                for rank in range(4)
                            )
                            for index in indices
                        ]
                        row[f"{stage}_sum_ms"] = sum(critical)
                row["whole_layer_sum_ms"] = sum(
                    row[f"{stage}_sum_ms"] for stage in BLOCK_STAGES
                )
                row["non_moe_layer_proxy_sum_ms"] = sum(
                    row[f"{stage}_sum_ms"]
                    for stage in (
                        "prepare_attn",
                        "attention",
                        "prepare_mlp",
                        "postprocess",
                    )
                )
                row["routed_moe_sum_ms"] = sum(
                    row[f"{stage}_sum_ms"]
                    for stage in ("router", "dispatch", "expert", "combine")
                )
                row["whole_moe_sum_ms"] = sum(
                    row[f"{stage}_sum_ms"] for stage in STAGES
                )
                for cost in ("whole_layer", "routed_moe", "whole_moe"):
                    row[f"{cost}_request_e2e_upper_percent"] = (
                        100.0 * row[f"{cost}_sum_ms"] / clean[dataset]
                    )
                output.append(row)

    # The block-level MLP bracket includes asynchronous stream waits and sums
    # to more than clean E2E, so it cannot be an additive request attribution.
    # Preserve it only as raw observer evidence. For a deliberately optimistic
    # full-layer oracle, allocate every clean-E2E millisecond not already
    # attributed to MoE across cells using the non-MoE layer-stage proxy. This
    # caps all full-layer work at exactly 100% of the request and favors, rather
    # than penalizes, a layer-skipping hypothesis.
    for dataset in ("gsm8k", "humaneval"):
        dataset_rows = [row for row in output if row["dataset"] == dataset]
        moe_percent = sum(
            row["whole_moe_request_e2e_upper_percent"] for row in dataset_rows
        )
        remaining_percent = max(0.0, 100.0 - moe_percent)
        proxy_total = sum(row["non_moe_layer_proxy_sum_ms"] for row in dataset_rows)
        for row in dataset_rows:
            row["non_moe_normalized_optimistic_e2e_percent"] = (
                remaining_percent * row["non_moe_layer_proxy_sum_ms"] / proxy_total
            )
            row["whole_layer_normalized_optimistic_e2e_percent"] = (
                row["whole_moe_request_e2e_upper_percent"]
                + row["non_moe_normalized_optimistic_e2e_percent"]
            )

    write_csv(ROOT / "LOW_OVERHEAD_LAYER_PHASE_COST.csv", output)

    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 7), squeeze=False)
    for axis, dataset in zip(axes[0], ("gsm8k", "humaneval")):
        matrix = np.full((32, 3), np.nan)
        for row in output:
            if row["dataset"] == dataset:
                matrix[int(row["layer"]), PHASES.index(row["phase"])] = row[
                    "whole_layer_normalized_optimistic_e2e_percent"
                ]
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(dataset)
        axis.set_xticks(range(3), PHASES)
        axis.set_xlabel("refinement phase")
        axis.set_ylabel("layer")
        figure.colorbar(
            image, ax=axis, shrink=0.75, label="clean request E2E upper share (%)"
        )
    figure.suptitle("Layer/phase low-overhead latency attribution")
    figure.tight_layout()
    figure.savefig(figures / "07_layer_phase_latency_map.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"rows": len(output), "clean_request_ms": clean}))


if __name__ == "__main__":
    main()
