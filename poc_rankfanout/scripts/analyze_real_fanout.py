#!/usr/bin/env python3
"""Aggregate directly hooked Qwen3-VL top-k routes and draw B1-B4."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in args.run:
        for path in run.glob("route_rows_dp*.jsonl"):
            with path.open() as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise SystemExit("no route rows found")

    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "real_route_invocations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["workload_id"], int(row["layer_id"]))].append(row)
    profile = []
    for (workload, layer), values in sorted(grouped.items()):
        profile.append({
            "workload_id": workload,
            "layer_id": layer,
            "n": len(values),
            "mean_fanout": float(np.mean([row["mean_fanout"] for row in values])),
            "p90_fanout": float(np.mean([row["p90_fanout"] for row in values])),
            "frac_full_fanout": float(np.mean([row["frac_full_fanout"] for row in values])),
            "rank_max_mean": float(np.mean([row["rank_load_max_over_mean"] for row in values])),
        })
    with (args.output_dir / "real_layer_fanout.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(profile)

    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for workload in sorted({row["workload_id"] for row in profile}):
        selected = [row for row in profile if row["workload_id"] == workload]
        ax.plot([row["layer_id"] for row in selected], [row["mean_fanout"] for row in selected], label=workload)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("mean destination-rank fanout")
    ax.set_ylim(1, 4.05)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "B1_real_layer_fanout.png", dpi=180)
    plt.close(fig)

    invocations = sorted(rows, key=lambda row: (row["workload_id"], row["request_id"], row["step_id"], row["layer_id"]))
    by_invocation: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    for row in invocations:
        by_invocation[(row["workload_id"], row["request_id"], int(row["step_id"]))][int(row["layer_id"])] = float(row["mean_fanout"])
    keys = sorted(by_invocation)
    heat = np.full((len(keys), 48), np.nan)
    for index, key in enumerate(keys):
        for layer, value in by_invocation[key].items():
            heat[index, layer] = value
    fig, ax = plt.subplots(figsize=(10, max(3, len(keys) * .25)))
    image = ax.imshow(heat, vmin=1, vmax=4, aspect="auto", cmap="viridis")
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("request/iteration")
    fig.colorbar(image, ax=ax, label="mean fanout")
    fig.tight_layout()
    fig.savefig(figures / "B2_real_fanout_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([row["mean_fanout"] for row in rows], bins=30)
    ax.set_xlabel("invocation mean fanout")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(figures / "B3_real_fanout_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([row["mean_fanout"] for row in rows], [row["rank_load_max_over_mean"] for row in rows], alpha=.35)
    ax.set_xlabel("mean fanout")
    ax.set_ylabel("rank max/mean")
    fig.tight_layout()
    fig.savefig(figures / "B4_fanout_vs_rank_imbalance.png", dpi=180)
    plt.close(fig)

    values = np.asarray([row["mean_fanout"] for row in rows], dtype=float)
    token_weights = np.asarray([row["num_tokens"] for row in rows], dtype=float)
    weighted_fractions = {
        f"token_weighted_frac_f{fanout}": float(np.average(
            [row[f"frac_f{fanout}"] for row in rows], weights=token_weights
        ))
        for fanout in range(1, 5)
    }
    workload_summary = []
    for workload in sorted({row["workload_id"] for row in rows}):
        selected = [row for row in rows if row["workload_id"] == workload]
        selected_values = np.asarray([row["mean_fanout"] for row in selected], dtype=float)
        selected_weights = np.asarray([row["num_tokens"] for row in selected], dtype=float)
        workload_summary.append({
            "workload_id": workload,
            "rows": len(selected),
            "mean_fanout": float(np.average(selected_values, weights=selected_weights)),
            "min_invocation_mean_fanout": float(selected_values.min()),
            "max_invocation_mean_fanout": float(selected_values.max()),
            "token_weighted_frac_f4": float(np.average(
                [row["frac_f4"] for row in selected], weights=selected_weights
            )),
        })
    output = {
        "rows": len(rows),
        "workloads": sorted({row["workload_id"] for row in rows}),
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, .1)),
        "p50": float(np.median(values)),
        "p90": float(np.quantile(values, .9)),
        "min": float(values.min()),
        "max": float(values.max()),
        "fraction_mean_ge_3_5": float(np.mean(values >= 3.5)),
        "fraction_full_narrow_regime": float(max(np.mean(values >= 3.75), np.mean(values < 2.5))),
        **weighted_fractions,
        "workload_summary": workload_summary,
    }
    (args.output_dir / "real_fanout_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
