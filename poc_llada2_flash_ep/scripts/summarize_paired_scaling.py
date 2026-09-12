#!/usr/bin/env python3
"""Aggregate independent paired TP4/EP4 restarts and render scaling figures."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("figure_dir", type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open()))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    paired: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["mode"] != "dynamic4":
            continue
        grouped[(row["topology"], int(row["batch"]))].append(row)
        paired[(int(row["batch"]), int(row["repeat"]))][row["topology"]] = row

    output_rows = []
    for batch in sorted({key[1] for key in grouped}):
        for topology in ("tp4", "ep4"):
            values = grouped.get((topology, batch), [])
            if not values:
                continue
            wall = np.asarray([float(row["wall_seconds"]) for row in values])
            ms_forward = np.asarray([float(row["mean_ms_per_forward"]) for row in values])
            tps = np.asarray([float(row["generated_tokens_per_second"]) for row in values])
            output_rows.append(
                {
                    "record_type": "aggregate",
                    "batch": batch,
                    "repeat": "",
                    "topology": topology,
                    "n": len(values),
                    "wall_seconds": float(np.median(wall)),
                    "mean_ms_per_forward": float(np.median(ms_forward)),
                    "generated_tokens_per_second": float(np.median(tps)),
                    "ep_wall_gain_percent": "",
                    "ep_forward_gain_percent": "",
                }
            )
        gains_wall, gains_forward = [], []
        for (candidate_batch, repeat), pair in paired.items():
            if candidate_batch != batch or not {"tp4", "ep4"} <= pair.keys():
                continue
            tp, ep = pair["tp4"], pair["ep4"]
            tp_wall, ep_wall = float(tp["wall_seconds"]), float(ep["wall_seconds"])
            tp_forward, ep_forward = float(tp["mean_ms_per_forward"]), float(ep["mean_ms_per_forward"])
            gain_wall = 100 * (tp_wall - ep_wall) / tp_wall
            gain_forward = 100 * (tp_forward - ep_forward) / tp_forward
            gains_wall.append(gain_wall)
            gains_forward.append(gain_forward)
            output_rows.append(
                {
                    "record_type": "pair",
                    "batch": batch,
                    "repeat": repeat,
                    "topology": "ep4_vs_tp4",
                    "n": 1,
                    "wall_seconds": "",
                    "mean_ms_per_forward": "",
                    "generated_tokens_per_second": "",
                    "ep_wall_gain_percent": gain_wall,
                    "ep_forward_gain_percent": gain_forward,
                }
            )
        if gains_wall:
            output_rows.append(
                {
                    "record_type": "paired_median",
                    "batch": batch,
                    "repeat": "",
                    "topology": "ep4_vs_tp4",
                    "n": len(gains_wall),
                    "wall_seconds": "",
                    "mean_ms_per_forward": "",
                    "generated_tokens_per_second": "",
                    "ep_wall_gain_percent": float(np.median(gains_wall)),
                    "ep_forward_gain_percent": float(np.median(gains_forward)),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    batches = sorted({key[1] for key in grouped})
    for metric, ylabel, filename in (
        ("wall_seconds", "32-request batch completion time (s)", "batch_scaling_wall.png"),
        ("generated_tokens_per_second", "Generated tokens / s", "batch_scaling_throughput.png"),
        ("mean_ms_per_forward", "Mean wall ms / model forward", "batch_scaling_forward.png"),
    ):
        plt.figure(figsize=(6.4, 4.0))
        for topology in ("tp4", "ep4"):
            medians = [
                np.median([float(row[metric]) for row in grouped[(topology, batch)]])
                if grouped.get((topology, batch)) else np.nan
                for batch in batches
            ]
            plt.plot(batches, medians, marker="o", label=topology.upper())
        plt.xscale("log", base=2)
        plt.xticks(batches, batches)
        plt.xlabel("Submitted request batch")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.figure_dir / filename, dpi=180)
        plt.close()


if __name__ == "__main__":
    main()
