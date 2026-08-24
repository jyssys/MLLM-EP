#!/usr/bin/env python3
"""Analyze the QuACK 0.6.4 closure without post-hoc shape selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    previous = json.loads(args.previous.read_text())
    records = data["records"]

    def rows(name: str) -> list[dict]:
        return [x for x in records if x["name"] == name]

    aligned = rows("aligned")
    heavy = rows("boundary_heavy")
    aligned_ms = float(np.median([x["median_ms"] for x in aligned]))
    heavy_ms = float(np.median([x["median_ms"] for x in heavy]))
    aligned_heavy_round_pct = [
        (h["median_ms"] / a["median_ms"] - 1) * 100
        for a, h in zip(sorted(aligned, key=lambda x: x["round"]),
                        sorted(heavy, key=lambda x: x["round"]))
    ]

    boundary = []
    for multiple in (1, 2, 3):
        round_penalties = []
        for round_id in range(data["independent_rounds"]):
            center = next(x for x in rows(f"{multiple}B_{0:+d}") if x["round"] == round_id)
            neighbors = [
                next(x for x in rows(f"{multiple}B_{delta:+d}") if x["round"] == round_id)
                for delta in (-1, 1)
            ]
            penalty = (
                np.median([x["median_ms"] for x in neighbors])
                / center["median_ms"] - 1
            ) * 100
            round_penalties.append(float(penalty))
        boundary.append(
            {
                "multiple": multiple,
                "round_penalty_pct": round_penalties,
                "median_penalty_pct": float(np.median(round_penalties)),
                "minimum_round_penalty_pct": float(min(round_penalties)),
                "sign_consistent": all(x > 0 for x in round_penalties),
            }
        )

    reproducible = [
        x["minimum_round_penalty_pct"] for x in boundary if x["sign_consistent"]
    ]
    largest = max(reproducible, default=0.0)
    largest_observed = max(
        abs(value) for item in boundary for value in item["round_penalty_pct"]
    )
    status = "GO" if largest >= 5 else ("HOLD" if largest >= 2 else "NO-GO")
    old = next(x for x in previous["shapes"] if x["name"] == "qwen3_30b_a3b")
    summary = {
        "QUACK064_BOUNDARY_STATUS": status,
        "REOPEN_RAGGED_GEMM": "YES" if status == "GO" else "NO",
        "actual_BLOCK_M": data["runtime_config"]["cta_block_m"],
        "aligned_latency_ms": aligned_ms,
        "boundary_heavy_latency_ms": heavy_ms,
        "boundary_heavy_relative_pct": (heavy_ms / aligned_ms - 1) * 100,
        "boundary_heavy_round_pct": aligned_heavy_round_pct,
        "boundary_sweeps": boundary,
        "largest_reproducible_boundary_jump_pct": largest,
        "largest_observed_single_round_jump_pct": largest_observed,
        "quack05": {
            "aligned_latency_ms": old["aligned_ms"],
            "boundary_heavy_latency_ms": old["boundary_heavy_ms"],
            "relative_pct": old["aligned_boundary_gap_pct"],
            "boundary_step_ratios_pct": [100 * x for x in old["boundary_step_ratios"]],
            "status": old["poc1_status"],
        },
        "versions": data["environment"],
        "gate_rule": (
            "largest minimum penalty shared by all three rounds: "
            "GO >=5%, HOLD >=2%, otherwise NO-GO"
        ),
    }
    result_dir = args.input.parent
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(data["independent_rounds"])
    axes[0].plot(x, [r["median_ms"] for r in sorted(aligned, key=lambda z: z["round"])], marker="o", label="aligned")
    axes[0].plot(x, [r["median_ms"] for r in sorted(heavy, key=lambda z: z["round"])], marker="o", label="boundary-heavy")
    axes[0].set_xlabel("independent round"); axes[0].set_ylabel("up+down latency (ms)"); axes[0].legend()
    for item in boundary:
        multiple = item["multiple"]
        deltas = (-1, 0, 1)
        medians = [
            np.median([r["median_ms"] for r in rows(f"{multiple}B_{delta:+d}")])
            for delta in deltas
        ]
        axes[1].plot(deltas, medians, marker="o", label=f"{multiple}×BLOCK_M")
        axes[2].plot(x, item["round_penalty_pct"], marker="o", label=f"{multiple}×BLOCK_M")
    axes[1].set_xticks([-1, 0, 1]); axes[1].set_xlabel("paired-expert offset from boundary")
    axes[1].set_ylabel("median latency (ms)"); axes[1].legend()
    axes[2].axhline(0, color="black", lw=.7); axes[2].axhline(5, color="red", lw=.7, ls="--")
    axes[2].set_xlabel("independent round"); axes[2].set_ylabel("±1 vs boundary penalty (%)"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(result_dir / "plot_quack064_boundary_closure.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
