#!/usr/bin/env python3
"""Build fixed-block summaries and figures from clean request-level runs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("accuracy"):
            continue
        # Sample-count changes both aggregate work and bounded quality.  Keep
        # it in the identity rather than accidentally pooling n=8 and n=32.
        # The no-EOS diagnostic is also a different decoder-effort control,
        # whereas labels such as ``matched`` and ``frontier`` merely describe
        # why an otherwise identical point was launched.
        decoder_control = {
            # These two campaigns were collected before discovering that
            # config 42 silently rewrote --threshold to 0.9.  Preserve them
            # for auditability but never pool them with the corrected runs.
            "effort": "legacy_threshold_overridden",
            "effortnoeos": "no_early_stop_threshold0.9",
            # Corrected: explicit threshold=1.0 plus EOS early-stop disabled.
            "matchedeffort": "matched_one_token_no_early_stop",
        }.get(row["regime"], "serving_native")
        key = (
            row["task"], int(row["n"]), int(row["block"]), int(row["mini"]),
            int(row["gen"]), float(row["threshold"]),
            int(row["target_total"] or 0), decoder_control,
        )
        grouped[key].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        task, sample_count, block, mini, generation, threshold, target_total, decoder_control = key
        times = np.asarray([float(row["bct_s"]) for row in values])
        qualities = np.asarray([float(row["accuracy"]) for row in values])
        nfes = np.asarray([float(row["nfe"]) for row in values])
        throughput = np.asarray([float(row["tps"]) for row in values])
        hbm = np.asarray([float(row["peak_hbm_mib"]) for row in values if row["peak_hbm_mib"]])
        output.append(
            {
                "task": task,
                "sample_count": sample_count,
                "block_length": block,
                "mini_batch_size": mini,
                "generation": generation,
                "threshold": threshold,
                "target_total_length": target_total,
                "decoder_control": decoder_control,
                "restarts": len(values),
                "bct_median_s": float(np.median(times)),
                "bct_min_s": float(np.min(times)),
                "bct_max_s": float(np.max(times)),
                "accuracy_median": float(np.median(qualities)),
                "accuracy_min": float(np.min(qualities)),
                "accuracy_max": float(np.max(qualities)),
                "nfe_median": float(np.median(nfes)),
                "throughput_median_tokens_per_s": float(np.median(throughput)),
                "peak_hbm_mib_median": float(np.median(hbm)) if len(hbm) else "",
                "source_regimes": ";".join(sorted({row["regime"] for row in values})),
            }
        )
    return output


def pareto(rows: list[dict]) -> list[dict]:
    output = []
    for task in sorted({row["task"] for row in rows}):
        candidates = [
            row for row in rows
            if row["task"] == task and row["sample_count"] == 32
            and row["target_total_length"] > 0
            and row["decoder_control"] == "serving_native"
        ]
        for row in candidates:
            dominated = any(
                float(other["bct_median_s"]) <= float(row["bct_median_s"])
                and float(other["accuracy_median"]) >= float(row["accuracy_median"])
                and (
                    float(other["bct_median_s"]) < float(row["bct_median_s"])
                    or float(other["accuracy_median"]) > float(row["accuracy_median"])
                )
                for other in candidates
            )
            output.append({**row, "pareto": not dominated})
    return output


def fixed_mini_plot(rows: list[dict], field: str, ylabel: str, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for axis, task in zip(axes, ("gsm8k", "humaneval")):
        subset = [
            row for row in rows
            if row["task"] == task and row["mini_batch_size"] == 8
            and row["sample_count"] == 32 and row["target_total_length"] > 0
            and row["decoder_control"] == "serving_native"
            and row["threshold"] == 0.9
        ]
        subset.sort(key=lambda row: row["block_length"])
        axis.plot([row["block_length"] for row in subset], [float(row[field]) for row in subset], marker="o")
        axis.set_title(task)
        axis.set_xlabel("block size B")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def mini_frontier_plot(rows: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    for axis, task in zip(axes, ("gsm8k", "humaneval")):
        subset = [
            row for row in rows if row["task"] == task
            and row["sample_count"] == 32 and row["target_total_length"] > 0
            and row["decoder_control"] == "serving_native"
            and row["threshold"] == 0.9
        ]
        for block in sorted({row["block_length"] for row in subset}):
            points = sorted((row for row in subset if row["block_length"] == block), key=lambda row: row["mini_batch_size"])
            axis.plot(
                [row["mini_batch_size"] for row in points],
                [float(row["bct_median_s"]) for row in points],
                marker="o", label=f"B{block}",
            )
        axis.set_title(task)
        axis.set_xlabel("mini_batch_size")
        axis.set_ylabel("clean BCT (s)")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "15_per_B_best_mini.png", dpi=180)
    plt.close(fig)


def pareto_plot(rows: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, task in zip(axes, ("gsm8k", "humaneval")):
        subset = [
            row for row in rows if row["task"] == task
            and row["sample_count"] == 32 and row["target_total_length"] > 0
            and row["decoder_control"] == "serving_native"
        ]
        for row in subset:
            axis.scatter(float(row["bct_median_s"]), 100 * float(row["accuracy_median"]), s=45 if row["pareto"] else 20)
            if row["pareto"]:
                axis.annotate(
                    f"B{row['block_length']}/m{row['mini_batch_size']}/t{row['threshold']}",
                    (float(row["bct_median_s"]), 100 * float(row["accuracy_median"])),
                    fontsize=7,
                )
        axis.set_title(task)
        axis.set_xlabel("clean BCT (s)")
        axis.set_ylabel("bounded score (%)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "16_fixed_B_quality_latency_pareto.png", dpi=180)
    plt.close(fig)


def main() -> None:
    raw = read_csv(ROOT / "STATIC_BLOCK_SWEEP.csv")
    quality_rows = [
        {
            "task": row["task"],
            "sample_count": row["n"],
            "regime": row["regime"],
            "decoder_control": (
                {
                    "effort": "legacy_threshold_overridden",
                    "effortnoeos": "no_early_stop_threshold0.9",
                    "matchedeffort": "matched_one_token_no_early_stop",
                }.get(row["regime"], "serving_native")
            ),
            "block_length": row["block"],
            "mini_batch_size": row["mini"],
            "generation": row["gen"],
            "threshold": row["threshold"],
            "threshold_requested": row.get("threshold_requested", row["threshold"]),
            "threshold_effective": row.get("threshold_effective", row["threshold"]),
            "target_total_length": row["target_total"],
            "repeat": row["repeat"],
            "quality_correct": row["quality_correct"],
            "quality_total": row["quality_total"],
            "accuracy": row["accuracy"],
            "nfe": row["nfe"],
            "bct_s": row["bct_s"],
            "run_dir": row["run_dir"],
        }
        for row in raw
    ]
    write_csv(ROOT / "BLOCK_QUALITY_METRICS.csv", quality_rows)
    summary = grouped_summary(raw)
    write_csv(ROOT / "STATIC_POINT_SUMMARY.csv", summary)
    frontier = pareto(summary)
    write_csv(ROOT / "QUALITY_LATENCY_FRONTIER.csv", frontier)
    per_b_best = []
    for task in ("gsm8k", "humaneval"):
        for block in sorted({row["block_length"] for row in summary if row["task"] == task}):
            candidates = [
                row for row in summary
                if row["task"] == task
                and row["sample_count"] == 32
                and row["block_length"] == block
                and row["target_total_length"] > 0
                and row["decoder_control"] == "serving_native"
                and row["threshold"] == 0.9
            ]
            if candidates:
                winner = min(candidates, key=lambda row: float(row["bct_median_s"]))
                per_b_best.append(winner)
    write_csv(ROOT / "PER_B_BEST_MINI.csv", per_b_best)
    fixed_mini_plot(summary, "accuracy_median", "bounded accuracy", "01_block_size_vs_quality.png")
    fixed_mini_plot(summary, "bct_median_s", "clean BCT (s)", "02_block_size_vs_BCT.png")
    fixed_mini_plot(
        summary,
        "throughput_median_tokens_per_s",
        "clean generated tokens / s",
        "03_block_size_vs_throughput.png",
    )
    fixed_mini_plot(summary, "nfe_median", "aggregate NFE", "04_block_size_vs_NFE.png")
    fixed_mini_plot(summary, "peak_hbm_mib_median", "peak sampled HBM (MiB)", "13_block_size_vs_HBM.png")
    mini_frontier_plot(summary)
    pareto_plot(frontier)
    print(f"wrote {len(summary)} static point summaries")


if __name__ == "__main__":
    main()
