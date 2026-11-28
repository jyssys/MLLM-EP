"""Render measured input geometry for the recovered, immutable Main400 family."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .datasets import read


COLORS = ("#123f74", "#bd5718", "#178975")
IMAGE_BINS = ((1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 100000))


def save(fig, directory, title):
    fig.savefig(directory / f"{title}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def counts(cohorts, out):
    fig, axis = plt.subplots(figsize=(8, 4.2), layout="constrained")
    x = np.arange(len(IMAGE_BINS))
    for n, (label, rows) in enumerate(cohorts.items()):
        fractions = [100 * sum(lo <= row["image_count"] <= hi for row in rows) / len(rows)
                     for lo, hi in IMAGE_BINS]
        axis.bar(x + ((n - 1) * .24), fractions, width=.23, label=label, color=COLORS[n])
    axis.set(xticks=x, xticklabels=("1", "2–4", "5–8", "9–16", "17–32", "33+"),
             xlabel="Images in an original request", ylabel="Share of requests (%)",
             title="Dataset image count distribution")
    axis.legend()
    axis.grid(axis="y", alpha=.2)
    save(fig, out, "dataset_image_count")


def cumulative(cohorts, out):
    fig, axis = plt.subplots(figsize=(7.4, 4.1), layout="constrained")
    for n, (label, rows) in enumerate(cohorts.items()):
        data = np.sort([row["visual_tokens"] for row in rows])
        axis.step(data, 100 * np.arange(1, len(data) + 1) / len(data), where="post",
                  label=label, color=COLORS[n], linewidth=1.8)
    axis.set(xlabel="Post-merger visual input tokens", ylabel="Cumulative share of requests (%)",
             title="Actual Qwen2.5-VL visual token distribution")
    axis.legend()
    axis.grid(alpha=.2)
    save(fig, out, "dataset_visual_tokens")


def fractions(cohorts, receipt, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    axis = axes[0]
    for n, (label, rows) in enumerate(cohorts.items()):
        data = 100 * np.sort([row["visual_ratio"] for row in rows])
        axis.step(data, 100 * np.arange(1, len(data) + 1) / len(data), where="post",
                  label=label, color=COLORS[n], linewidth=1.6)
    axis.set(xlabel="Visual share of original LM prompt (%)", ylabel="Cumulative share (%)",
             title="Natural request ratios")
    minimum = min(row["visual_ratio"] for rows in cohorts.values() for row in rows)
    axis.set_xlim(max(0, 100 * minimum - 3), 100)
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(alpha=.2)
    axis = axes[1]
    names = [f"controlled_16384_v{i:02d}_n24" for i in (0, 25, 50, 75, 95)]
    normalized = []
    for name in names:
        if name in receipt:
            entry = receipt[name]
            denom = entry["input_tokens"]["mean"]
            normalized.append((entry["target_visual_ratio"],
                               100 * entry["visual_tokens"]["mean"] / denom,
                               100 * entry["text_tokens"]["mean"] / denom,
                               100 * entry["structural_tokens"]["mean"] / denom))
    if normalized:
        targets, visual, text, structure = map(np.array, zip(*normalized, strict=True))
        x = np.arange(len(visual))
        axis.bar(x, visual, label="Visual", color="#1c6397")
        axis.bar(x, text, label="Text", bottom=visual, color="#c5703d")
        axis.bar(x, structure, label="Structural", bottom=visual+text, color="#4a4a4a")
        axis.set(xticks=x, xticklabels=[f"{int(100*t)}%" for t in targets],
                 xlabel="Picked visual fraction target", ylabel="Actual input share (%)",
                 title="16K controls, 24 requests per cell")
        axis.legend(loc="lower right", fontsize=8)
    save(fig, out, "dataset_visual_text_ratio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = read(args.audit)
    receipt = read(args.receipts)
    ids = {name: set(receipt[name]["request_ids"]) for name in ("main100", "natural149")}
    cohorts = {"Main400":data, "Natural149":[row for row in data if row["request_id"] in ids["natural149"]],
               "Main100":[row for row in data if row["request_id"] in ids["main100"]]}
    args.out.mkdir(parents=True, exist_ok=True)
    counts(cohorts, args.out)
    cumulative(cohorts, args.out)
    fractions(cohorts, receipt, args.out)


if __name__ == "__main__":
    main()
