"""Draw workload and offline-cache comparisons from frozen result receipts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def relative(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0


def reuse_distribution(result, output):
    labels = {"main100": "Main100", "main400": "Main400", "natural149": "Local-media 149",
              "unique_images": "Unique 91"}
    deltas = {"prior_image_requests": [], "prior_prefix_requests": [], "prior_image_occurrences": [],
              "prior_prefix_rows": []}
    for name in labels:
        row = result["cohorts"][name]
        image = row["image_and_document"]
        prefix = row["causal_full_block_prefix"]["128"]
        deltas["prior_image_requests"].append(relative(image["requests_with_image_hit"], image["requests"]))
        deltas["prior_prefix_requests"].append(relative(prefix["requests_with_full_block_hit"], image["requests"]))
        deltas["prior_image_occurrences"].append(relative(image["image_hits_from_earlier_requests"],
                                                          image["total_image_occurrences"]))
        deltas["prior_prefix_rows"].append(relative(prefix["reusable_prompt_tokens"], prefix["prompt_tokens"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    x = np.arange(len(labels))
    for ax, modes in zip(axes, (("prior_image_requests", "prior_prefix_requests"),
                               ("prior_image_occurrences", "prior_prefix_rows")), strict=True):
        left = ax.bar(x - .18, deltas[modes[0]], .36, color="#256971", label=(
            "Any prior image" if ax is axes[0] else "Prior image occurrences"))
        right = ax.bar(x + .18, deltas[modes[1]], .36, color="#794c2c", label=(
            "Any exact 128-row block" if ax is axes[0] else "Reusable prompt rows"))
        ax.bar_label(left, fmt="%.1f", fontsize=8, padding=2)
        ax.bar_label(right, fmt="%.1f", fontsize=8, padding=2)
        ax.set_xticks(x, labels.values())
        ax.set_ylim(0, 70)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Share of requests (%)")
    axes[1].set_ylabel("Share of each representation (%)")
    fig.suptitle("Frozen cohort recurrence before capacity and concurrency limits", fontweight="bold")
    fig.savefig(output, dpi=165)
    plt.close(fig)


def capacity_curve(result, output):
    cohorts = {"natural149_manifest": "Original frozen order", "natural149_grouped_manifest": "Adjacent document arrivals",
               "main400": "Main400 frozen order"}
    policies = {0: ("Prefix KV only", "#794c2c"), .5: ("Half embedding / half KV", "#3c5587"),
                1: ("Embedding only", "#256971")}
    fig, axes = plt.subplots(2, len(cohorts), figsize=(12.2, 6.1), constrained_layout=True, sharex=True, sharey="row")
    for column, (name, title) in enumerate(cohorts.items()):
        for fraction, (policy, color) in policies.items():
            data = sorted((item for item in result["results"] if item["cohort"] == name and
                           item["embedding_fraction"] == fraction and item["budget_GiB"] > 0),
                          key=lambda item: item["budget_GiB"])
            budget = [item["budget_GiB"] for item in data]
            axes[0, column].plot(budget, [relative(item["encode_avoided_visual_rows"], item["visual_rows"]) for item in data],
                                marker="o", markersize=3, color=color, label=policy)
            axes[1, column].plot(budget, [relative(item["prefill_avoided_rows"], item["prompt_rows"]) for item in data],
                                marker="o", markersize=3, color=color, label=policy)
        axes[0, column].set_title(title, fontsize=10)
        axes[1, column].set_xlabel("Cache budget (GiB)")
        for ax in axes[:, column]:
            ax.set_xscale("log", base=2)
            ax.set_xticks([.25, 1, 4, 16, 64], ["0.25", "1", "4", "16", "64"])
            ax.grid(alpha=.18)
            ax.spines[["right", "top"]].set_visible(False)
    axes[0, 0].set_ylabel("Visual rows eligible to skip (%)")
    axes[1, 0].set_ylabel("Prompt rows eligible to skip (%)")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Sequential byte-bounded LRU replay\nPrefix-covered images count once if Encode can be bypassed",
                 fontweight="bold")
    fig.savefig(output, dpi=165)
    plt.close(fig)


def stress_curve(data, output):
    traces = (("w5_natural_two_wave149", "Natural questions · two waves", "#26847c"),
              ("w5_eviction188", "Cyclic repeat stress · 188 arrivals", "#884f33"))
    fig, ax = plt.subplots(figsize=(7.7, 4.2), constrained_layout=True)
    for name, label, color in traces:
        row = data["cohorts"][name]
        ax.plot([point["P_KV_GiB"] for point in row["simulation"]],
                [relative(point["second_wave_prefix_hit_requests"], row["second_wave_requests"]) for point in row["simulation"]],
                marker="o", color=color, label=label)
    ax.axvline(42, color="#24414e", linestyle="--", linewidth=1)
    ax.text(41, 4, "Native P budget", rotation=90, va="bottom", ha="right", color="#24414e", fontsize=8)
    ax.set(xlabel="Prefix KV budget (GiB)", ylabel="Second-wave requests eligible to hit (%)", ylim=(-3, 104))
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(alpha=.2)
    ax.set_title("Two-wave cache-capacity screening under generic LRU", fontweight="bold")
    fig.savefig(output, dpi=165)
    plt.close(fig)


def calibrated_policies(data, output):
    cohorts = (("natural149_manifest", "Natural149 in manifest order"),
               ("w5_eviction188_manifest", "Controlled cyclic eviction stress"))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4), constrained_layout=True)
    static_styles = {0: ("KV only", "#855d38"), .25: ("25:75 embedding/KV", "#327076"),
                     .5: ("50:50 embedding/KV", "#364776"), 1: ("Embedding only", "#ab6264")}
    for ax, (cohort, title) in zip(axes, cohorts, strict=True):
        runs = data["cohorts"][cohort]["runs"]
        runs = [row for row in runs if not row["allow_prefix_to_bypass_encode"] and row["budget_GiB"] in (4, 16, 64)]
        for fraction, (label, color) in static_styles.items():
            points = [row for row in runs if row["embedding_fraction"] == fraction]
            estimate = [row["stages"]["modeled_E_P_compute_stage_max_after_hits"] / 1000 for row in points]
            ax.plot([row["budget_GiB"] for row in points], estimate, marker="o", markersize=4, color=color, label=label)
        for name, label, marker in (("generic", "Cost and frequency", "D"), ("scope", "Plus image reuse scope", "x")):
            points = [row for row in runs if row["policy"] == name]
            estimate = [row["stages"]["modeled_E_P_compute_stage_max_after_hits"] / 1000 for row in points]
            ax.plot([row["budget_GiB"] for row in points], estimate, marker=marker, markersize=6,
                    markerfacecolor="none", color="#191919", linestyle="none", label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 16, 64], ["4", "16", "64"])
        ax.set(xlabel="Matched cache budget (GiB)", title=title)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Modeled dominant E/P compute demand (s)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Offline policy replay with isolated GPU calibration", fontweight="bold")
    fig.savefig(output, dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, plot, target in (("workload_reuse", reuse_distribution, "workload_reuse_distribution"),
                               ("cache_replay", capacity_curve, "workload_cache_capacity_replay"),
                               ("capacity_stress_workloads", stress_curve, "workload_cache_stress"),
                               ("matched_policy_screen_calibrated", calibrated_policies, "workload_matched_policy_screen")):
        plot(json.loads((args.results / f"{name}.json").read_bytes()), args.out / f"{target}.png")


if __name__ == "__main__":
    main()
