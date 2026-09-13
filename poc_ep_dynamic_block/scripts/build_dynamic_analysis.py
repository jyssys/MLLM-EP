#!/usr/bin/env python3
"""Compare actual dynamic trajectories with the global fixed-B frontier.

A dynamic schedule is never credited against only the fixed policies bundled
in the same run. It must beat the fastest measured fixed-B point at the same
or better bounded quality. The isolated mini=1 tournament is analysed
separately as a future-knowledge per-request heterogeneity oracle; it is not
reported as a deployable controller.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIR_RE = re.compile(
    r"schedule_(?P<plan>.+)_(?P<task>gsm8k|humaneval)_n(?P<n>\d+)_m(?P<mini>\d+)_"
    r"L(?P<target>\d+)_t(?P<threshold>[0-9.]+)_r(?P<repeat>\d+)"
    r"(?:_(?P<barrier>barrier1))?$"
)


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def global_frontier_comparison() -> list[dict]:
    static = [
        row for row in read_csv(ROOT / "STATIC_POINT_SUMMARY.csv")
        if row["decoder_control"] == "serving_native"
    ]
    policies = read_csv(ROOT / "POLICY_COMPARISON.csv")
    write_csv(
        ROOT / "EXHAUSTIVE_SCHEDULE_RESULTS.csv",
        [row for row in policies if row["plan"] == "exhaustive_128"],
    )
    output = []
    for row in policies:
        if row["fixed"].lower() == "true":
            continue
        candidates = [
            item for item in static
            if item["task"] == row["task"]
            and int(item["sample_count"]) == int(row["n"])
            and int(item["target_total_length"]) == int(row["target"])
        ]
        for fixed in policies:
            if (
                fixed["fixed"].lower() == "true"
                and fixed["task"] == row["task"]
                and int(fixed["n"]) == int(row["n"])
                and int(fixed["target"]) == int(row["target"])
            ):
                fixed_schedule = json.loads(fixed["block_schedule"])
                candidates.append(
                    {
                        "block_length": fixed_schedule[0],
                        "mini_batch_size": fixed["mini"],
                        "threshold": fixed["threshold"],
                        "bct_median_s": fixed["model_seconds"],
                        "accuracy_median": fixed["accuracy"],
                        "source_regimes": "actual_fixed_schedule_control",
                    }
                )
        if not candidates:
            continue
        quality = float(row["accuracy"])
        quality_safe = [item for item in candidates if float(item["accuracy_median"]) >= quality]
        fastest_any = min(candidates, key=lambda item: float(item["bct_median_s"]))
        fastest_safe = (
            min(quality_safe, key=lambda item: float(item["bct_median_s"]))
            if quality_safe else None
        )
        dynamic_time = float(row["model_seconds"])
        output.append(
            {
                "task": row["task"],
                "plan": row["plan"],
                "repeat": row["repeat"],
                "schedule_barrier": row["schedule_barrier"],
                "policy": row["policy"],
                "block_schedule": row["block_schedule"],
                "dynamic_seconds": dynamic_time,
                "dynamic_accuracy": quality,
                "dynamic_nfe": row["nfe"],
                "global_fastest_seconds": fastest_any["bct_median_s"],
                "latency_only_gain_vs_global_fixed_percent": 100 * (
                    float(fastest_any["bct_median_s"]) - dynamic_time
                ) / float(fastest_any["bct_median_s"]),
                "quality_safe_fixed_available": fastest_safe is not None,
                "quality_matched_fixed_block": fastest_safe["block_length"] if fastest_safe else "",
                "quality_matched_fixed_mini": fastest_safe["mini_batch_size"] if fastest_safe else "",
                "quality_matched_fixed_threshold": fastest_safe["threshold"] if fastest_safe else "",
                "quality_matched_fixed_seconds": fastest_safe["bct_median_s"] if fastest_safe else "",
                "quality_matched_fixed_accuracy": fastest_safe["accuracy_median"] if fastest_safe else "",
                "quality_matched_gain_vs_global_fixed_percent": (
                    100 * (float(fastest_safe["bct_median_s"]) - dynamic_time)
                    / float(fastest_safe["bct_median_s"])
                    if fastest_safe else ""
                ),
                "evidence": "clean actual trajectory versus all measured fixed-B/mini/threshold points",
            }
        )
    write_csv(ROOT / "DYNAMIC_VS_GLOBAL_STATIC.csv", output)
    return output


def request_heterogeneity() -> tuple[list[dict], list[dict]]:
    detail_rows = []
    summaries = []
    for directory in sorted((ROOT / "results" / "schedules").glob("schedule_request_tournament_*")):
        match = DIR_RE.fullmatch(directory.name)
        quality_path = directory / "schedule_quality.json"
        if not match or not quality_path.exists():
            continue
        quality = {row["policy"]: row for row in json.loads(quality_path.read_text())}
        by_request: dict[int, list[dict]] = {}
        for path in sorted(directory.glob("policy_*.json")):
            payload = json.loads(path.read_text())
            policy = payload["policy"]["name"]
            schedule = [int(value) for value in payload["policy"]["block_schedule"]]
            fixed = len(set(schedule)) == 1
            passes = {int(item["id"]): bool(item["pass"]) for item in quality[policy]["details"]}
            for physical in payload["rows"]:
                request_id = int(physical["original_index"])
                by_request.setdefault(request_id, []).append(
                    {
                        "policy": policy,
                        "schedule": schedule,
                        "fixed": fixed,
                        "seconds": float(physical["physical_batch_model_seconds"]),
                        "nfe": int(physical["physical_batch_nfe"]),
                        "pass": passes[request_id],
                    }
                )
        preferred = Counter()
        dynamic_wins = 0
        gains = []
        for request_id, rows in sorted(by_request.items()):
            fixed_rows = [row for row in rows if row["fixed"]]
            dynamic_rows = [row for row in rows if not row["fixed"]]
            fixed_passes = [row for row in fixed_rows if row["pass"]]
            reference = min(fixed_passes or fixed_rows, key=lambda row: row["seconds"])
            candidates = [row for row in dynamic_rows if (not reference["pass"] or row["pass"])]
            dynamic = min(candidates, key=lambda row: row["seconds"]) if candidates else None
            preferred[str(reference["schedule"][0])] += 1
            gain = (
                100 * (reference["seconds"] - dynamic["seconds"]) / reference["seconds"]
                if dynamic else np.nan
            )
            if dynamic and gain > 0:
                dynamic_wins += 1
            if dynamic:
                gains.append(gain)
            detail_rows.append(
                {
                    "task": match.group("task"),
                    "request_id": request_id,
                    "fixed_reference_policy": reference["policy"],
                    "fixed_reference_B": reference["schedule"][0],
                    "fixed_reference_seconds": reference["seconds"],
                    "fixed_reference_pass": reference["pass"],
                    "best_safe_dynamic_policy": dynamic["policy"] if dynamic else "NONE",
                    "best_safe_dynamic_schedule": json.dumps(dynamic["schedule"], separators=(",", ":")) if dynamic else "[]",
                    "dynamic_seconds": dynamic["seconds"] if dynamic else "",
                    "dynamic_pass": dynamic["pass"] if dynamic else "",
                    "dynamic_gain_percent": gain if dynamic else "",
                    "oracle_scope": "isolated-request future-knowledge; mini=1; not a live controller",
                }
            )
        summaries.append(
            {
                "task": match.group("task"),
                "requests": len(by_request),
                "preferred_fixed_B_histogram": json.dumps(dict(sorted(preferred.items())), separators=(",", ":")),
                "distinct_preferred_fixed_B": len(preferred),
                "dynamic_win_requests": dynamic_wins,
                "dynamic_win_fraction": dynamic_wins / len(by_request),
                "median_safe_dynamic_gain_percent": float(np.median(gains)) if gains else "",
                "mean_safe_dynamic_gain_percent": float(np.mean(gains)) if gains else "",
                "evidence": "actual isolated trajectories; per-request future knowledge",
            }
        )
    write_csv(ROOT / "REQUEST_HETEROGENEITY.csv", detail_rows)
    write_csv(ROOT / "REQUEST_HETEROGENEITY_SUMMARY.csv", summaries)
    return detail_rows, summaries


def plots(global_rows: list[dict], request_rows: list[dict]) -> None:
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    if global_rows:
        fig, axis = plt.subplots(figsize=(8.5, 4.4))
        labels = [f"{row['task']}:{row['policy']}" for row in global_rows]
        values = [float(row["quality_matched_gain_vs_global_fixed_percent"]) for row in global_rows]
        colors = ["#2c7fb8" if value >= 0 else "#d95f0e" for value in values]
        axis.bar(np.arange(len(values)), values, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(values)))
        axis.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        axis.set_ylabel("gain vs global quality-matched fixed frontier (%)")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "19_quality_safe_dynamic_vs_fixed.png", dpi=180)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(8.5, 4.4))
        latency_values = [
            float(row["latency_only_gain_vs_global_fixed_percent"])
            for row in global_rows
        ]
        axis.bar(
            np.arange(len(latency_values)),
            latency_values,
            color=["#2c7fb8" if value >= 0 else "#d95f0e" for value in latency_values],
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        axis.set_ylabel("latency-only gain vs global fastest fixed (%)")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "18_latency_only_dynamic_vs_fixed.png", dpi=180)
        plt.close(fig)
    if request_rows:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        for axis, task in zip(axes, ("gsm8k", "humaneval")):
            subset = [row for row in request_rows if row["task"] == task]
            counts: dict[int, int] = {}
            for row in subset:
                block = int(row["fixed_reference_B"])
                counts[block] = counts.get(block, 0) + 1
            blocks = sorted(counts)
            axis.bar([str(block) for block in blocks], [counts[block] for block in blocks])
            axis.set_title(task)
            axis.set_xlabel("per-request quality-safe winning fixed B")
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("request count")
        fig.tight_layout()
        fig.savefig(figures / "17_request_winning_B_distribution.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        for axis, task in zip(axes, ("gsm8k", "humaneval")):
            subset = [row for row in request_rows if row["task"] == task]
            axis.bar(
                [int(row["request_id"]) for row in subset],
                [float(row["dynamic_gain_percent"]) for row in subset],
                color=["#2c7fb8" if float(row["dynamic_gain_percent"]) >= 0 else "#d95f0e" for row in subset],
            )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(task)
            axis.set_xlabel("request id")
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("best safe mixed schedule gain vs per-request fixed (%)")
        fig.tight_layout()
        fig.savefig(figures / "18_request_schedule_heterogeneity.png", dpi=180)
        plt.close(fig)

    policies = read_csv(ROOT / "POLICY_COMPARISON.csv")
    examples = [
        row for row in policies
        if row["plan"] == "dynamic_core" and int(row["repeat"]) == 1
        and row["schedule_barrier"].lower() == "false"
    ]
    if examples:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
        for axis, task in zip(axes, ("gsm8k", "humaneval")):
            subset = sorted(
                (row for row in examples if row["task"] == task),
                key=lambda row: float(row["model_seconds"]),
            )
            axis.bar(
                np.arange(len(subset)),
                [float(row["model_seconds"]) for row in subset],
                color=["#4daf4a" if row["fixed"].lower() == "true" else "#984ea3" for row in subset],
            )
            axis.set_xticks(np.arange(len(subset)))
            axis.set_xticklabels([row["policy"] for row in subset], rotation=60, ha="right", fontsize=7)
            axis.set_title(task)
            axis.set_ylabel("clean model wall (s)")
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "20_dynamic_schedule_examples.png", dpi=180)
        plt.close(fig)

    long_rows = [
        row for row in policies
        if int(row["repeat"]) in (7, 8)
        and row["plan"] in ("dynamic_wide", "long_16_control")
    ]
    write_csv(ROOT / "LONG_GENERATION_RESULTS.csv", long_rows)
    if long_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for axis, task in zip(axes, ("gsm8k", "humaneval")):
            subset = sorted(
                (row for row in long_rows if row["task"] == task),
                key=lambda row: float(row["model_seconds"]),
            )
            axis.bar(
                np.arange(len(subset)),
                [float(row["model_seconds"]) for row in subset],
                color=["#4daf4a" if row["fixed"].lower() == "true" else "#984ea3" for row in subset],
            )
            axis.set_xticks(np.arange(len(subset)))
            axis.set_xticklabels([row["policy"] for row in subset], rotation=70, ha="right", fontsize=6)
            axis.set_title(task)
            axis.set_ylabel("clean long-generation wall (s)")
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "23_long_generation_validation.png", dpi=180)
        plt.close(fig)


def main() -> None:
    global_rows = global_frontier_comparison()
    request_rows, summaries = request_heterogeneity()
    plots(global_rows, request_rows)
    print(
        f"wrote {len(global_rows)} dynamic/global comparisons, "
        f"{len(request_rows)} request rows, {len(summaries)} request summaries"
    )


if __name__ == "__main__":
    main()
