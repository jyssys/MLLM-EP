#!/usr/bin/env python3
"""Aggregate controlled fanout runs, verify outputs, and draw A1-A3."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct_gap(faster: float, slower: float) -> float:
    return 100.0 * (slower - faster) / faster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in args.input:
        run_name = path.parent.name
        restart = None
        if run_name.startswith("restart"):
            restart = int(run_name.split("_", 1)[0].removeprefix("restart"))
        for row in read_jsonl(path):
            rows.append({**row, "source_run": run_name, "restart": restart})
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["M_per_source_rank"]), int(row["requested_fanout"]), str(row["backend"]))].append(row)

    summary = []
    for (num_tokens, fanout, backend), values in sorted(groups.items()):
        data = np.asarray([row["moe_total_ms"] for row in values], dtype=float)
        entry = {
            "M_per_source_rank": num_tokens,
            "fanout": fanout,
            "backend": backend,
            "n": len(values),
            "dispatch_p50_ms": statistics.median(row["dispatch_ms"] for row in values),
            "expert_p50_ms": statistics.median(row["expert_ms"] for row in values),
            "combine_p50_ms": statistics.median(row["combine_ms"] for row in values),
            "moe_p50_ms": float(np.median(data)),
            "moe_p90_ms": float(np.quantile(data, 0.9)),
            "moe_p99_ms": float(np.quantile(data, 0.99)),
            "moe_iqr_ms": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
            "rank_max_mean": values[0]["rank_load_max_over_mean"],
            "expert_max_mean": values[0]["expert_load_max_over_mean"],
            "active_experts": values[0]["active_experts"],
        }
        summary.append(entry)
    fieldnames = list(summary[0])
    with (args.output_dir / "synthetic_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    by_key = {(row["M_per_source_rank"], row["fanout"], row["backend"]): row for row in summary}
    deltas = []
    for num_tokens in sorted({row["M_per_source_rank"] for row in summary}):
        for fanout in sorted({row["fanout"] for row in summary}):
            agrs = by_key.get((num_tokens, fanout, "agrs"))
            deep = by_key.get((num_tokens, fanout, "deepep_ht"))
            if not agrs or not deep:
                continue
            delta = deep["moe_p50_ms"] - agrs["moe_p50_ms"]
            winner = "agrs" if delta > 0 else "deepep_ht"
            gap = pct_gap(min(agrs["moe_p50_ms"], deep["moe_p50_ms"]), max(agrs["moe_p50_ms"], deep["moe_p50_ms"]))
            deltas.append({
                "M_per_source_rank": num_tokens,
                "fanout": fanout,
                "agrs_p50_ms": agrs["moe_p50_ms"],
                "deepep_p50_ms": deep["moe_p50_ms"],
                "deepep_minus_agrs_ms": delta,
                "winner": winner,
                "winner_gap_pct": gap,
            })
    if deltas:
        with (args.output_dir / "synthetic_backend_delta.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(deltas[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(deltas)

        figures = args.output_dir / "figures"
        figures.mkdir(exist_ok=True)
        token_values = sorted({row["M_per_source_rank"] for row in deltas})
        fig, axes = plt.subplots(len(token_values), 1, figsize=(7, 3 * len(token_values)), squeeze=False)
        for axis, num_tokens in zip(axes[:, 0], token_values):
            for backend in ("agrs", "deepep_ht"):
                selected = [row for row in summary if row["M_per_source_rank"] == num_tokens and row["backend"] == backend]
                axis.plot([row["fanout"] for row in selected], [row["moe_p50_ms"] for row in selected], marker="o", label=backend)
            axis.set_title(f"M/source={num_tokens}")
            axis.set_xlabel("destination-rank fanout")
            axis.set_ylabel("whole MoE p50 (ms)")
            axis.grid(alpha=.3)
            axis.legend()
        fig.tight_layout()
        fig.savefig(figures / "A1_latency_vs_fanout.png", dpi=180)
        plt.close(fig)

        # Treat engine/process restart—not individual repetitions—as the
        # robustness unit.  Pooled repetition medians can otherwise hide a
        # first-use or runtime-state sign reversal.
        restart_groups: dict[tuple[int, int, int, str], list[dict]] = defaultdict(list)
        for row in rows:
            if row["restart"] is None:
                continue
            restart_groups[(int(row["restart"]), int(row["M_per_source_rank"]), int(row["requested_fanout"]), str(row["backend"]))].append(row)
        restart_summary = {}
        for key, values in restart_groups.items():
            restart_summary[key] = {
                "moe": statistics.median(float(row["moe_total_ms"]) for row in values),
                "dispatch": statistics.median(float(row["dispatch_ms"]) for row in values),
                "expert": statistics.median(float(row["expert_ms"]) for row in values),
                "combine": statistics.median(float(row["combine_ms"]) for row in values),
            }
        restart_deltas = []
        for restart in sorted({key[0] for key in restart_summary}):
            for num_tokens in sorted({key[1] for key in restart_summary if key[0] == restart}):
                for fanout in range(1, 5):
                    agrs = restart_summary.get((restart, num_tokens, fanout, "agrs"))
                    deep = restart_summary.get((restart, num_tokens, fanout, "deepep_ht"))
                    if not agrs or not deep:
                        continue
                    row = {"restart": restart, "M_per_source_rank": num_tokens, "fanout": fanout}
                    for stage in ("dispatch", "expert", "combine", "moe"):
                        row[f"agrs_{stage}_p50_ms"] = agrs[stage]
                        row[f"deepep_{stage}_p50_ms"] = deep[stage]
                        row[f"{stage}_deepep_minus_agrs_ms"] = deep[stage] - agrs[stage]
                    row["winner"] = "agrs" if row["moe_deepep_minus_agrs_ms"] > 0 else "deepep_ht"
                    restart_deltas.append(row)
        if restart_deltas:
            with (args.output_dir / "synthetic_restart_effects.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(restart_deltas[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(restart_deltas)

            robust = []
            for num_tokens in sorted({row["M_per_source_rank"] for row in restart_deltas}):
                for fanout in range(1, 5):
                    selected = [row for row in restart_deltas if row["M_per_source_rank"] == num_tokens and row["fanout"] == fanout]
                    if not selected:
                        continue
                    wins = [row["winner"] for row in selected]
                    robust.append({
                        "M_per_source_rank": num_tokens,
                        "fanout": fanout,
                        "restart_count": len(selected),
                        "agrs_win_restarts": wins.count("agrs"),
                        "deepep_win_restarts": wins.count("deepep_ht"),
                        "unanimous_winner": wins[0] if len(set(wins)) == 1 else "mixed",
                        "median_deepep_minus_agrs_ms": statistics.median(row["moe_deepep_minus_agrs_ms"] for row in selected),
                        "median_expert_deepep_minus_agrs_ms": statistics.median(row["expert_deepep_minus_agrs_ms"] for row in selected),
                    })
            with (args.output_dir / "synthetic_restart_robustness.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(robust[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(robust)

        fig, ax = plt.subplots(figsize=(8, 5))
        for num_tokens in token_values:
            selected = [row for row in deltas if row["M_per_source_rank"] == num_tokens]
            ax.plot([row["fanout"] for row in selected], [row["deepep_minus_agrs_ms"] for row in selected], marker="o", label=f"M={num_tokens}")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("destination-rank fanout")
        ax.set_ylabel("DeepEP HT - AGRS (ms)")
        ax.legend(ncol=2)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(figures / "A2_backend_delta_vs_fanout.png", dpi=180)
        plt.close(fig)

        matrix = np.full((len(token_values), 4), np.nan)
        for row in deltas:
            matrix[token_values.index(row["M_per_source_rank"]), row["fanout"] - 1] = row["deepep_minus_agrs_ms"]
        fig, ax = plt.subplots(figsize=(6, max(3, len(token_values) * .55)))
        image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(4), labels=[1, 2, 3, 4])
        ax.set_yticks(range(len(token_values)), labels=token_values)
        ax.set_xlabel("fanout")
        ax.set_ylabel("M/source")
        fig.colorbar(image, ax=ax, label="DeepEP - AGRS (ms)")
        fig.tight_layout()
        fig.savefig(figures / "A3_winner_heatmap.png", dpi=180)
        plt.close(fig)

    sample_sets: dict[str, dict[str, np.ndarray]] = {}
    for path in args.input:
        summary_path = path.with_suffix(".summary.json")
        sample_path = path.with_suffix(".samples.npz")
        if not summary_path.exists() or not sample_path.exists():
            continue
        backend = json.loads(summary_path.read_text())["backend"]
        with np.load(sample_path) as payload:
            for key in payload.files:
                sample_sets.setdefault(key, {})[backend] = payload[key]
    correctness = []
    for key, values in sorted(sample_sets.items()):
        if set(values) != {"agrs", "deepep_ht"}:
            continue
        a = values["agrs"].astype(np.float64).ravel()
        b = values["deepep_ht"].astype(np.float64).ravel()
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(a))
        correctness.append({"key": key, "cosine": cosine, "relative_l2": rel_l2})
    (args.output_dir / "synthetic_correctness.json").write_text(json.dumps(correctness, indent=2) + "\n")
    print(json.dumps({"raw_rows": len(rows), "summary_cells": len(summary), "delta_cells": len(deltas), "correctness_cells": len(correctness)}, indent=2))


if __name__ == "__main__":
    main()
