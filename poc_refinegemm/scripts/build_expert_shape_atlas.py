#!/usr/bin/env python3
"""Build the RefineGEMM expert-row atlas from measured LLaDA2 EP4 routes.

The current production runtime still executes dense 32-row blocks.  The
primary M_e corpus below therefore filters measured top-k routes with the
future-known live mask.  It is a post-compaction *sensitivity workload*, not a
measured Epoch/FreshLane implementation.  Dense runtime timings are joined
only to preserve an honest request-critical attribution boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERTS = 256
EXPERTS_PER_RANK = 64
TOPK = 8


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(values, q * 100)) if values else 0.0


def load_cases(raw_root: Path) -> dict[tuple[str, int, int], dict]:
    cases: dict[tuple[str, int, int], dict] = {}
    for task in ("gsm8k", "humaneval"):
        files = sorted((raw_root / task / "r1").glob("shape_rank[0-3].jsonl"))
        if len(files) != 4:
            raise RuntimeError(f"expected 4 rank traces for {task}, found {files}")
        for path in files:
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("request_id") != "measured_baseline_0" or int(row.get("wave", -1)) < 0:
                    continue
                rank = int(row["rank"])
                key = (task, int(row["wave"]), int(row["layer"]))
                case = cases.setdefault(
                    key,
                    {
                        "task": task,
                        "wave": int(row["wave"]),
                        "layer": int(row["layer"]),
                        "phase": row["phase"],
                        "physical_m": int(row["physical_m_global"]),
                        "source_topk": [None] * 4,
                    },
                )
                mask = [int(v) for part in row["mask_before"] for v in part]
                start = int(row["global_row_start"])
                local_m = int(row["physical_m_local"])
                local_mask = mask[start : start + local_m]
                topk = row["topk_ids_local_source"]
                if len(local_mask) != len(topk):
                    raise RuntimeError(f"mask/top-k mismatch: {key}, rank {rank}")
                case["source_topk"][rank] = [
                    [int(e) for e in ids] for ids, live in zip(topk, local_mask) if live
                ]
    for case in cases.values():
        if any(rows is None for rows in case["source_topk"]):
            raise RuntimeError(f"incomplete case {case['task']} w{case['wave']} l{case['layer']}")
    return cases


def histogram(case: dict) -> list[int]:
    counts = [0] * EXPERTS
    for source_rows in case["source_topk"]:
        for ids in source_rows:
            for expert in ids:
                counts[expert] += 1
    return counts


def shape_metrics(values: list[int]) -> dict[str, float]:
    active = [v for v in values if v > 0]
    mean = float(np.mean(active)) if active else 0.0
    std = float(np.std(active)) if active else 0.0
    return {
        "assignments": int(sum(values)),
        "active_experts": len(active),
        "inactive_fraction": float(sum(v == 0 for v in values) / len(values)),
        "mean_m_e": mean,
        "median_m_e": float(np.median(active)) if active else 0.0,
        "p90_m_e": percentile(active, 0.9),
        "max_m_e": max(active, default=0),
        "m_e_std": std,
        "m_e_cv": std / mean if mean else 0.0,
        "tiny_le1_active_fraction": float(sum(v <= 1 for v in active) / len(active)) if active else 0.0,
        "tiny_le2_active_fraction": float(sum(v <= 2 for v in active) / len(active)) if active else 0.0,
        "tiny_le4_active_fraction": float(sum(v <= 4 for v in active) / len(active)) if active else 0.0,
        "tiny_le8_active_fraction": float(sum(v <= 8 for v in active) / len(active)) if active else 0.0,
        "large_ge16_active_fraction": float(sum(v >= 16 for v in active) / len(active)) if active else 0.0,
        "large_ge32_active_fraction": float(sum(v >= 32 for v in active) / len(active)) if active else 0.0,
    }


def read_timing(path: Path) -> dict[tuple[str, int, int, int], dict]:
    rows: dict[tuple[str, int, int, int], dict] = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset"], int(row["wave"]), int(row["layer"]), int(row["rank"]))
            rows[key] = row
    return rows


def make_control(values: list[int], mode: str) -> list[int]:
    total = int(sum(values))
    active = max(1, sum(v > 0 for v in values))
    out = [0] * len(values)
    if mode == "low_heterogeneity":
        for i in range(active):
            out[i] = total // active + int(i < total % active)
    elif mode == "high_heterogeneity":
        tiny = max(1, active // 2)
        if total < tiny:
            tiny = total
        for i in range(tiny):
            out[i] = 1
        remain = total - tiny
        big = max(1, active - tiny)
        for i in range(big):
            out[tiny + i] = remain // big + int(i < remain % big)
    else:
        raise ValueError(mode)
    return out


def plot_atlas(summary: pd.DataFrame, rows: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    global_rows = summary[summary["rank"] == -1].copy()
    metrics = [
        ("fresh_m", "Global fresh M", "02_global_fresh_m_vs_progress.png"),
        ("median_m_e", "Median active $M_e$", "03_median_me_vs_progress.png"),
        ("tiny_le4_active_fraction", "Active experts with $M_e\\leq4$", "04_tiny_fraction_vs_progress.png"),
        ("m_e_cv", "$M_e$ CV", "05_me_cv_vs_progress.png"),
    ]
    for metric, ylabel, name in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for task, group in global_rows.groupby("task"):
            binned = group.groupby(pd.cut(group["normalized_progress"], bins=np.linspace(0, 1, 11), include_lowest=True), observed=False)[metric].median()
            ax.plot(np.linspace(0.05, 0.95, 10), binned.values, marker="o", label=task)
        ax.set(xlabel="Normalized refinement progress", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    active_rows = rows[rows["m_e"] > 0]
    for ax, phase in zip(axes, ("early", "middle", "late")):
        vals = active_rows.loc[active_rows["phase"] == phase, "m_e"].clip(upper=64)
        ax.hist(vals, bins=[1, 2, 3, 5, 9, 17, 33, 65], density=True, alpha=0.8)
        ax.set_title(phase)
        ax.set_xlabel("$M_e$ (clipped at 64)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(figures / "06_phase_me_histograms.png", dpi=180)
    plt.close(fig)

    # One representative middle layer per task; each cell is global expert M_e.
    for task in ("gsm8k", "humaneval"):
        part = rows[(rows["task"] == task) & (rows["layer"] == 16)].copy()
        pivot = part.pivot_table(index="expert_global", columns="wave", values="m_e", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(np.log1p(pivot.values), aspect="auto", interpolation="nearest", cmap="magma")
        ax.set(xlabel="Refinement wave", ylabel="Expert index", title=f"{task} layer 16: log(1+$M_e$)")
        fig.colorbar(im, ax=ax, label="log(1+$M_e$)")
        fig.tight_layout()
        fig.savefig(figures / f"07_{task}_expert_step_heatmap.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.raw_root)
    timing = read_timing(args.timing)
    max_wave = {task: max(k[1] for k in cases if k[0] == task) for task in ("gsm8k", "humaneval")}
    row_records: list[dict] = []
    summary_records: list[dict] = []
    replay_candidates: list[dict] = []

    for key, case in sorted(cases.items()):
        task, wave, layer = key
        counts = histogram(case)
        fresh_m = sum(len(x) for x in case["source_topk"])
        progress = wave / max(1, max_wave[task])
        scopes = [(-1, counts)] + [(r, counts[r * 64 : (r + 1) * 64]) for r in range(4)]
        for rank, values in scopes:
            metrics = shape_metrics(values)
            timed = timing.get((task, wave, layer, rank), {}) if rank >= 0 else {}
            record = {
                "task": task,
                "wave": wave,
                "normalized_progress": progress,
                "phase": case["phase"],
                "layer": layer,
                "rank": rank,
                "physical_dense_m": case["physical_m"],
                "fresh_m": fresh_m,
                **metrics,
                "heterogeneous_25tiny_10large": int(
                    metrics["tiny_le4_active_fraction"] >= 0.25
                    and metrics["large_ge16_active_fraction"] >= 0.10
                ),
                "heterogeneous_40tiny_05large": int(
                    metrics["tiny_le4_active_fraction"] >= 0.40
                    and metrics["large_ge16_active_fraction"] >= 0.05
                ),
                "dense_measured_expert_ms": float(timed.get("expert_ms", "nan")),
                "compacted_predicted_expert_ms": float(timed.get("compact_predicted_expert_ms", "nan")),
                "evidence_scope": "measured_route_plus_future_known_live_mask; compacted timing is model sensitivity",
            }
            summary_records.append(record)
            if rank >= 0:
                for local, m_e in enumerate(values):
                    row_records.append(
                        {
                            "task": task,
                            "wave": wave,
                            "normalized_progress": progress,
                            "phase": case["phase"],
                            "layer": layer,
                            "rank": rank,
                            "expert_local": local,
                            "expert_global": rank * 64 + local,
                            "physical_dense_m": case["physical_m"],
                            "fresh_m": fresh_m,
                            "m_e": m_e,
                            "active": int(m_e > 0),
                            "source": "measured top-k filtered by future-known live mask",
                        }
                    )
                replay_candidates.append({**record, "m_e_vector": values})

    rows = pd.DataFrame(row_records)
    summary = pd.DataFrame(summary_records)
    rows.to_parquet(args.output / "EXPERT_ROW_TRACE.parquet", index=False)
    summary.to_csv(args.output / "EXPERT_SHAPE_SUMMARY.csv", index=False)

    # Select representative local vectors across phase, fresh-M, and heterogeneity.
    selected: list[dict] = []
    candidates = pd.DataFrame([{k: v for k, v in x.items() if k != "m_e_vector"} for x in replay_candidates])
    for (task, phase), group in candidates.groupby(["task", "phase"]):
        order = group.sort_values(["fresh_m", "m_e_cv"])
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            selected.append(replay_candidates[int(order.iloc[round((len(order) - 1) * q)].name)])
    unique: dict[tuple, dict] = {}
    for item in selected:
        key = (item["task"], item["wave"], item["layer"], item["rank"])
        unique[key] = item

    replay_rows: list[dict] = []
    with (args.output / "EXPERT_REPLAY.jsonl").open("w") as stream:
        case_index = 0
        for item in unique.values():
            base = {k: v for k, v in item.items() if k not in ("m_e_vector", "evidence_scope")}
            for mode, vector in (
                ("real", item["m_e_vector"]),
                ("low_heterogeneity_control", make_control(item["m_e_vector"], "low_heterogeneity")),
                ("high_heterogeneity_control", make_control(item["m_e_vector"], "high_heterogeneity")),
            ):
                metrics = shape_metrics(vector)
                out = {
                    "case_id": f"case_{case_index:03d}_{mode}",
                    "geometry": mode,
                    **base,
                    **metrics,
                    "m_e_vector": vector,
                    "same_total_assignment_control_group": f"case_{case_index:03d}",
                }
                stream.write(json.dumps(out, allow_nan=True) + "\n")
                replay_rows.append({k: v for k, v in out.items() if k != "m_e_vector"})
            case_index += 1
    pd.DataFrame(replay_rows).to_csv(args.output / "EXPERT_REPLAY_SUMMARY.csv", index=False)

    plot_atlas(summary, rows, args.output / "figures")

    global_summary = summary[summary["rank"] == -1]
    phase_summary = (
        global_summary.groupby(["task", "phase"])[
            ["fresh_m", "active_experts", "median_m_e", "p90_m_e", "max_m_e", "tiny_le4_active_fraction", "m_e_cv", "heterogeneous_25tiny_10large", "heterogeneous_40tiny_05large"]
        ]
        .median()
        .reset_index()
    )
    phase_summary.to_csv(args.output / "REFINEMENT_REGIME_ATLAS.csv", index=False)

    # Use critical-rank compacted predictions only for time-mass sensitivity.
    local = summary[summary["rank"] >= 0].copy()
    idx = local.groupby(["task", "wave", "layer"])["compacted_predicted_expert_ms"].idxmax()
    critical = local.loc[idx]
    hetero_mass = []
    for task, group in critical.groupby("task"):
        total = group["compacted_predicted_expert_ms"].sum()
        for field in ("heterogeneous_25tiny_10large", "heterogeneous_40tiny_05large"):
            mass = group.loc[group[field] == 1, "compacted_predicted_expert_ms"].sum()
            hetero_mass.append(
                {
                    "task": task,
                    "definition": field,
                    "invocation_fraction": float(group[field].mean()),
                    "predicted_compacted_expert_time_fraction": float(mass / total) if total else 0.0,
                    "evidence_scope": "model-based post-compaction sensitivity, not measured Epoch latency",
                }
            )
    pd.DataFrame(hetero_mass).to_csv(args.output / "HETEROGENEOUS_TIME_MASS.csv", index=False)

    manifest = {
        "cases": len(cases),
        "expert_rows": len(rows),
        "summary_rows": len(summary),
        "replay_geometries": len(replay_rows),
        "invariants": {
            "all_global_assignments_equal_8x_fresh_m": bool(
                (global_summary["assignments"] == TOPK * global_summary["fresh_m"]).all()
            ),
            "all_local_sums_equal_global": bool(
                summary[summary["rank"] >= 0].groupby(["task", "wave", "layer"])["assignments"].sum().equals(
                    global_summary.set_index(["task", "wave", "layer"])["assignments"]
                )
            ),
        },
        "evidence_boundary": "M_e is exact for measured routes under a future-known live-row filter; compacted latency is analytical sensitivity",
    }
    (args.output / "ATLAS_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
