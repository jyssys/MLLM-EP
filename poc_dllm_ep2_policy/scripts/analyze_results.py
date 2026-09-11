#!/usr/bin/env python3
"""Aggregate the EP2 mechanism-screening artifacts into tables and plots.

All request-level conclusions use clean runs.  The observer-heavy trajectory
captures are used only for structural attribution and an optimistic oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKENDS = ("naive", "deepep_high_throughput")
STAGES = ("dispatch_ms", "expert_ms", "combine_ms", "moe_ms")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def active_bin(ratio: float) -> str:
    if ratio > 0.67:
        return "early"
    if ratio > 0.33:
        return "middle"
    return "late"


def collect_scaling(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    run_specs = {
        "naive": sorted((root / "scaling").glob("naive_run[1-3]")),
        "deepep_high_throughput": sorted((root / "scaling").glob("deepep_ht_run[1-3]")),
    }
    per_shape_restart: list[dict[str, Any]] = []
    output_by_key: dict[tuple[str, int, int, int], float] = {}

    for backend, run_dirs in run_specs.items():
        for restart, run_dir in enumerate(run_dirs, start=1):
            rank_rows = {}
            for rank_file in sorted(run_dir.glob("scaling_*_rank*.jsonl")):
                for row in read_jsonl(rank_file):
                    key = (row["active_positions"], row["batch_size"], row["rep"])
                    rank_rows.setdefault(key, []).append(row)
                    output_by_key[(backend, row["active_positions"], row["batch_size"],
                                   row["rank"])] = row["output_sum"]
            critical = []
            for (active, batch, rep), rows in rank_rows.items():
                item = {
                    "backend": backend,
                    "restart": restart,
                    "active_positions": active,
                    "batch_size": batch,
                    "local_m": active * batch,
                    "rep": rep,
                }
                # Logical EP latency is the critical (slowest) rank duration.
                for stage in STAGES:
                    item[stage] = max(float(row[stage]) for row in rows)
                critical.append(item)
            shapes = sorted({(row["active_positions"], row["batch_size"])
                             for row in critical})
            for active, batch in shapes:
                selected = [row for row in critical
                            if row["active_positions"] == active and row["batch_size"] == batch]
                summary = {
                    "backend": backend,
                    "restart": restart,
                    "active_positions": active,
                    "batch_size": batch,
                    "local_m": active * batch,
                }
                for stage in STAGES:
                    summary[stage] = median(row[stage] for row in selected)
                per_shape_restart.append(summary)

    shape_rows: list[dict[str, Any]] = []
    shape_keys = sorted({(row["backend"], row["active_positions"], row["batch_size"])
                        for row in per_shape_restart})
    for backend, active, batch in shape_keys:
        selected = [row for row in per_shape_restart
                    if (row["backend"], row["active_positions"], row["batch_size"])
                    == (backend, active, batch)]
        item = {
            "backend": backend,
            "active_positions": active,
            "batch_size": batch,
            "local_m": active * batch,
            "restarts": len(selected),
        }
        for stage in STAGES:
            values = [row[stage] for row in selected]
            item[stage] = median(values)
            item[stage.replace("_ms", "_restart_cv_pct")] = (
                float(np.std(values, ddof=1) / np.mean(values) * 100) if len(values) > 1 else 0.0)
        item["dispatch_fraction"] = item["dispatch_ms"] / item["moe_ms"]
        item["expert_fraction"] = item["expert_ms"] / item["moe_ms"]
        item["combine_fraction"] = item["combine_ms"] / item["moe_ms"]
        shape_rows.append(item)

    m_rows: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for local_m in sorted({row["local_m"] for row in shape_rows if row["backend"] == backend}):
            selected = [row for row in shape_rows
                        if row["backend"] == backend and row["local_m"] == local_m]
            item = {"backend": backend, "local_m": local_m,
                    "shape_factorizations": len(selected)}
            for stage in STAGES:
                item[stage] = median(row[stage] for row in selected)
            item["dispatch_fraction"] = item["dispatch_ms"] / item["moe_ms"]
            item["expert_fraction"] = item["expert_ms"] / item["moe_ms"]
            item["combine_fraction"] = item["combine_ms"] / item["moe_ms"]
            m_rows.append(item)

    # Equal routes and inputs imply identical sums; retain the maximum observed
    # backend delta as a compact scaling-correctness check.
    deltas = []
    for key, value in output_by_key.items():
        backend, active, batch, rank = key
        other = "deepep_high_throughput" if backend == "naive" else "naive"
        if (other, active, batch, rank) in output_by_key:
            deltas.append(abs(value - output_by_key[(other, active, batch, rank)]))
    return shape_rows, m_rows, max(deltas, default=math.nan)


def collect_requests(root: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in sorted((root / "request_runs_warm5").iterdir()):
        rank_payloads = [json.loads(path.read_text(encoding="utf-8"))
                         for path in sorted(directory.glob("*rank*.json"))]
        if not rank_payloads:
            continue
        rank0 = next(item for item in rank_payloads if item["rank"] == 0)
        rows.append({
            "run": directory.name,
            "backend": rank0["backend"],
            "request_ms": max(float(item["request_ms"]) for item in rank_payloads),
            "nfe": rank0["nfe"],
            "output_sha256": __import__("hashlib").sha256(
                json.dumps(rank0["output_ids"], separators=(",", ":")).encode()).hexdigest(),
        })
    return rows


def forward_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    grouped: dict[tuple[int, int, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["block_id"], row["iteration_id"], row["model_calls"],
               row["model_positions"], row["masked_before"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(grouped.items()):
        block, iteration, model_call, model_positions, masked = key
        total = int(group[0]["total_block_positions"])
        result.append({
            "block_id": block,
            "iteration_id": iteration,
            "model_call": model_call,
            "model_positions": model_positions,
            "masked_before": masked,
            "total_block_positions": total,
            "active_ratio": masked / total,
            "phase": active_bin(masked / total),
            "layers": len(group),
            "moe_layers_ms": sum(float(row["moe_ms"]) for row in group),
        })
    return result


def trajectory_oracle(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {
        "naive": root / "trajectories/gen256_prefix_noeos_trace/trajectory_gen256_prefix_b1_noeos_rank0.jsonl",
        "deepep_high_throughput": root / "trajectories/gen256_prefix_noeos_ht_trace/trajectory_gen256_prefix_b1_noeos_rank0.jsonl",
    }
    backend_rows = {backend: forward_rows(path) for backend, path in paths.items()}
    keyed = {backend: {
        (row["block_id"], row["iteration_id"], row["model_call"], row["model_positions"]): row
        for row in rows} for backend, rows in backend_rows.items()}
    common = sorted(set(keyed["naive"]) & set(keyed["deepep_high_throughput"]))
    paired = []
    for key in common:
        a, b = keyed["naive"][key], keyed["deepep_high_throughput"][key]
        assert a["masked_before"] == b["masked_before"]
        item = {k: a[k] for k in ("block_id", "iteration_id", "model_call",
                                  "model_positions", "masked_before",
                                  "total_block_positions", "active_ratio", "phase")}
        item["naive_ms"] = a["moe_layers_ms"]
        item["deepep_high_throughput_ms"] = b["moe_layers_ms"]
        item["winner"] = min(BACKENDS, key=lambda backend: item[f"{backend}_ms"])
        paired.append(item)

    totals = {backend: sum(row[f"{backend}_ms"] for row in paired) for backend in BACKENDS}
    best_backend = min(totals, key=totals.get)
    best_static = totals[best_backend]
    perfect = sum(min(row[f"{backend}_ms"] for backend in BACKENDS) for row in paired)

    # Optimistic in-sample single-threshold search.  Both orientations are
    # allowed; switching overhead is intentionally zero here.
    candidates = sorted({row["active_ratio"] for row in paired} | {0.0, 1.000001})
    policies = []
    for high_backend in BACKENDS:
        low_backend = next(item for item in BACKENDS if item != high_backend)
        for threshold in candidates:
            total = sum(row[f"{high_backend}_ms"] if row["active_ratio"] >= threshold
                        else row[f"{low_backend}_ms"] for row in paired)
            policies.append((total, threshold, high_backend, low_backend))
    policy_total, threshold, high_backend, low_backend = min(policies)
    summary = {
        "paired_forwards": len(paired),
        "backend_totals_ms": totals,
        "best_static_backend": best_backend,
        "best_static_ms": best_static,
        "perfect_dynamic_ms": perfect,
        "perfect_dynamic_moe_gain_pct": (1 - perfect / best_static) * 100,
        "oracle_saved_ms": best_static - perfect,
        "threshold": threshold,
        "threshold_high_backend": high_backend,
        "threshold_low_backend": low_backend,
        "threshold_policy_ms": policy_total,
        "threshold_policy_moe_gain_pct": (1 - policy_total / best_static) * 100,
        "threshold_recovery_pct": ((best_static - policy_total) / (best_static - perfect) * 100
                                   if best_static > perfect else 0.0),
    }
    return paired, summary


def trajectory_structure(root: Path) -> list[dict[str, Any]]:
    specs = [
        ("batch1_cacheoff", root / "trajectories/gen64_cacheoff_trace/trajectory_gen64_b1_trace_rank0.jsonl"),
        ("batch4_cacheoff", root / "trajectories/gen64_batch4_cacheoff_trace/trajectory_gen64_batch4_cacheoff_rank0.jsonl"),
        ("batch1_prefix_gen256", root / "trajectories/gen256_prefix_noeos_trace/trajectory_gen256_prefix_b1_noeos_rank0.jsonl"),
    ]
    output = []
    for label, path in specs:
        for row in forward_rows(path):
            output.append({"trajectory": label, **row})
    return output


def make_plots(out: Path, shapes: list[dict[str, Any]], m_rows: list[dict[str, Any]],
               paired: list[dict[str, Any]], structure: list[dict[str, Any]],
               oracle: dict[str, Any], request_rows: list[dict[str, Any]]) -> None:
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for backend in BACKENDS:
        rows = [row for row in m_rows if row["backend"] == backend]
        x = [row["local_m"] for row in rows]
        for stage in ("dispatch_ms", "expert_ms", "combine_ms"):
            axes[0].plot(x, [row[stage] for row in rows], marker="o",
                         label=f"{backend}:{stage[:-3]}")
        axes[1].plot(x, [row["moe_ms"] for row in rows], marker="o", label=backend)
    axes[0].set(xlabel="Local physical M", ylabel="Rank-critical latency (ms)",
                title="EP2 stage scaling")
    axes[1].set(xlabel="Local physical M", ylabel="MoE latency (ms)",
                title="Backend latency vs work")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([2, 8, 32, 128, 256], ["2", "8", "32", "128", "256"])
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "01_stage_and_backend_scaling.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for backend in BACKENDS:
        rows = [row for row in m_rows if row["backend"] == backend]
        x = np.array([row["local_m"] for row in rows])
        axes[0].stackplot(x,
                          [row["dispatch_fraction"] for row in rows],
                          [row["expert_fraction"] for row in rows],
                          [row["combine_fraction"] for row in rows],
                          labels=("dispatch", "expert", "combine"), alpha=.75)
        axes[0].set_title(f"Stage fractions: {backend}")
        break
    for batch in sorted({row["batch_size"] for row in shapes}):
        rows = [row for row in shapes if row["backend"] == "naive" and row["batch_size"] == batch]
        axes[1].plot([row["active_positions"] for row in rows],
                     [row["expert_fraction"] for row in rows], marker="o", label=f"batch={batch}")
    axes[0].set(xlabel="Local physical M", ylabel="Fraction of full MoE")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([2, 8, 32, 128, 256], ["2", "8", "32", "128", "256"])
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Active-size label in fixed replay", ylabel="Expert / MoE",
                title="Stage fraction vs batch factorization")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([2, 8, 32, 128, 256], ["2", "8", "32", "128", "256"])
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(plots / "02_stage_fractions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label in ("batch1_cacheoff", "batch4_cacheoff"):
        rows = [row for row in structure if row["trajectory"] == label]
        axes[0].plot(range(len(rows)), [row["masked_before"] for row in rows], marker=".", label=label)
        axes[1].plot(range(len(rows)), [row["model_positions"] for row in rows], marker=".", label=label)
    axes[0].set(title="Unresolved positions", xlabel="Physical forward sequence", ylabel="Masked positions")
    axes[1].set(title="Physical MoE rows", xlabel="Physical forward sequence", ylabel="M per DP rank")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "03_active_vs_physical_work.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(paired))
    axes[0].plot(x, [row["naive_ms"] for row in paired], marker=".", label="naive")
    axes[0].plot(x, [row["deepep_high_throughput_ms"] for row in paired], marker=".", label="DeepEP HT")
    axes[0].set(title="Backend cost along real trajectory", xlabel="Physical forward", ylabel="Sum of 16 MoE layers (ms)")
    axes[0].legend()
    counts = {phase: {backend: 0 for backend in BACKENDS} for phase in ("early", "middle", "late")}
    for row in paired:
        counts[row["phase"]][row["winner"]] += 1
    phases = list(counts)
    bottom = np.zeros(len(phases))
    for backend in BACKENDS:
        values = np.array([counts[phase][backend] for phase in phases])
        axes[1].bar(phases, values, bottom=bottom, label=backend)
        bottom += values
    axes[1].set(title="Observed winner by active-ratio bin", ylabel="Physical forwards")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(plots / "04_real_trajectory_winners.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Best static", "Perfect dynamic", "Best active threshold"]
    values = [oracle["best_static_ms"], oracle["perfect_dynamic_ms"], oracle["threshold_policy_ms"]]
    axes[0].bar(labels, values)
    axes[0].set(ylabel="Observer-heavy MoE-layer sum (ms)", title="Static vs optimistic oracles")
    grouped = {backend: [row["request_ms"] for row in request_rows if row["backend"] == backend]
               for backend in BACKENDS}
    axes[1].boxplot([grouped[backend] for backend in BACKENDS], tick_labels=["naive", "DeepEP HT"])
    axes[1].set(ylabel="Clean request latency (ms)", title="3 independent warm-5 restarts")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(plots / "05_oracle_and_clean_request.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    table_rows = [
        ["Peers", "1 remote peer", "3 remote peers; remeasure"],
        ["Experts/rank", "32", "16"],
        ["Fanout", "<=2 ranks", "<=4 ranks"],
        ["Message size", "EP2 measured", "smaller/rank likely"],
        ["Winner", "HT static", "unknown"],
        ["Crossover", "none", "must remeasure"],
    ]
    axis.axis("off")
    axis.table(cellText=table_rows, colLabels=["Variable", "EP2", "EP4 transfer"], loc="center")
    axis.set_title("EP2-to-EP4 transfer-risk table")
    fig.tight_layout()
    fig.savefig(plots / "06_ep2_ep4_transfer_risk.png", dpi=180)
    plt.close(fig)

    phase_rows = []
    for phase in ("early", "middle", "late"):
        selected = [row for row in paired if row["phase"] == phase]
        phase_rows.append((phase,
                           np.mean([row["naive_ms"] for row in selected]),
                           np.mean([row["deepep_high_throughput_ms"] for row in selected])))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(phase_rows))
    axis.bar(x - .18, [row[1] for row in phase_rows], .36, label="naive")
    axis.bar(x + .18, [row[2] for row in phase_rows], .36, label="DeepEP HT")
    axis.set_xticks(x, [row[0] for row in phase_rows])
    axis.set(ylabel="Mean 16-layer MoE time/forward (ms)", title="Early/middle/late latency breakdown")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "07_phase_latency_breakdown.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(["best static", "perfect future", "active threshold"],
             [oracle["best_static_ms"], oracle["perfect_dynamic_ms"], oracle["threshold_policy_ms"]])
    axis.set(ylabel="Observer-heavy MoE-layer sum (ms)", title="Best static vs dynamic and causal threshold")
    axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(plots / "08_static_dynamic_threshold.png", dpi=180)
    plt.close(fig)

    shared_m = sorted({row["local_m"] for row in m_rows})
    wins = []
    for local_m in shared_m:
        values = {row["backend"]: row["moe_ms"] for row in m_rows if row["local_m"] == local_m}
        wins.append(0 if values["naive"] <= values["deepep_high_throughput"] else 1)
    fig, axis = plt.subplots(figsize=(10, 2.3))
    image = axis.imshow([wins], aspect="auto", cmap=matplotlib.colors.ListedColormap(["#4c78a8", "#f58518"]), vmin=0, vmax=1)
    axis.set_xticks(range(len(shared_m)), shared_m)
    axis.set_yticks([0], ["winner"])
    axis.set_xlabel("Local physical M")
    axis.set_title("Fixed-work backend winner map (blue=Naive, orange=DeepEP HT)")
    fig.tight_layout()
    fig.savefig(plots / "09_backend_winner_map.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for batch in sorted({row["batch_size"] for row in shapes}):
        rows = [row for row in shapes if row["backend"] == "naive" and row["batch_size"] == batch]
        axis.plot([row["active_positions"] for row in rows],
                  [row["expert_fraction"] for row in rows], marker="o", label=f"batch={batch}")
    axis.set_xscale("log", base=2)
    axis.set_xticks([2, 8, 32, 128, 256], ["2", "8", "32", "128", "256"])
    axis.set(xlabel="Active-size label", ylabel="Expert / MoE", title="Stage fraction vs batch size")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "10_stage_fraction_vs_batch.png", dpi=180)
    plt.close(fig)

    rows = [row for row in structure if row["trajectory"] == "batch1_prefix_gen256"]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(range(len(rows)), [row["masked_before"] for row in rows], marker="o", label="unresolved")
    axis.plot(range(len(rows)), [row["model_positions"] for row in rows], marker=".", label="physical M")
    axis.set(xlabel="Physical forward", ylabel="Positions", title="Real denoising state vs physical EP work")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "11_real_active_positions_vs_iteration.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    shapes, m_rows, scaling_output_delta = collect_scaling(root)
    requests = collect_requests(root)
    paired, oracle = trajectory_oracle(root)
    structure = trajectory_structure(root)
    write_csv(analysis / "scaling_by_shape.csv", shapes)
    write_csv(analysis / "scaling_by_physical_m.csv", m_rows)
    write_csv(analysis / "clean_request_runs.csv", requests)
    write_csv(analysis / "paired_real_trajectory.csv", paired)
    write_csv(analysis / "trajectory_structure.csv", structure)

    request_medians = {
        backend: median(row["request_ms"] for row in requests if row["backend"] == backend)
        for backend in BACKENDS
    }
    output_hashes = {row["output_sha256"] for row in requests}
    clean_denominator = request_medians[oracle["best_static_backend"]]
    oracle["request_oracle_gain_pct_optimistic"] = oracle["oracle_saved_ms"] / clean_denominator * 100
    oracle["clean_request_medians_ms"] = request_medians
    oracle["clean_request_ht_vs_naive_pct"] = (
        1 - request_medians["deepep_high_throughput"] / request_medians["naive"]) * 100
    oracle["clean_output_hash_count"] = len(output_hashes)
    oracle["scaling_max_output_sum_backend_delta"] = scaling_output_delta

    cacheoff_trace = json.loads((root / "trajectories/gen64_cacheoff_trace/trajectory_gen64_b1_trace_trace_rank0.json").read_text())
    oracle["observer_tax_pct_rough"] = (
        cacheoff_trace["request_ms"] / request_medians["naive"] - 1) * 100
    batch4 = json.loads((root / "trajectories/gen64_batch4_cacheoff_trace/trajectory_gen64_batch4_cacheoff_trace_rank0.json").read_text())
    oracle["batch4_trace"] = {
        "nfe": batch4["nfe"],
        "request_ms_observer_heavy": batch4["request_ms"],
        "masked_range": [min(row["masked_before"] for row in batch4["iterations"]),
                         max(row["masked_before"] for row in batch4["iterations"])],
        "physical_model_positions": sorted({row.get("model_positions") for row in batch4["iterations"]}),
    }
    (analysis / "summary.json").write_text(json.dumps(oracle, indent=2) + "\n", encoding="utf-8")
    make_plots(analysis, shapes, m_rows, paired, structure, oracle, requests)
    print(json.dumps(oracle, indent=2))


if __name__ == "__main__":
    main()
