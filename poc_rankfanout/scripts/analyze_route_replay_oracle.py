#!/usr/bin/env python3
"""Analyze strict real-route replays and Amdahl-adjusted clean TTFT oracles."""

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
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def clean_ttft(root: Path, restart: int, backend: str, workload: str) -> float:
    short = "agrs" if backend == "agrs" else "deepep"
    payload = json.loads((root / "real" / f"clean_r{restart}_{short}" / "summary.json").read_text())
    values = []
    for row in payload["rows"]:
        if row["warmup"] or row["workload_id"] != workload:
            continue
        values.extend(float(item["ttft_ms"]) for item in row["requests"])
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = []
    input_info = []
    for path in args.input:
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        restart = int(path.parent.name.split("_", 1)[0].removeprefix("restart"))
        backend = summary["backend"]
        rows = read_jsonl(path)
        raw.extend({**row, "restart": restart, "backend": backend} for row in rows)
        input_info.append((path, restart, backend))

    grouped: dict[tuple[int, str, int, str], list[dict]] = defaultdict(list)
    for row in raw:
        grouped[(row["restart"], row["workload_id"], row["layer_id"], row["backend"])].append(row)
    cells = []
    for (restart, workload, layer, backend), values in sorted(grouped.items()):
        cell = {
            "restart": restart, "workload_id": workload, "layer_id": layer,
            "backend": backend, "n": len(values),
            "M_per_source_rank": values[0]["M_per_source_rank"],
            "route_hash": values[0]["route_hash"],
            "mean_fanout": values[0]["mean_fanout"],
            "frac_full_fanout": values[0]["frac_full_fanout"],
            "rank_load_max_over_mean": values[0]["rank_load_max_over_mean"],
            "expert_load_max_over_mean": values[0]["expert_load_max_over_mean"],
        }
        for stage in ("dispatch", "expert", "combine", "moe_total"):
            data = np.asarray([float(row[f"{stage}_ms"]) for row in values])
            cell[f"{stage}_p50_ms"] = float(np.median(data))
            cell[f"{stage}_p90_ms"] = float(np.quantile(data, .9))
        cells.append(cell)
    write_csv(args.output_dir / "replay_cells.csv", cells)

    paired = defaultdict(dict)
    for row in cells:
        paired[(row["restart"], row["workload_id"], row["layer_id"])][row["backend"]] = row
    pair_rows = []
    for key, values in sorted(paired.items()):
        if set(values) != {"agrs", "deepep_ht"}:
            continue
        a, d = values["agrs"], values["deepep_ht"]
        if a["route_hash"] != d["route_hash"]:
            raise RuntimeError(f"strict replay route mismatch {key}")
        row = {
            "restart": key[0], "workload_id": key[1], "layer_id": key[2],
            "M_per_source_rank": a["M_per_source_rank"], "route_hash": a["route_hash"],
            "mean_fanout": a["mean_fanout"], "frac_full_fanout": a["frac_full_fanout"],
            "rank_load_max_over_mean": a["rank_load_max_over_mean"],
            "expert_load_max_over_mean": a["expert_load_max_over_mean"],
        }
        for stage in ("dispatch", "expert", "combine", "moe_total"):
            av, dv = float(a[f"{stage}_p50_ms"]), float(d[f"{stage}_p50_ms"])
            row[f"agrs_{stage}_ms"] = av; row[f"deepep_{stage}_ms"] = dv
            row[f"{stage}_deepep_minus_agrs_ms"] = dv - av
        row["winner"] = "agrs" if row["moe_total_deepep_minus_agrs_ms"] > 0 else "deepep_ht"
        pair_rows.append(row)
    write_csv(args.output_dir / "strict_aligned_pairs.csv", pair_rows)

    oracle_rows = []
    def stage_cost(row: dict, backend: str, stage: str = "moe_total") -> float:
        prefix = "agrs" if backend == "agrs" else "deepep"
        return float(row[f"{prefix}_{stage}_ms"])

    for restart in sorted({row["restart"] for row in pair_rows}):
        for workload in sorted({row["workload_id"] for row in pair_rows if row["restart"] == restart}):
            selected = [row for row in pair_rows if row["restart"] == restart and row["workload_id"] == workload]
            totals = {backend: sum(stage_cost(row, backend) for row in selected) for backend in ("agrs", "deepep_ht")}
            best_backend = min(totals, key=totals.get); best = totals[best_backend]
            layer_oracle = sum(min(stage_cost(row, "agrs"), stage_cost(row, "deepep_ht")) for row in selected)
            ttft = clean_ttft(args.result_root, restart, best_backend, workload)
            saved = best - layer_oracle
            oracle_rows.append({
                "restart": restart, "workload_id": workload,
                "best_static_backend": best_backend,
                "agrs_replay_moe_ms": totals["agrs"], "deepep_replay_moe_ms": totals["deepep_ht"],
                "best_static_replay_moe_ms": best, "per_layer_oracle_moe_ms": layer_oracle,
                "per_layer_oracle_improvement_pct": 100 * saved / best,
                # One backend per serving step/request; each workload is one
                # prefill step, so O2 and O3 equal its best static backend.
                "per_step_oracle_moe_ms": best, "per_step_oracle_improvement_pct": 0.0,
                "per_request_oracle_moe_ms": best, "per_request_oracle_improvement_pct": 0.0,
                "best_static_clean_ttft_ms": ttft,
                "amdahl_saved_ms": saved,
                "projected_ttft_oracle_improvement_pct": 100 * saved / ttft,
                "agrs_winner_layers": sum(row["winner"] == "agrs" for row in selected),
                "deepep_winner_layers": sum(row["winner"] == "deepep_ht" for row in selected),
            })
    write_csv(args.output_dir / "oracle_by_workload_restart.csv", oracle_rows)

    # Spec-defined oracle granularities use one best-static backend over the
    # whole measured matrix.  A workload here is one synchronized prefill
    # step (two DP requests), so O2 and O3 collapse to the same feasible
    # choice granularity in this workload design.
    global_oracle_rows = []
    for restart in sorted({row["restart"] for row in pair_rows}):
        selected = [row for row in pair_rows if row["restart"] == restart]
        totals = {
            backend: sum(stage_cost(row, backend) for row in selected)
            for backend in ("agrs", "deepep_ht")
        }
        best_backend = min(totals, key=totals.get)
        best_static = totals[best_backend]
        per_layer = sum(
            min(stage_cost(row, "agrs"), stage_cost(row, "deepep_ht"))
            for row in selected
        )
        per_step = 0.0
        for workload in sorted({row["workload_id"] for row in selected}):
            step = [row for row in selected if row["workload_id"] == workload]
            per_step += min(
                sum(stage_cost(row, "agrs") for row in step),
                sum(stage_cost(row, "deepep_ht") for row in step),
            )
        clean_ttft_total = sum(
            clean_ttft(args.result_root, restart, best_backend, workload)
            for workload in sorted({row["workload_id"] for row in selected})
        )
        global_oracle_rows.append({
            "restart": restart,
            "best_static_backend": best_backend,
            "agrs_moe_ms": totals["agrs"],
            "deepep_moe_ms": totals["deepep_ht"],
            "best_static_moe_ms": best_static,
            "o1_per_layer_moe_ms": per_layer,
            "o1_improvement_pct": 100 * (best_static - per_layer) / best_static,
            "o2_per_step_moe_ms": per_step,
            "o2_improvement_pct": 100 * (best_static - per_step) / best_static,
            "o3_per_request_moe_ms": per_step,
            "o3_improvement_pct": 100 * (best_static - per_step) / best_static,
            "best_static_clean_ttft_sum_ms": clean_ttft_total,
            "o1_amdahl_ttft_improvement_pct": 100 * (best_static - per_layer) / clean_ttft_total,
            "o2_amdahl_ttft_improvement_pct": 100 * (best_static - per_step) / clean_ttft_total,
            "o3_amdahl_ttft_improvement_pct": 100 * (best_static - per_step) / clean_ttft_total,
        })
    write_csv(args.output_dir / "global_oracle_by_restart.csv", global_oracle_rows)

    # Held-out one-dimensional threshold, direction fixed by the hypothesis.
    all_restarts = sorted({int(row["restart"]) for row in pair_rows})
    # Restart 1 exposed a separately documented cold communication-path
    # confound.  Calibrate on the first stabilized restart and evaluate all
    # later restarts; never train a selector on the artifact.
    train_restart = 2 if 2 in all_restarts else all_restarts[0]
    test_restarts = [restart for restart in all_restarts if restart > train_restart]
    train = [row for row in pair_rows if row["restart"] == train_restart]
    test = [row for row in pair_rows if row["restart"] in test_restarts]
    candidates = []
    for threshold in sorted({round(float(row["mean_fanout"]), 6) for row in train}):
        cost = sum(stage_cost(row, "agrs" if row["mean_fanout"] >= threshold else "deepep_ht") for row in train)
        candidates.append((cost, threshold))
    _, threshold = min(candidates)
    totals = {backend: sum(stage_cost(row, backend) for row in test) for backend in ("agrs", "deepep_ht")}
    best = min(totals.values())
    perfect = sum(min(stage_cost(row, "agrs"), stage_cost(row, "deepep_ht")) for row in test)
    policy = sum(stage_cost(row, "agrs" if row["mean_fanout"] >= threshold else "deepep_ht") for row in test)
    threshold_result = {
        "train_restart": train_restart, "test_restarts": test_restarts, "threshold": threshold,
        "best_static_ms": best, "perfect_layer_oracle_ms": perfect,
        "fanout_threshold_ms": policy,
        "perfect_improvement_pct": 100 * (best - perfect) / best,
        "threshold_improvement_pct": 100 * (best - policy) / best,
        "oracle_recovery_pct": 100 * (best - policy) / (best - perfect) if best > perfect else 0.0,
    }
    (args.output_dir / "heldout_fanout_threshold.json").write_text(json.dumps(threshold_result, indent=2) + "\n")

    correctness = []
    by_restart = defaultdict(dict)
    for path, restart, backend in input_info:
        with np.load(path.with_suffix(".samples.npz")) as payload:
            by_restart[restart][backend] = {key: payload[key] for key in payload.files}
    for restart, values in sorted(by_restart.items()):
        for key in sorted(set(values["agrs"]) & set(values["deepep_ht"])):
            a = values["agrs"][key].astype(np.float64).ravel(); d = values["deepep_ht"][key].astype(np.float64).ravel()
            correctness.append({"restart": restart, "condition": key, "cosine": float(np.dot(a, d) / (np.linalg.norm(a) * np.linalg.norm(d))), "relative_l2": float(np.linalg.norm(a - d) / np.linalg.norm(a))})
    write_csv(args.output_dir / "correctness.csv", correctness)

    figures = args.output_dir / "figures"; figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for workload in sorted({row["workload_id"] for row in pair_rows}):
        selected = [row for row in pair_rows if row["workload_id"] == workload]
        ax.scatter([row["mean_fanout"] for row in selected], [row["moe_total_deepep_minus_agrs_ms"] for row in selected], alpha=.45, label=workload)
    ax.axhline(0, color="black", linewidth=1); ax.set_xlabel("mean destination-rank fanout"); ax.set_ylabel("DeepEP HT - AGRS exact-replay MoE (ms)"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figures / "B6_backend_delta_vs_real_fanout.png", dpi=180); plt.close(fig)

    total_a = sum(row["agrs_moe_total_ms"] for row in test)
    total_d = sum(row["deepep_moe_total_ms"] for row in test)
    labels = ["best static", "perfect layer", "fanout threshold"]
    values = [min(total_a, total_d), perfect, policy]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(labels, values); ax.set_ylabel("held-out exact-replay MoE total (ms)"); fig.tight_layout(); fig.savefig(figures / "B7_static_oracle_threshold.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); labels = [f"r{r['restart']}:{r['workload_id']}" for r in oracle_rows]; x = np.arange(len(labels)); ax.bar(x, [r["per_layer_oracle_improvement_pct"] for r in oracle_rows], label="MoE oracle"); ax.bar(x, [r["projected_ttft_oracle_improvement_pct"] for r in oracle_rows], label="Amdahl TTFT"); ax.set_xticks(x, labels=labels, rotation=45, ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "B8_moe_vs_ttft_oracle.png", dpi=180); plt.close(fig)

    stable_oracles = [row for row in oracle_rows if row["restart"] >= train_restart]
    stable_global = [row for row in global_oracle_rows if row["restart"] >= train_restart]
    summary = {
        "raw_rows": len(raw), "aligned_layer_pairs": len(pair_rows),
        "strict_route_hashes_equal": True,
        "min_cosine": min(row["cosine"] for row in correctness),
        "max_relative_l2": max(row["relative_l2"] for row in correctness),
        "threshold": threshold_result,
        "stabilized_restart_range": [train_restart, max(all_restarts)],
        "stabilized_projected_ttft_oracle_median_pct": float(np.median([row["projected_ttft_oracle_improvement_pct"] for row in stable_oracles])),
        "stabilized_projected_ttft_oracle_max_pct": max(row["projected_ttft_oracle_improvement_pct"] for row in stable_oracles),
        "stabilized_moe_oracle_median_pct": float(np.median([row["per_layer_oracle_improvement_pct"] for row in stable_oracles])),
        "spec_granularity_global_oracles": global_oracle_rows,
        "stabilized_global_o1_moe_median_pct": float(np.median([row["o1_improvement_pct"] for row in stable_global])),
        "stabilized_global_o1_moe_max_pct": max(row["o1_improvement_pct"] for row in stable_global),
        "stabilized_global_o2_moe_median_pct": float(np.median([row["o2_improvement_pct"] for row in stable_global])),
        "stabilized_global_o2_moe_max_pct": max(row["o2_improvement_pct"] for row in stable_global),
        "stabilized_global_o1_ttft_median_pct": float(np.median([row["o1_amdahl_ttft_improvement_pct"] for row in stable_global])),
        "stabilized_global_o1_ttft_max_pct": max(row["o1_amdahl_ttft_improvement_pct"] for row in stable_global),
        "stabilized_global_o2_ttft_median_pct": float(np.median([row["o2_amdahl_ttft_improvement_pct"] for row in stable_global])),
        "stabilized_global_o2_ttft_max_pct": max(row["o2_amdahl_ttft_improvement_pct"] for row in stable_global),
        "oracle_rows": oracle_rows,
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
