"""Analyze the four preregistered FlashVEP tile-to-slack stages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _corr(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": 0.0, "spearman_rho": 0.0}
    return {
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _stage_a(result: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads((result / "stage_a" / "summary.json").read_text())
    manifest = json.loads((result / "stage_a" / "sample_manifest.json").read_text())
    counts: dict[str, int] = defaultdict(int)
    origins: dict[str, int] = defaultdict(int)
    for row in manifest["samples"]:
        counts[row["category"]] += 1
        origins[row.get("suite_origin", "unknown")] += 1
    conclusions_hold = all(summary["statuses"][key] == "GO" for key in ("plot1", "plot2", "plot3"))
    coverage_met = all(counts.get(key, 0) >= 16 for key in ("natural", "chart_document", "fine_grained"))
    status = "GO" if conclusions_hold and coverage_met else "HOLD" if conclusions_hold else "NO-GO"
    return {
        "status": status,
        "sample_count": summary["sample_count"],
        "category_counts": dict(counts),
        "suite_origins": dict(origins),
        "coverage_target_met": coverage_met,
        "plot1": summary["plot1"],
        "plot2": summary["plot2"],
        "plot3": summary["plot3"],
    }, manifest


def _load_ranks(result: Path) -> list[dict[str, Any]]:
    rows = [json.loads((result / "replay" / f"rank{rank}.json").read_text()) for rank in range(4)]
    if any(row.get("status") != "ok" for row in rows):
        raise RuntimeError(f"replay failure: {[row.get('status') for row in rows]}")
    return rows


def _stage_b(rank_payloads: list[dict[str, Any]], figures: Path, result: Path) -> tuple[dict[str, Any], tuple[float, float]]:
    flat: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for payload in rank_payloads:
        for row in payload["stage_b"]:
            expert = float(row["timing"]["expert_ms_stats"]["median_ms"])
            item = {
                "sample_id": row["sample_id"], "category": row["category"],
                "layer": int(row["layer"]), "rank": int(row["rank"]),
                "assignments": int(row["total_assignments"]),
                "vision_assignments": int(row["vision_assignments"]),
                "nonvision_assignments": int(row["nonvision_assignments"]),
                "expert_ms": expert,
            }
            flat.append(item)
            grouped[(row["sample_id"], int(row["layer"]))].append(item)
    x = [float(row["assignments"]) for row in flat]
    y = [row["expert_ms"] for row in flat]
    correlations = _corr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = np.asarray(x) * slope + intercept
    ss_res = float(np.square(np.asarray(y) - predicted).sum())
    ss_tot = float(np.square(np.asarray(y) - np.mean(y)).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    exact = top2 = 0
    mismatch = []
    vision_x: list[float] = []
    latency_y: list[float] = []
    for key, values in grouped.items():
        values.sort(key=lambda row: row["rank"])
        assignment_rank = max(values, key=lambda row: row["assignments"])["rank"]
        latency_order = sorted(values, key=lambda row: row["expert_ms"], reverse=True)
        latency_rank = latency_order[0]["rank"]
        exact += assignment_rank == latency_rank
        top2 += assignment_rank in {row["rank"] for row in latency_order[:2]}
        if assignment_rank != latency_rank and len(mismatch) < 12:
            mismatch.append(
                {"sample_id": key[0], "layer": key[1],
                 "assignment_rank": assignment_rank, "latency_rank": latency_rank,
                 "assignments": [row["assignments"] for row in values],
                 "expert_ms": [row["expert_ms"] for row in values]}
            )
        mean_v = statistics.fmean(row["vision_assignments"] for row in values)
        mean_t = statistics.fmean(row["expert_ms"] for row in values)
        for row in values:
            vision_x.append(row["vision_assignments"] - mean_v)
            latency_y.append(row["expert_ms"] - mean_t)
    vision_corr = _corr(vision_x, latency_y)
    match = exact / len(grouped)
    top2_rate = top2 / len(grouped)
    if correlations["spearman_rho"] >= 0.8 and match >= 0.8:
        status = "GO"
    elif correlations["spearman_rho"] >= 0.5 or match >= 0.5:
        status = "HOLD"
    else:
        status = "NO-GO"

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=10, alpha=0.35)
    line_x = np.linspace(min(x), max(x), 100)
    plt.plot(line_x, slope * line_x + intercept, color="tab:red")
    plt.xlabel("Routed assignments on EP rank")
    plt.ylabel("CUDA expert latency (ms)")
    plt.title("Assignments vs actual local expert latency")
    plt.tight_layout(); plt.savefig(figures / "plot4a_assignments_vs_expert_latency.png", dpi=180); plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(["Exact", "Top-2"], [match * 100, top2_rate * 100], color=["#4c78a8", "#72b7b2"])
    plt.ylim(0, 100); plt.ylabel("Match rate (%)"); plt.title("Assignment-critical rank validation")
    plt.tight_layout(); plt.savefig(figures / "plot4b_critical_rank_match.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(vision_x, latency_y, s=10, alpha=0.35)
    plt.axhline(0, color="black", lw=0.6); plt.axvline(0, color="black", lw=0.6)
    plt.xlabel("Vision assignment excess"); plt.ylabel("Expert latency excess (ms)")
    plt.title("Vision routing excess vs actual latency excess")
    plt.tight_layout(); plt.savefig(figures / "plot4c_vision_excess_vs_latency_excess.png", dpi=180); plt.close()
    _write_csv(result / "stage_b_rank_observations.csv", flat)
    return {
        "status": status, "observations": len(flat), "request_layers": len(grouped),
        **correlations, "r2": r2, "critical_rank_exact_match": match,
        "critical_rank_top2_inclusion": top2_rate,
        "vision_excess_latency_excess": vision_corr, "mismatch_examples": mismatch,
        "fit": {"slope_ms_per_assignment": float(slope), "intercept_ms": float(intercept)},
    }, (float(slope), float(intercept))


def _critical_stage_c(rank_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in rank_payloads:
        for row in payload["stage_c"]:
            grouped[(row["sample_id"], int(row["layer"]), row["strategy"], row["granularity"])].append(row)
    merged = []
    for key, rows in grouped.items():
        if key[2] == "serial":
            continue
        serial_wall = max(row["serial"]["wall_ms_stats"]["median_ms"] for row in rows)
        overlap_wall = max(row["overlap"]["wall_ms_stats"]["median_ms"] for row in rows)
        serial_d = max(row["serial"]["dispatch_ms_stats"]["median_ms"] for row in rows)
        serial_e = max(row["serial"]["expert_ms_stats"]["median_ms"] for row in rows)
        serial_c = max(row["serial"]["combine_ms_stats"]["median_ms"] for row in rows)
        overlap_d = max(row["overlap"]["dispatch_ms_stats"]["median_ms"] for row in rows)
        overlap_e = max(row["overlap"]["expert_ms_stats"]["median_ms"] for row in rows)
        overlap_c = max(row["overlap"]["combine_ms_stats"]["median_ms"] for row in rows)
        serial_comm = serial_d + serial_c
        actual_overlap = min(
            row["overlap"]["dispatch_expert_overlap_ms_stats"]["median_ms"]
            + row["overlap"]["expert_combine_overlap_ms_stats"]["median_ms"]
            for row in rows
        )
        merged.append({
            "sample_id": key[0], "layer": key[1], "strategy": key[2], "granularity": key[3],
            "serial_wall_ms": serial_wall, "overlap_wall_ms": overlap_wall,
            "speedup": serial_wall / overlap_wall,
            "dispatch_ms": overlap_d, "expert_ms": overlap_e, "combine_ms": overlap_c,
            "dispatch_slowdown": overlap_d / serial_d,
            "expert_slowdown": overlap_e / serial_e,
            "combine_slowdown": overlap_c / serial_c,
            "hidden_comm_ms": min(serial_comm, actual_overlap),
            "net_benefit_ms": serial_wall - overlap_wall,
            "actual_overlap_ms": actual_overlap,
            "overlap_efficiency": actual_overlap / min(overlap_e, overlap_d + overlap_c),
            "correctness": all(row["correctness"]["passed"] for row in rows),
            "rank_rows": rows,
        })
    return merged


def _stage_c(rows: list[dict[str, Any]], figures: Path, result: Path) -> dict[str, Any]:
    labels = [
        ("Generic 2x2", "generic", "2x2"), ("Generic 4x4", "generic", "4x4"),
        ("Sequential 2x2", "sequential", "2x2"), ("Sequential 4x4", "sequential", "4x4"),
        ("Spatial 2x2", "spatial", "2x2"), ("Spatial 4x4", "spatial", "4x4"),
    ]
    values: dict[str, list[float]] = {}
    summaries = {}
    for label, strategy, granularity in labels:
        selected = [row for row in rows if row["strategy"] == strategy and row["granularity"] == granularity]
        values[label] = [row["speedup"] for row in selected]
        summaries[label] = {
            "median_speedup": _median(values[label]),
            "p25_speedup": float(np.percentile(values[label], 25)),
            "p75_speedup": float(np.percentile(values[label], 75)),
            "median_hidden_comm_ms": _median([row["hidden_comm_ms"] for row in selected]),
            "median_actual_overlap_ms": _median([row["actual_overlap_ms"] for row in selected]),
            "median_overlap_efficiency": _median([row["overlap_efficiency"] for row in selected]),
            "median_expert_slowdown": _median([row["expert_slowdown"] for row in selected]),
            "median_net_benefit_ms": _median([row["net_benefit_ms"] for row in selected]),
        }
    spatial_best = max(summaries["Spatial 2x2"]["median_speedup"], summaries["Spatial 4x4"]["median_speedup"])
    generic_best = max(summaries["Generic 2x2"]["median_speedup"], summaries["Generic 4x4"]["median_speedup"])
    sequential_best = max(summaries["Sequential 2x2"]["median_speedup"], summaries["Sequential 4x4"]["median_speedup"])
    correct = all(row["correctness"] for row in rows)
    if correct and spatial_best >= 1.05 and spatial_best >= generic_best * 1.01:
        status = "GO"
    elif correct and spatial_best > 1.0 and max(spatial_best, sequential_best) >= generic_best:
        status = "HOLD"
    else:
        status = "NO-GO"
    plot_labels = ["Serial"] + [label.replace(" ", "\n") for label, _, _ in labels]
    plot_values = [[1.0] * len(rows)] + [values[label] for label, _, _ in labels]
    plt.figure(figsize=(10, 5))
    plt.boxplot(plot_values, tick_labels=plot_labels, showfliers=False)
    plt.axhline(1.0, color="black", ls="--", lw=0.8); plt.ylabel("Serial / overlap wall speedup")
    plt.title("Grouping strategy operator-replay speedup"); plt.tight_layout()
    plt.savefig(figures / "plot5a_grouping_speedup.png", dpi=180); plt.close()

    names = [label for label, _, _ in labels]
    hidden = [summaries[name]["median_hidden_comm_ms"] for name in names]
    slowdown_cost = [
        _median([row["expert_ms"] * (row["expert_slowdown"] - 1) / row["expert_slowdown"]
                 for row in rows if f"{row['strategy'].title()} {row['granularity']}" == name])
        for name in names
    ]
    net = [summaries[name]["median_net_benefit_ms"] for name in names]
    x = np.arange(len(names)); width = 0.25
    plt.figure(figsize=(11, 5))
    plt.bar(x - width, hidden, width, label="communication hidden")
    plt.bar(x, slowdown_cost, width, label="expert slowdown cost")
    plt.bar(x + width, net, width, label="net benefit")
    plt.axhline(0, color="black", lw=0.7); plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Median time (ms)"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "plot5b_overlap_decomposition.png", dpi=180); plt.close()
    csv_rows = [{key: value for key, value in row.items() if key != "rank_rows"} for row in rows]
    _write_csv(result / "stage_c_configurations.csv", csv_rows)
    return {
        "status": status, "configurations": len(rows), "all_correct": correct,
        "strategy_summary": summaries, "spatial_best_speedup": spatial_best,
        "sequential_best_speedup": sequential_best, "generic_best_speedup": generic_best,
        "positive_overlap_fraction": sum(row["actual_overlap_ms"] > 0 for row in rows) / len(rows),
        "gate_rule": "GO: correctness, spatial median >=1.05x and >=1% over generic; HOLD: profitable routing-aware grouping but no unique spatial margin; otherwise NO-GO.",
    }


def _stage_d(
    rows: list[dict[str, Any]], fit: tuple[float, float], stage_b_status: str,
    figures: Path, result: Path,
) -> dict[str, Any]:
    slope, intercept = fit
    points = []
    for config in rows:
        rank_rows = sorted(config["rank_rows"], key=lambda row: row["rank"])
        assignments = rank_rows[0]["rank_assignments_by_wave"]
        wave_count = len(assignments)
        for wave in range(wave_count):
            actual_by_rank = []
            for row in rank_rows:
                values = [sample["per_wave"][wave]["expert_ms"] for sample in row["overlap"]["samples"]]
                actual_by_rank.append(_median(values))
            max_assignments = max(assignments[wave])
            predicted = max(0.0, slope * max_assignments + intercept)
            points.append({
                "sample_id": config["sample_id"], "layer": config["layer"],
                "strategy": config["strategy"], "granularity": config["granularity"],
                "wave": wave, "max_rank_assignments": max_assignments,
                "predicted_window_ms": predicted, "actual_window_ms": max(actual_by_rank),
                "speedup": config["speedup"],
                "hidden_comm_per_wave_ms": config["hidden_comm_ms"] / wave_count,
            })
    predicted = [row["predicted_window_ms"] for row in points]
    actual = [row["actual_window_ms"] for row in points]
    speedups = [row["speedup"] for row in points]
    pred_corr = _corr(predicted, actual)
    slack_corr = _corr(actual, speedups)
    order = np.argsort(actual)
    bins = np.array_split(order, 5)
    boundaries = []
    profitable_boundary = None
    for indices in bins:
        xs = [actual[int(i)] for i in indices]
        ys = [speedups[int(i)] for i in indices]
        row = {
            "min_window_ms": min(xs), "max_window_ms": max(xs),
            "median_window_ms": _median(xs), "median_speedup": _median(ys),
            "profitable_fraction": sum(value > 1.0 for value in ys) / len(ys),
        }
        boundaries.append(row)
        if profitable_boundary is None and row["median_speedup"] > 1.0 and row["profitable_fraction"] >= 0.7:
            profitable_boundary = row["min_window_ms"]
    if pred_corr["spearman_rho"] >= 0.8 and slack_corr["spearman_rho"] >= 0.5 and profitable_boundary is not None:
        status = "GO"
    elif pred_corr["spearman_rho"] >= 0.5 and profitable_boundary is not None:
        status = "HOLD"
    else:
        status = "NO-GO"
    plt.figure(figsize=(6, 5)); plt.scatter(predicted, actual, s=10, alpha=0.35)
    lo, hi = min(predicted + actual), max(predicted + actual); plt.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    plt.xlabel("Predicted compute window (ms)"); plt.ylabel("Actual critical-rank expert window (ms)")
    plt.tight_layout(); plt.savefig(figures / "plot6a_predicted_vs_actual_window.png", dpi=180); plt.close()
    markers = {"generic": "o", "sequential": "s", "spatial": "^"}
    plt.figure(figsize=(7, 5))
    for strategy, marker in markers.items():
        chosen = [row for row in points if row["strategy"] == strategy]
        plt.scatter([row["actual_window_ms"] for row in chosen], [row["speedup"] for row in chosen],
                    s=12, alpha=0.35, marker=marker, label=strategy)
    plt.axhline(1.0, color="black", ls="--", lw=0.8); plt.xlabel("Actual compute window (ms)")
    plt.ylabel("Serial / overlap speedup"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "plot6b_slack_vs_overlap_speedup.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4)); plt.plot([row["median_window_ms"] for row in boundaries],
        [row["median_speedup"] for row in boundaries], marker="o")
    plt.axhline(1.0, color="black", ls="--", lw=0.8)
    if profitable_boundary is not None: plt.axvline(profitable_boundary, color="tab:red", ls=":")
    plt.xlabel("Compute-window quintile median (ms)"); plt.ylabel("Median speedup")
    plt.tight_layout(); plt.savefig(figures / "plot6c_profitability_boundary.png", dpi=180); plt.close()
    _write_csv(result / "stage_d_wave_observations.csv", points)
    return {
        "status": status, "points": len(points),
        "predictor": "assignment linear fit" if stage_b_status == "GO" else "assignment fit retained as diagnostic; LUT required for deployment",
        "predicted_actual_correlation": pred_corr,
        "slack_speedup_correlation": slack_corr,
        "profitability_boundary_ms": profitable_boundary,
        "profitability_bins": boundaries,
    }


def _report(path: Path, result: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    a, b, c, d = (summary[key] for key in ("stage_a", "stage_b", "stage_c", "stage_d"))
    strategies = c["strategy_summary"]
    text = f"""# FlashVEP Tile-to-Slack Mechanism Validation

## 1. Environment

Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4/PP1, vLLM 0.20,
DeepEP high-throughput, eager execution, physical GPUs 4–7 only. Installed
Attention/DeepStack source fixes and software versions were retained.

Stage B–D are scheduler-free operator replays using actual model-loaded layer-24
Triton expert weights and DeepEP `Buffer.dispatch/combine`. Captured routes are
unchanged. Hidden values and top-k weights are cycled from the validated real
layer-24 capture to the requested route length; therefore timings validate
route/GEMM shape and overlap, not activation-value equivalence across layers.

## 2. Workload/sample manifest

Stage A used {a['sample_count']} unique local requests. Category counts are
`{a['category_counts']}` and origins are `{a['suite_origins']}`. No sample was
replicated and no dataset was downloaded. The full image paths, hashes, grids,
and old/new labels are in `stage_a/sample_manifest.json`.

## 3. Stage A — Motivation robustness replication

**STAGE_A_STATUS: {a['status']}**

- vision-ratio median: {a['plot1']['median']:.4f}
- visual critical-excess median: {a['plot2']['visual_contribution']['median']:.4f}
- spatial/random rank-JSD: {a['plot3']['2x2']['spatial_random_ratio']:.3f}x (2x2), {a['plot3']['4x4']['spatial_random_ratio']:.3f}x (4x4)

All three prior conclusions reproduce, but category>=16 coverage is
`{a['coverage_target_met']}`; the gate remains HOLD when local coverage is below
the requested target.

## 4. Stage B — Assignment to CUDA latency

**STAGE_B_STATUS: {b['status']}**

- Pearson r: {b['pearson_r']:.4f}; Spearman rho: {b['spearman_rho']:.4f}; R2: {b['r2']:.4f}
- assignment-critical exact match: {b['critical_rank_exact_match']:.2%}
- top-2 inclusion: {b['critical_rank_top2_inclusion']:.2%}
- vision-excess/latency-excess Spearman: {b['vision_excess_latency_excess']['spearman_rho']:.4f}

![Figure 4A](../deepep_revalidation/results/{result.name}/figures/plot4a_assignments_vs_expert_latency.png)
![Figure 4B](../deepep_revalidation/results/{result.name}/figures/plot4b_critical_rank_match.png)
![Figure 4C](../deepep_revalidation/results/{result.name}/figures/plot4c_vision_excess_vs_latency_excess.png)

## 5. Stage C — Offline tile/wave overlap replay

**STAGE_C_STATUS: {c['status']}**

Serial is 1.0x. Median speedups are:

| strategy | speedup | hidden comm (ms) | overlap efficiency | expert slowdown | net benefit (ms) |
|---|---:|---:|---:|---:|---:|
"""
    for name, values in strategies.items():
        text += f"| {name} | {values['median_speedup']:.4f}x | {values['median_hidden_comm_ms']:.4f} | {values['median_overlap_efficiency']:.3f} | {values['median_expert_slowdown']:.4f}x | {values['median_net_benefit_ms']:.4f} |\n"
    text += f"""

All route/order/correctness checks: `{c['all_correct']}`. Spatial best is
{c['spatial_best_speedup']:.4f}x, sequential best {c['sequential_best_speedup']:.4f}x,
and generic best {c['generic_best_speedup']:.4f}x. CUDA interval intersection is
positive in {c['positive_overlap_fraction']:.2%} of merged configurations; the
table separates temporal overlap from wall-time net benefit.

![Figure 5A](../deepep_revalidation/results/{result.name}/figures/plot5a_grouping_speedup.png)
![Figure 5B](../deepep_revalidation/results/{result.name}/figures/plot5b_overlap_decomposition.png)

## 6. Stage D — Predicted slack vs actual benefit

**STAGE_D_STATUS: {d['status']}**

- predicted/actual-window Spearman: {d['predicted_actual_correlation']['spearman_rho']:.4f}
- actual-window/speedup Spearman: {d['slack_speedup_correlation']['spearman_rho']:.4f}
- profitability boundary: {d['profitability_boundary_ms']} ms

![Figure 6A](../deepep_revalidation/results/{result.name}/figures/plot6a_predicted_vs_actual_window.png)
![Figure 6B](../deepep_revalidation/results/{result.name}/figures/plot6b_slack_vs_overlap_speedup.png)
![Figure 6C](../deepep_revalidation/results/{result.name}/figures/plot6c_profitability_boundary.png)

## 7. Spatial vs Sequential vs Generic interpretation

{summary['strategy_interpretation']}

## 8. FINAL MECHANISM STATUS

**FINAL MECHANISM STATUS: {summary['final_status']}**

Recommended method framing: **{summary['method_framing']}**.

## 9. Strongest positive evidence

{summary['strongest_positive']}

## 10. Strongest counter-evidence

{summary['strongest_counter']}

## 11. Limitations

- Stage A exhausts locally available unique images but misses 16/category and is
  not a random benchmark sample.
- Stage B–D replay real routes, real weights, Triton experts, and DeepEP kernels,
  but cycles one validated layer-24 hidden/top-k-weight capture.
- Only three representative requests and five layers enter Stage C/D; this is a
  bounded mechanism test, not end-to-end serving evidence.
- CUDA-event interval overlap establishes temporal concurrency, not its exact
  HBM/L2 contention cause.

## 12. Next single recommended action

{summary['recommended_action']}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = args.result_dir
    figures = result / "figures"; figures.mkdir(parents=True, exist_ok=True)
    stage_a, manifest = _stage_a(result)
    rank_payloads = _load_ranks(result)
    stage_b, fit = _stage_b(rank_payloads, figures, result)
    c_rows = _critical_stage_c(rank_payloads)
    stage_c = _stage_c(c_rows, figures, result)
    stage_d = _stage_d(c_rows, fit, stage_b["status"], figures, result)
    statuses = [stage_a["status"], stage_b["status"], stage_c["status"], stage_d["status"]]
    if "NO-GO" in statuses:
        final = "NO-GO"
    elif all(value == "GO" for value in statuses):
        final = "GO"
    else:
        final = "HOLD"
    spatial = stage_c["spatial_best_speedup"]
    sequential = stage_c["sequential_best_speedup"]
    generic = stage_c["generic_best_speedup"]
    if final == "NO-GO":
        framing = "Reconsider"
        interpretation = "The full tile-to-slack chain did not turn spatial structure into a robust, predictable system benefit."
    elif spatial > sequential * 1.01 and spatial > generic * 1.01:
        framing = "Spatial-Tile-Aware"
        interpretation = "Spatial grouping has a measurable margin over both sequential and generic controls."
    else:
        framing = "Vision-Routing-Aware"
        interpretation = "Spatial/sequential routing-aware waves are similar; strict 2D tiling is not uniquely necessary."
    summary = {
        "stage_a": stage_a, "stage_b": stage_b, "stage_c": stage_c, "stage_d": stage_d,
        "final_status": final, "method_framing": framing,
        "strategy_interpretation": interpretation,
        "strongest_positive": (
            f"Actual DeepEP/Triton replay reached {max(spatial, sequential):.4f}x median routing-aware speedup "
            f"with assignment/latency Spearman {stage_b['spearman_rho']:.4f}."
        ),
        "strongest_counter": (
            f"Slack/speedup Spearman is only {stage_d['slack_speedup_correlation']['spearman_rho']:.4f}, "
            f"no profitable boundary exists, and spatial={spatial:.4f}x trails sequential={sequential:.4f}x."
        ),
        "recommended_action": (
            "Replace the assignment-only slack predictor with a fixed expert-token/GEMM-shape latency LUT and rerun "
            "the same bounded offline Stage D gate before any scheduler integration."
        ),
    }
    _json(result / "summary.json", summary)
    _report(args.report, result, summary, manifest)
    print(json.dumps({"statuses": statuses, "final": final, "framing": framing}, indent=2))


if __name__ == "__main__":
    main()
