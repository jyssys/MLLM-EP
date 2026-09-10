#!/usr/bin/env python3
"""Summarize observer-free Qwen3-VL AGRS/DeepEP static serving runs.

The two backends require separate engine processes, so a restart is the
statistical unit.  This analyzer intentionally reports restart sensitivity
instead of pooling request rows into a deceptively precise estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    return payload["proof"], payload["rows"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request_rows: list[dict] = []
    proofs: list[dict] = []
    outputs: dict[tuple[int, str, int, str, str], dict[str, tuple[int, ...]]] = {}
    for restart in range(1, args.restarts + 1):
        for short, backend in (("agrs", "agrs"), ("deepep", "deepep_ht")):
            path = args.result_root / "real" / f"clean_r{restart}_{short}" / "summary.json"
            proof, rows = load_rows(path)
            proofs.append({"restart": restart, "backend": backend, **proof})
            for row in rows:
                if row["warmup"]:
                    continue
                for request in row["requests"]:
                    key = (
                        restart,
                        str(row["workload_id"]),
                        int(row["iteration"]),
                        str(request["request_id"]),
                        backend,
                    )
                    outputs[key] = backend, tuple(int(token) for token in request["output_token_ids"])
                    request_rows.append({
                        "restart": restart,
                        "backend": backend,
                        "workload_id": row["workload_id"],
                        "iteration": row["iteration"],
                        "request_id": request["request_id"],
                        "prompt_tokens": request["prompt_tokens"],
                        "output_tokens": request["output_tokens"],
                        "ttft_ms": request["ttft_ms"],
                        "e2e_ms": request["e2e_ms"],
                    })
    write_csv(args.output_dir / "clean_request_rows.csv", request_rows)

    grouped: dict[tuple[int, str, str], list[float]] = {}
    for row in request_rows:
        key = (int(row["restart"]), str(row["workload_id"]), str(row["backend"]))
        grouped.setdefault(key, []).append(float(row["ttft_ms"]))
    cell_rows: list[dict] = []
    for (restart, workload, backend), values in sorted(grouped.items()):
        cell_rows.append({
            "restart": restart,
            "workload_id": workload,
            "backend": backend,
            "n_requests": len(values),
            "ttft_p50_ms": statistics.median(values),
            "ttft_p90_ms": float(np.quantile(values, 0.9)),
            "ttft_mean_ms": statistics.mean(values),
        })
    write_csv(args.output_dir / "clean_static_cells.csv", cell_rows)

    cells = {(row["restart"], row["workload_id"], row["backend"]): row for row in cell_rows}
    workloads = sorted({str(row["workload_id"]) for row in cell_rows})
    pair_rows: list[dict] = []
    restart_rows: list[dict] = []
    for restart in range(1, args.restarts + 1):
        totals = {"agrs": 0.0, "deepep_ht": 0.0}
        per_workload_best = 0.0
        for workload in workloads:
            a = float(cells[(restart, workload, "agrs")]["ttft_p50_ms"])
            d = float(cells[(restart, workload, "deepep_ht")]["ttft_p50_ms"])
            totals["agrs"] += a
            totals["deepep_ht"] += d
            per_workload_best += min(a, d)
            pair_rows.append({
                "restart": restart,
                "workload_id": workload,
                "agrs_ttft_p50_ms": a,
                "deepep_ttft_p50_ms": d,
                "deepep_minus_agrs_ms": d - a,
                "winner": "agrs" if a < d else "deepep_ht",
                "winner_margin_pct": 100.0 * abs(d - a) / min(a, d),
            })
        best_static_backend = min(totals, key=totals.get)
        best_static = totals[best_static_backend]
        restart_rows.append({
            "restart": restart,
            "agrs_sum_workload_p50_ms": totals["agrs"],
            "deepep_sum_workload_p50_ms": totals["deepep_ht"],
            "best_static_backend": best_static_backend,
            "best_static_sum_ms": best_static,
            "descriptive_per_workload_oracle_ms": per_workload_best,
            "descriptive_per_workload_oracle_improvement_pct": 100.0 * (best_static - per_workload_best) / best_static,
        })
    write_csv(args.output_dir / "clean_static_pairs.csv", pair_rows)
    write_csv(args.output_dir / "clean_static_restart_summary.csv", restart_rows)

    correctness_rows: list[dict] = []
    for restart in range(1, args.restarts + 1):
        for workload in workloads:
            for iteration in (1, 2, 3):
                request_ids = sorted({
                    key[3] for key in outputs
                    if key[:3] == (restart, workload, iteration)
                })
                for request_id in request_ids:
                    a = outputs[(restart, workload, iteration, request_id, "agrs")][1]
                    d = outputs[(restart, workload, iteration, request_id, "deepep_ht")][1]
                    correctness_rows.append({
                        "restart": restart,
                        "workload_id": workload,
                        "iteration": iteration,
                        "request_id": request_id,
                        "exact_token_agreement": a == d,
                        "agrs_tokens": " ".join(map(str, a)),
                        "deepep_tokens": " ".join(map(str, d)),
                    })
    write_csv(args.output_dir / "clean_output_correctness.csv", correctness_rows)

    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    x = np.arange(args.restarts)
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [row["agrs_sum_workload_p50_ms"] for row in restart_rows], width, label="AGRS")
    ax.bar(x + width / 2, [row["deepep_sum_workload_p50_ms"] for row in restart_rows], width, label="DeepEP HT")
    ax.set_xticks(x, [f"r{restart}" for restart in range(1, args.restarts + 1)])
    ax.set_ylabel("sum of clean workload TTFT medians (ms)")
    ax.set_title("Static backend result is restart-state sensitive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "B5_clean_static_restart_sensitivity.png", dpi=180)
    plt.close(fig)

    summary = {
        "restarts": args.restarts,
        "workloads": workloads,
        "measured_request_rows": len(request_rows),
        "aligned_output_pairs": len(correctness_rows),
        "exact_output_agreement_fraction": statistics.mean(float(row["exact_token_agreement"]) for row in correctness_rows),
        "restart_best_static_backends": [row["best_static_backend"] for row in restart_rows],
        "agrs_best_restart_count": sum(row["best_static_backend"] == "agrs" for row in restart_rows),
        "deepep_best_restart_count": sum(row["best_static_backend"] == "deepep_ht" for row in restart_rows),
        "descriptive_per_workload_oracle_improvement_median_pct": statistics.median(
            float(row["descriptive_per_workload_oracle_improvement_pct"]) for row in restart_rows
        ),
        "descriptive_per_workload_oracle_improvement_max_pct": max(
            float(row["descriptive_per_workload_oracle_improvement_pct"]) for row in restart_rows
        ),
        "interpretation": (
            "CAUTION: backend processes cannot be interleaved inside one engine and clean TTFT exhibits "
            "large restart-state drift. These rows validate static serving and output equivalence, but "
            "the bit-exact same-route replay is the causal dynamic-oracle evidence."
        ),
        "restart_rows": restart_rows,
    }
    (args.output_dir / "clean_static_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
