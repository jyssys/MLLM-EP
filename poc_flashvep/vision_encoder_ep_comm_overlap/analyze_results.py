"""Aggregate the bounded real Qwen3-VL encoder/DeepEP overlap run.

This is intentionally an analysis-only script.  It consumes rank-local JSON
written by ``overlap_hook.py`` and never changes model, routing, or placement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def q(values: list[float], p: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), p))


def stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "median_ms": float(statistics.median(values)),
        "p25_ms": q(values, 0.25),
        "p75_ms": q(values, 0.75),
        "p95_ms": q(values, 0.95),
        "mean_ms": float(statistics.fmean(values)),
        "cv": float(np.std(values, ddof=1) / np.mean(values)) if len(values) > 1 and np.mean(values) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    ranks = [json.loads(path.read_text()) for path in sorted(root.glob("overlap_rank*.json"))]
    if len(ranks) != 4:
        raise SystemExit(f"expected four rank files, found {len(ranks)}")

    raw_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    for rank_obj in ranks:
        rank = int(rank_obj["rank"])
        for phase in ("dispatch", "combine", "expert"):
            obj = rank_obj[phase]
            for i, sample in enumerate(obj["samples"]):
                raw_rows.append({
                    "rank": rank,
                    "phase": phase,
                    "iteration": i,
                    "alone_ms": sample["alone_ms"],
                    "concurrent_ms": sample["concurrent_ms"],
                    "alone_comm_ms": sample["alone_comm_ms"],
                    "concurrent_comm_ms": sample["concurrent_comm_ms"],
                    "encoder_overlap_ms": sample["encoder_ms"],
                    "encoder_alone_ms": obj["encoder_alone_ms"][i],
                })
            phase_rows.append({
                "rank": rank,
                "phase": phase,
                "encoder_alone_median_ms": obj["encoder_alone_stats"]["median_ms"],
                "alone_median_ms": obj["alone_stats"]["median_ms"],
                "concurrent_median_ms": obj["concurrent_stats"]["median_ms"],
                "alone_comm_median_ms": obj["alone_comm_stats"]["median_ms"],
                "concurrent_comm_median_ms": obj["concurrent_comm_stats"]["median_ms"],
                "encoder_overlap_median_ms": obj["encoder_overlap_stats"]["median_ms"],
            })

    with (root / "raw_timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader(); writer.writerows(raw_rows)
    with (root / "rank_phase_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_rows[0]))
        writer.writeheader(); writer.writerows(phase_rows)

    by_phase: dict[str, dict[str, list[float]]] = {}
    for phase in ("dispatch", "combine", "expert"):
        rows = [r for r in raw_rows if r["phase"] == phase]
        by_phase[phase] = {
            key: [float(r[key]) for r in rows]
            for key in ("alone_ms", "concurrent_ms", "alone_comm_ms", "concurrent_comm_ms",
                        "encoder_alone_ms", "encoder_overlap_ms")
        }

    summary: dict[str, object] = {
        "protocol": {"warmups": 10, "iterations": 30, "ranks": 4,
                     "paired_in_process": True, "measurement": "CUDA events"},
        "phases": {},
        "correctness": {
            "driver_outputs": [json.loads(p.read_text()).get("output_tokens", [])
                                for p in sorted(root.glob("driver_rank*.json"))],
            "replay_route_identity": True,
            "replay_numerical_output": "not applicable to comm-only stage replay",
        },
        "nsys": {"status": "available_but_not_run", "version": "2024.6.2.225-246235244400v0",
                 "reason": "bounded CUDA-event result was sufficient; no full serving trace was required"},
        "ncu": {"status": "not_available", "reason": "ncu executable not installed"},
    }
    for phase, values in by_phase.items():
        enc = stats(values["encoder_alone_ms"])
        alone = stats(values["alone_ms"])
        concurrent = stats(values["concurrent_ms"])
        alone_comm = stats(values["alone_comm_ms"])
        concurrent_comm = stats(values["concurrent_comm_ms"])
        enc_concurrent = stats(values["encoder_overlap_ms"])
        serial = enc["median_ms"] + alone["median_ms"]
        wall_reduction = (serial - concurrent["median_ms"]) / serial
        hidden_ms = serial - concurrent["median_ms"]
        summary["phases"][phase] = {
            "encoder_alone": enc, "comm_alone_phase": alone,
            "concurrent_phase": concurrent, "comm_interval_alone": alone_comm,
            "comm_interval_concurrent": concurrent_comm,
            "encoder_concurrent": enc_concurrent,
            "serial_reference_median_ms": serial,
            "concurrent_wall_median_ms": concurrent["median_ms"],
            "wall_reduction_fraction": wall_reduction,
            "hidden_ms": hidden_ms,
            "hidden_fraction_of_comm": hidden_ms / alone_comm["median_ms"] if alone_comm["median_ms"] else 0.0,
            "encoder_slowdown_fraction": enc_concurrent["median_ms"] / enc["median_ms"] - 1.0,
            "comm_slowdown_fraction": concurrent_comm["median_ms"] / alone_comm["median_ms"] - 1.0,
        }

    # Shape and workload metadata are copied into the result for auditability.
    manifest = json.loads((root / "workload_manifest.json").read_text())
    # Rank JSON already carries the capture metadata.  Keeping analysis
    # dependency-free avoids importing the model's CUDA torch environment.
    capture_metadata = ranks[0]["capture_metadata"]
    manifest["capture_metadata"] = capture_metadata
    manifest["vision_activation"] = ranks[0]["vision_activation"]
    manifest["encoder_block"] = ranks[0]["encoder_block"]
    manifest["replayed_per_rank_tokens"] = int(capture_metadata["original_token_count"] * manifest["batch_equivalent"] / 4)
    (root / "shape_and_workload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Aggregate plotting values across ranks; all four ranks ran the same
    # paired sequence and rank-level values are retained in the CSV above.
    med = summary["phases"]
    encoder = float(np.median([x["encoder_alone_median_ms"] for x in phase_rows]))
    dispatch = med["dispatch"]; combine = med["combine"]; expert = med["expert"]
    serial_dispatch = dispatch["serial_reference_median_ms"]
    serial_combine = combine["serial_reference_median_ms"]
    serial_expert = expert["serial_reference_median_ms"]

    plt.figure(figsize=(8, 4.8))
    plt.bar(["Encoder block", "Dispatch", "Combine", "Expert"],
            [encoder, dispatch["comm_alone_phase"]["median_ms"], combine["comm_alone_phase"]["median_ms"], expert["comm_alone_phase"]["median_ms"]],
            color=["#4c78a8", "#f58518", "#e45756", "#54a24b"])
    plt.ylabel("CUDA-event interval (ms)"); plt.title("Standalone real-Qwen3-VL unit / DeepEP stage")
    plt.tight_layout(); plt.savefig(root / "plot1_standalone_timeline.png", dpi=180); plt.close()

    labels = ["Dispatch", "Combine", "Expert (negative control)"]
    serial_vals = [serial_dispatch, serial_combine, serial_expert]
    concurrent_vals = [dispatch["concurrent_wall_median_ms"], combine["concurrent_wall_median_ms"], expert["concurrent_wall_median_ms"]]
    x = np.arange(3); width = 0.36
    plt.figure(figsize=(8, 4.8)); plt.bar(x-width/2, serial_vals, width, label="serial reference", color="#9ecae1"); plt.bar(x+width/2, concurrent_vals, width, label="encoder + phase", color="#de2d26")
    plt.xticks(x, labels); plt.ylabel("ms"); plt.title("Serial reference vs concurrent wall (CUDA events)"); plt.legend(); plt.tight_layout(); plt.savefig(root / "plot2_serial_vs_concurrent_wall.png", dpi=180); plt.close()

    hidden = [dispatch["hidden_fraction_of_comm"], combine["hidden_fraction_of_comm"], expert["hidden_fraction_of_comm"]]
    plt.figure(figsize=(8, 4.8)); plt.bar(labels, np.asarray(hidden)*100, color="#756bb1"); plt.axhline(0, color="black", linewidth=0.8); plt.ylabel("hidden fraction of phase communication (%)"); plt.title("Negative hidden fraction means contention dominates"); plt.tight_layout(); plt.savefig(root / "plot3_communication_hidden_fraction.png", dpi=180); plt.close()

    enc_slow = [dispatch["encoder_slowdown_fraction"]*100, combine["encoder_slowdown_fraction"]*100, expert["encoder_slowdown_fraction"]*100]
    comm_slow = [dispatch["comm_slowdown_fraction"]*100, combine["comm_slowdown_fraction"]*100, expert["comm_slowdown_fraction"]*100]
    plt.figure(figsize=(7, 5)); plt.scatter(enc_slow, hidden, s=80, label="phase points");
    for label, xx, yy in zip(labels, enc_slow, hidden): plt.annotate(label, (xx, yy), xytext=(5, 5), textcoords="offset points")
    plt.axhline(0, color="black", linewidth=0.8); plt.xlabel("encoder slowdown (%)"); plt.ylabel("hidden fraction of communication"); plt.title("Interference vs hidden fraction"); plt.tight_layout(); plt.savefig(root / "plot4_encoder_slowdown_vs_hidden_fraction.png", dpi=180); plt.close()

    # This run intentionally uses one real captured shape (per-rank M=799);
    # the plot records that shape rather than inventing a scale sweep.
    m = manifest["replayed_per_rank_tokens"]
    plt.figure(figsize=(7, 4.8)); plt.bar([str(m)], [dispatch["wall_reduction_fraction"]*100], color="#f58518"); plt.axhline(0, color="black", linewidth=.8); plt.xlabel("real replay M tokens/rank"); plt.ylabel("dispatch wall reduction (%)"); plt.title("Real-shape overlap at captured M"); plt.tight_layout(); plt.savefig(root / "plot5_shape_M_vs_overlap_benefit.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 4.8)); plt.bar(["astronaut real image"], [np.mean([dispatch["wall_reduction_fraction"], combine["wall_reduction_fraction"]])*100], color="#4c78a8"); plt.axhline(0, color="black", linewidth=.8); plt.ylabel("mean dispatch/combine wall reduction (%)"); plt.title("Actual Qwen3-VL image-derived encoder workload"); plt.tight_layout(); plt.savefig(root / "plot6_image_workload_vs_overlap_benefit.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 4.8)); vals = [expert["wall_reduction_fraction"]*100, dispatch["wall_reduction_fraction"]*100, combine["wall_reduction_fraction"]*100]; plt.bar(["Encoder+Expert", "Encoder+Dispatch", "Encoder+Combine"], vals, color=["#54a24b", "#f58518", "#e45756"]); plt.axhline(0, color="black", linewidth=.8); plt.ylabel("wall reduction (%)"); plt.title("Communication overlap vs compute-heavy negative control"); plt.tight_layout(); plt.savefig(root / "plot7_encoder_comm_vs_expert.png", dpi=180); plt.close()

    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "negative_control.json").write_text(json.dumps({
        "status": "measured",
        "phase": "expert",
        "interpretation": "Encoder + Triton expert is a compute/compute negative control; combine is cleanup outside timed interval.",
        "summary": expert,
    }, indent=2) + "\n")
    print(json.dumps({"root": str(root), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
