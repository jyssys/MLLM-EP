#!/usr/bin/env python3
"""Analyze temporal cacheability and payload-to-request compression oracles."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYERS = (1, 8, 16, 24, 31)
HIDDEN = 4096


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def clean_median_ms(root: Path, dataset: str) -> float:
    values = []
    for path in sorted((root / "logs").glob(f"clean_{dataset}_*.log")):
        match = re.findall(r"Forward:\s*\d+, Time:\s*([0-9.]+)", path.read_text())
        if match:
            values.append(float(match[-1]) * 1000.0)
    if len(values) < 3:
        raise RuntimeError(f"need three clean restarts for {dataset}; found {values}")
    return float(np.median(values))


def interpolate(xs: np.ndarray, ys: np.ndarray, value: float) -> float:
    value = max(float(value), float(xs[0]))
    return float(np.interp(np.log2(value), np.log2(xs), ys))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    comm = list(csv.DictReader((root / "trackA/deepep_payload_sweep/summary.csv").open()))
    comm_x = np.array([float(row["source_tokens_per_rank"]) for row in comm])
    dispatch_y = np.array([float(row["dispatch_p50_ms"]) for row in comm])
    codec_rows = list(csv.DictReader((root / "trackA/delta_codec.csv").open()))
    codec_by_name = {}
    for name in ("fp8", "int8_row_scaled"):
        selected = [row for row in codec_rows if row["codec"] == name]
        codec_by_name[name] = (
            np.array([float(row["rows"]) for row in selected]),
            np.array([float(row["codec_p50_ms"]) for row in selected]),
        )

    temporal_summary = []
    oracle_rows = []
    for dataset in ("gsm8k", "humaneval"):
        trace_dir = root / f"trackA/trace/{dataset}/r1"
        temporal = jsonl(trace_dir / "temporal_comm_rank0.jsonl")
        rank_rows = []
        for rank in range(4):
            rows = [
                row
                for row in jsonl(trace_dir / f"ep_rank{rank}.jsonl")
                if row["request_id"].startswith("measured")
                and row["layer"] in LAYERS
                and row["physical_m_global"] <= 1024
            ]
            if len(rows) != len(temporal):
                raise RuntimeError(f"{dataset} rank {rank}: {len(rows)} != {len(temporal)}")
            rank_rows.append(rows)
        # Temporal tensor diagnostics deliberately run between the router and
        # DeepEP call and contaminate event timing via outstanding collectives.
        # Never use those observer-heavy dispatch events for an E2E claim.
        dispatch_observer = np.array(
            [max(rank_rows[rank][index]["dispatch_ms"] for rank in range(4)) for index in range(len(temporal))]
        )
        clean_ms = clean_median_ms(root, dataset)

        for lag in (1, 2, 4):
            records = [row["lags"][str(lag)] for row in temporal]
            for phase in ("all", "early", "middle", "late"):
                indices = [
                    index for index, row in enumerate(temporal)
                    if phase == "all" or row["phase"] == phase
                ]
                selected = [records[index] for index in indices]
                remote = sum(row["current_remote_destinations"] for row in selected)
                hit = sum(row["cache_hit_remote_destinations"] for row in selected)
                same_expert = sum(row["same_expert_remote_destinations"] for row in selected)
                numeric = [row for row in selected if "hidden_cosine_p50" in row]
                temporal_summary.append(
                    {
                        "dataset": dataset,
                        "lag": lag,
                        "phase": phase,
                        "waves_x_layers": len(selected),
                        "remote_destinations": remote,
                        "remote_cache_hit_fraction": hit / remote if remote else 0.0,
                        "same_expert_remote_fraction": same_expert / remote if remote else 0.0,
                        "same_rank_different_expert_fraction": (hit - same_expert) / remote if remote else 0.0,
                        "hidden_cosine_p50_median": float(np.median([row["hidden_cosine_p50"] for row in numeric])) if numeric else np.nan,
                        "hidden_rel_l2_p50_median": float(np.median([row["hidden_rel_l2_p50"] for row in numeric])) if numeric else np.nan,
                        "fp8_reconstruction_rel_l2_p50_median": float(np.median([row["fp8"]["rel_l2_p50"] for row in numeric])) if numeric else np.nan,
                        "int8_reconstruction_rel_l2_p50_median": float(np.median([row["int8"]["rel_l2_p50"] for row in numeric])) if numeric else np.nan,
                        "delta_lt_1e_2_fraction_median": float(np.median([row["delta_lt_1e_2_fraction"] for row in numeric])) if numeric else np.nan,
                        "delta_top10_energy_fraction_median": float(np.median([row["delta_top10_energy_fraction"] for row in numeric])) if numeric else np.nan,
                        "int8_symbol_entropy_bits_median": float(np.median([row["int8_symbol_entropy_bits"] for row in numeric])) if numeric else np.nan,
                    }
                )

        # Only lag-1 is usable for an adjacent cache protocol.  Periodic full
        # refresh removes a proportional share of otherwise eligible delta hits.
        lag1 = [row["lags"]["1"] for row in temporal]
        calibrated_full_dispatch_ms = sum(
            interpolate(
                comm_x,
                dispatch_y,
                row["lags"]["1"]["current_remote_destinations"] / 12.0,
            )
            for row in temporal
        )
        extrapolation = 32.0 / len(LAYERS)
        dispatch_share_pct = 100.0 * calibrated_full_dispatch_ms * extrapolation / clean_ms
        for refresh_period in (2, 4, 8, 16):
            refresh_multiplier = (refresh_period - 1.0) / refresh_period
            for codec in ("fp8", "int8_row_scaled"):
                gross_saved_ms = 0.0
                codec_cost_ms = 0.0
                full_bytes = 0.0
                wire_bytes = 0.0
                cache_hits = 0.0
                remote_destinations = 0.0
                for index, row in enumerate(lag1):
                    remote = float(row["current_remote_destinations"])
                    hits = float(row["cache_hit_remote_destinations"]) * refresh_multiplier
                    misses = remote - hits
                    metadata = 2.0 * hits if codec == "int8_row_scaled" else 0.0
                    full = remote * HIDDEN * 2.0
                    wire = misses * HIDDEN * 2.0 + hits * HIDDEN + metadata
                    full_bytes += full
                    wire_bytes += wire
                    cache_hits += hits
                    remote_destinations += remote

                    full_equiv = remote / 12.0
                    compressed_equiv = wire / (HIDDEN * 2.0 * 12.0)
                    calibration_full = interpolate(comm_x, dispatch_y, full_equiv)
                    calibration_compressed = interpolate(comm_x, dispatch_y, compressed_equiv)
                    gross_saved_ms += max(0.0, calibration_full - calibration_compressed)
                    codec_x, codec_y = codec_by_name[codec]
                    codec_cost_ms += interpolate(codec_x, codec_y, hits / 4.0) if hits else 0.0

                gross_e2e = 100.0 * gross_saved_ms * extrapolation / clean_ms
                feasible_saved = max(0.0, gross_saved_ms - codec_cost_ms)
                feasible_e2e = 100.0 * feasible_saved * extrapolation / clean_ms
                raw_byte_saving = 100.0 * (1.0 - wire_bytes / full_bytes)
                oracle_rows.append(
                    {
                        "dataset": dataset,
                        "codec": codec,
                        "refresh_period": refresh_period,
                        "clean_e2e_ms": clean_ms,
                        "selected_layer_calibrated_dispatch_ms": calibrated_full_dispatch_ms,
                        "selected_layer_observer_dispatch_ms_excluded": float(dispatch_observer.sum()),
                        "extrapolated_dispatch_share_pct": dispatch_share_pct,
                        "remote_cache_hit_fraction_effective": cache_hits / remote_destinations,
                        "raw_dispatch_byte_saving_pct": raw_byte_saving,
                        "payload_sensitive_gross_e2e_oracle_pct": gross_e2e,
                        "unfused_codec_cost_selected_ms": codec_cost_ms,
                        "feasible_e2e_oracle_pct": feasible_e2e,
                        "gate": "KILL" if feasible_e2e < 5 else ("WEAK" if feasible_e2e < 8 else "PROMOTE"),
                        "evidence_boundary": "5 traced layers extrapolated to 32; clean request denominator; observer event timing; no accumulated-error quality credit",
                    }
                )

    write_csv(analysis / "trackA_temporal_summary.csv", temporal_summary)
    write_csv(analysis / "trackA_compression_oracles.csv", oracle_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for dataset in ("gsm8k", "humaneval"):
        rows = [row for row in temporal_summary if row["dataset"] == dataset and row["lag"] == 1 and row["phase"] != "all"]
        axes[0].plot([row["phase"] for row in rows], [100 * row["remote_cache_hit_fraction"] for row in rows], marker="o", label=dataset)
    axes[0].set_ylabel("lag-1 remote destination cache hit (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    best_rows = [row for row in oracle_rows if row["refresh_period"] == 16]
    labels = [f"{row['dataset']}\n{row['codec'].split('_')[0]}" for row in best_rows]
    axes[1].bar(labels, [row["feasible_e2e_oracle_pct"] for row in best_rows])
    axes[1].axhline(5, color="red", linestyle="--", linewidth=1, label="kill gate")
    axes[1].set_ylabel("feasible request E2E oracle (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(analysis / "trackA_cacheability_and_e2e_oracle.png", dpi=180)


if __name__ == "__main__":
    main()
