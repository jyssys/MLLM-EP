#!/usr/bin/env python3
"""Extract cleanly aligned logical refinement state from stability logs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def median(values):
    return statistics.median(values) if values else float("nan")


rows = []
for dataset in ("gsm8k", "humaneval"):
    log = ROOT / f"logs/stability_{dataset}_ep4_b32_mini32_r1_g32.log"
    for line in log.open():
        if not line.startswith("[LLADA_DENOISE]"):
            continue
        record = json.loads(line[len("[LLADA_DENOISE]") :])
        if not str(record.get("request_id", "")).startswith("measured"):
            continue
        live_confidence = []
        live_margin = []
        for mask, confidence, margin in zip(
            record["mask_before"],
            record["token_confidence"],
            record["confidence_margin"],
        ):
            for is_live, token_confidence, token_margin in zip(mask, confidence, margin):
                if is_live:
                    live_confidence.append(float(token_confidence))
                    live_margin.append(float(token_margin))
        rows.append(
            {
                "dataset": dataset,
                "iteration": int(record["iteration"]),
                "phase": record["phase"],
                "normalized_iteration": float(record["iteration"])
                / max(1, int(record["iteration"])),
                "ready_requests": len(record["sequence_ids"]),
                "physical_rows": int(record["physical_rows"]),
                "decision_live_rows": sum(sum(int(value) for value in mask) for mask in record["mask_before"]),
                "live_ratio": float(record["live_ratio"]),
                "accepted_rows": sum(int(value) for value in record["accepted"]),
                "acceptance_rate_live": sum(int(value) for value in record["accepted"])
                / max(1, sum(sum(int(value) for value in mask) for mask in record["mask_before"])),
                "live_confidence_median": median(live_confidence),
                "live_margin_median": median(live_margin),
                "model_forward_ms_observer": float(record["model_forward_ms"]),
            }
        )

for dataset in ("gsm8k", "humaneval"):
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    denominator = max(row["iteration"] for row in dataset_rows)
    for row in dataset_rows:
        row["normalized_iteration"] = row["iteration"] / max(denominator, 1)

destination = ROOT / "REFINEMENT_STATE_TRACE.csv"
with destination.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
for dataset in ("gsm8k", "humaneval"):
    values = [row for row in rows if row["dataset"] == dataset]
    axes[0].plot(
        [row["normalized_iteration"] for row in values],
        [row["live_ratio"] for row in values],
        marker=".",
        label=dataset,
    )
    axes[1].plot(
        [row["normalized_iteration"] for row in values],
        [row["live_margin_median"] for row in values],
        marker=".",
        label=dataset,
    )
axes[0].set_ylabel("decision-live / physical rows")
axes[1].set_ylabel("live confidence margin (median)")
axes[1].set_xlabel("normalized refinement iteration")
for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend()
figure.suptitle("Logical refinement state (observer run; not latency evidence)")
figure.tight_layout()
figure.savefig(ROOT / "figures/00_refinement_state.png", dpi=180)
plt.close(figure)
print(json.dumps({"rows": len(rows), "output": str(destination)}))
