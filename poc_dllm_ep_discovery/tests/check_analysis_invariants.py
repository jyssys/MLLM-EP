#!/usr/bin/env python3
"""Fail closed if the final discovery claims drift from generated evidence."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(name: str) -> list[dict]:
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    clean = rows("BASELINE_CLEAN.csv")
    by_dataset = defaultdict(list)
    for row in clean:
        by_dataset[row["dataset"]].append(row)
    assert {key: len(value) for key, value in by_dataset.items()} == {
        "gsm8k": 3,
        "humaneval": 3,
    }

    quality = rows("QUALITY_REPRODUCIBILITY.csv")
    for dataset in ("gsm8k", "humaneval"):
        selected = [row for row in quality if row["dataset"] == dataset]
        assert len(selected) == 3
        assert len({row["answer_sha256"] for row in selected}) == 1
        assert len({row["accuracy"] for row in selected}) == 1

    checks = rows("TRACE_IDENTITY_CHECKS.csv")
    assert len(checks) == 2
    assert all(row["shape_stage_trajectory_exact"] == "True" for row in checks)

    layers = rows("LAYER_WAVE_METRICS.csv")
    assert len(layers) == 4650
    for row in layers:
        assert int(row["full_pairs"]) == int(row["physical_rows"]) * 8
        assert int(row["compact_pairs"]) == int(row["live_rows"]) * 8
        assert int(row["compact_active_experts"]) <= int(row["full_active_experts"])

    predictors = rows("PREDICTOR_RESULTS.csv")
    lookup = {(row["target"], row["sample_filter"], row["model"]): row for row in predictors}
    count = float(lookup[("expert_ms", "group_p99_trimmed", "M0_count")]["leave_dataset_out_rmse_ms"])
    shape = float(lookup[("expert_ms", "group_p99_trimmed", "M1_shape")]["leave_dataset_out_rmse_ms"])
    assert shape < 0.60 * count

    fixed = rows("FIXED_M_CONTROLS.csv")
    for dataset in ("gsm8k", "humaneval"):
        delta = next(row for row in fixed if row["dataset"] == dataset and row["stratum"] == "low_minus_high_percent")
        assert float(delta["full_active_experts_median"]) > 0

    oracles = rows("CANDIDATE_ORACLES.csv")
    for row in oracles:
        if row["candidate"] in {
            "perfect_current_fragmentation_removal_p99_trimmed",
            "post_compaction_fragmentation_residual_p99_trimmed",
            "feasible_tiny_shape_specialization_50pct_capture_p99_trimmed",
        }:
            assert float(row["clean_e2e_percent"]) < 5.0

    methods = rows("METHOD_CANDIDATES.csv")
    assert not any("HOLD" in row["decision"] or "GO" in row["decision"] for row in methods)
    print("analysis invariants: PASS")


if __name__ == "__main__":
    main()
