#!/usr/bin/env python3
"""Fail-fast invariants for logical wave/rank aggregation."""

from __future__ import annotations

import csv
from pathlib import Path


root = Path(__file__).resolve().parents[1]
with (root / "WAVE_SHAPES.csv").open() as handle:
    waves = list(csv.DictReader(handle))
with (root / "LAYER_SHAPES.csv").open() as handle:
    layers = list(csv.DictReader(handle))

assert {int(row["mini_batch_size"]) for row in waves} == {1, 2, 4, 8, 16, 32}
assert {int(row["layer"]) for row in layers} == {1, 16, 31}
for row in waves:
    assert int(float(row["physical_rows"])) == 32 * int(float(row["selected_sequences"]))
    assert 0 <= float(row["live_ratio"]) <= 1
for row in layers:
    assignments = int(float(row["physical_rows"])) * 8
    assert abs(float(row["rank_load_mean"]) * 4 - assignments) < 1e-6
    assert 0 <= int(float(row["remote_pairs"])) <= assignments
    assert row["phase"] in {"early", "middle", "late"}

print(f"validated {len(waves)} waves and {len(layers)} representative-layer rows")
