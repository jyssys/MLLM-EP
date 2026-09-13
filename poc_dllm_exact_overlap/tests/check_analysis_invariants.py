#!/usr/bin/env python3
"""Fail closed if a report-critical scope or correctness invariant regresses."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


summary = json.loads((ROOT / "analysis_summary.json").read_text())
clean = rows("BASELINE_CLEAN.csv")
quality = rows("QUALITY_CORRECTNESS.csv")
pairs = rows("PAIRWISE_OVERLAP_MATRIX.csv")
phase_pairs = rows("PHASE_PAIR_OVERLAP_MATRIX.csv")
oracles = rows("CANDIDATE_ORACLES.csv")

assert summary["clean_restarts"] == {"gsm8k": 3, "humaneval": 3}
assert len(clean) == 6
assert all(row["matches_clean_reference"] == "True" for row in quality)
assert min(int(row["samples"]) for row in pairs) >= 30
assert len(pairs) == 48
assert min(int(row["samples"]) for row in phase_pairs) >= 30
assert len(phase_pairs) == 24
assert min(float(row["min_compute_cosine"]) for row in pairs) >= 0.9999
assert max(float(row["max_compute_rel_l2"]) for row in pairs) <= 0.005
assert min(float(row["min_compute_cosine"]) for row in phase_pairs) >= 0.9999
assert max(float(row["max_compute_rel_l2"]) for row in phase_pairs) <= 0.005
assert summary["strongest_direct_request_percent"] < 5.0
assert summary["strongest_service_upper_percent"] < 5.0
assert summary["strongest_max_every_wave_service_upper_percent"] < 8.0
assert summary["decision"] == "NO-OVERLAP-SIGNAL"

pipeline = [
    row for row in oracles
    if row["candidate"] == "complete_wave_dispatch_expert_combine_pipeline"
]
assert len(pipeline) == 2
assert all(float(row["direct_request_e2e_percent"]) == 0.0 for row in pipeline)

print("all exact-overlap analysis invariants pass")
