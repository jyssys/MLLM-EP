#!/usr/bin/env python3
"""Lightweight invariant and oracle checks for the EP2 PoC artifacts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "team_positive_20260911_211456" / "raw"


def load(name: str) -> dict:
    return json.loads((RAW / name).read_text())


class EP2Invariants(unittest.TestCase):
    def test_ownership_is_complete_and_disjoint(self):
        rank0 = load("stageB_warm_team_ep2_r4_rank0.json")["ownership"]
        rank1 = load("stageB_warm_team_ep2_r4_rank1.json")["ownership"]
        self.assertEqual(rank0["experts_per_rank"], 64)
        self.assertEqual(rank1["experts_per_rank"], 64)
        self.assertEqual(rank0["owned_expert_range"], [0, 63])
        self.assertEqual(rank1["owned_expert_range"], [64, 127])

    def test_true_ep2_topology(self):
        payload = load("stageB_warm_team_ep2_r4_rank0.json")
        self.assertEqual(payload["topology"], {"tp": 1, "dp": 1, "ep": 2})
        self.assertIn("all_to_all_single", payload["backend"])

    def test_remote_dispatch_is_nonzero(self):
        payload = load("stageC_trace_team_ep2_fused_shapes_rank0.json")
        remote = sum(row["remote_assignments_from_source"] for row in payload["stage_rows"])
        total = sum(row["assignments"] for row in payload["stage_rows"])
        self.assertGreater(remote, 0)
        self.assertLess(remote, total)

    def test_dispatch_combine_work_conservation(self):
        payload = load("stageC_trace_team_ep2_fused_shapes_rank0.json")
        for row in payload["stage_rows"]:
            self.assertEqual(row["local_assignments"] + row["remote_assignments_from_source"], row["assignments"])
            self.assertEqual(row["dispatch_bytes_hidden"], row["combine_bytes_hidden"])

    def test_team_positive_under_ep2(self):
        baseline = load("stageB_warm_baseline_ep2_r4_rank0.json")["records"][0]["elapsed_s"]
        team = load("stageB_warm_team_ep2_r4_rank0.json")["records"][0]["elapsed_s"]
        self.assertGreater(baseline / team, 1.1)

    def test_layer_numerical_equivalence(self):
        for name in (
            "stageB_layer_validation_python_rank0.json",
            "stageC_layer_validation_fused_rank0.json",
        ):
            validation = load(name)["layer_validation"]
            self.assertGreaterEqual(validation["output_cosine"], 0.9999)
            self.assertLessEqual(validation["output_rel_l2"], 0.001)
            self.assertEqual(validation["router_logits_max_abs"], 0.0)

    def test_oracle_never_exceeds_source_mass(self):
        payload = load("stageC_trace_team_ep2_fused_shapes_rank0.json")
        clean = load("stageC_clean_team_ep2_fused_rank0.json")["records"][0]["elapsed_s"] * 1000
        rows = payload["module_event_summary"]
        m32 = rows["decoder_layer_rows_32"]
        m128 = rows["decoder_layer_rows_128"]
        saving = m128["count"] * (m128["mean_ms"] - m32["mean_ms"])
        self.assertGreater(saving, 0)
        self.assertLess(saving, clean)
        self.assertLess(saving / clean * 100, 10)


if __name__ == "__main__":
    unittest.main()
