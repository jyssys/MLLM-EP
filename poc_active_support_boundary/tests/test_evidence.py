"""CPU-only evidence checks; do not initialize CUDA."""

import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_support_boundary.py"
SPEC = importlib.util.spec_from_file_location("support_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def read_rows(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


class EvidenceTests(unittest.TestCase):
    def test_three_independent_restarts_and_matched_work(self):
        controls = read_rows("SYNTHETIC_CONTROLS.csv")
        self.assertEqual(len(controls), 65)
        self.assertEqual({r["restarts"] for r in controls}, {"3"})
        self.assertEqual({r["repetitions_per_restart"] for r in controls}, {"40"})
        for row in controls:
            self.assertGreaterEqual(int(row["total_rows"]),
                                    int(row["active_experts"]))
            self.assertEqual(row["useful_routed_rows_fixed"], "yes")
            self.assertEqual(row["useful_expert_arithmetic_flops_fixed"], "yes")

    def test_never_count_work_or_metadata_twice(self):
        cases = read_rows("REAL_REFINEMENT_MAPPING.csv")
        self.assertEqual(len(cases), 46)
        for row in cases:
            base = float(row["torch_grouped_full64_ms"])
            saved = float(row["combined_credible_saved_ms"])
            self.assertGreater(base, 0.0)
            self.assertGreaterEqual(saved, 0.0)
            self.assertLessEqual(saved, base)
            self.assertLessEqual(float(row["credible_boundary_saved_ms"]),
                                 float(row["perfect_boundary_zero_cost_saved_ms"]))
            self.assertEqual(row["materialization_hbm_direct_measured"], "no")

    def test_aim_at_strongest_grouped_operator(self):
        raw = read_rows("REQUEST_ORACLES.csv")
        for row in raw:
            self.assertIn("hypothetical_post_grouped_expert_pie_pct", row)
            self.assertLess(float(row["hypothetical_post_grouped_expert_pie_pct"]),
                            float(row["observer_attributed_production_expert_pie_pct"]))
            self.assertNotIn("NCCL", row["evidence"])
            ceiling = float(row["projected_optimistic_request_e2e_upper_bound_pct"])
            if row["oracle"] == "O2_plus_all_materialization_unrealistic":
                self.assertLess(ceiling, 8)
            else:
                self.assertLess(ceiling, 5)

    def test_numerical_and_cuda_boundary(self):
        summary = json.loads((ROOT / "ANALYSIS_SUMMARY.json").read_text())
        self.assertGreater(summary["min_cosine_fused_vs_grouped"], 0.9999)
        self.assertLess(summary["max_relative_l2_fused_vs_grouped"], 0.005)
        for mode in ("support", "dense_all", "compacted_all"):
            for restart in range(1, 4):
                audit = json.loads((ROOT / "results" / f"{mode}_r{restart}" /
                                    "AUDIT.json").read_text())
                self.assertEqual(audit["cuda_visible_devices"], "4,5,6,7")
                self.assertEqual(audit["physical_gpu"], 4)
                self.assertEqual(audit["repetitions"], 40)


if __name__ == "__main__":
    unittest.main()
