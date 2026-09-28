from __future__ import annotations

import json
import unittest
from pathlib import Path

from poc_mllm_rank_aware_active_expert_reduction.contract import expert_owner


ROOT = Path(__file__).resolve().parents[1]


class AnalysisContractTest(unittest.TestCase):
    def test_linear_expert_ownership(self) -> None:
        for ep in (2, 4, 8):
            owners = [expert_owner(expert, ep) for expert in range(128)]
            self.assertEqual(set(owners), set(range(ep)))
            self.assertTrue(all(owners.count(rank) == 128 // ep for rank in range(ep)))

    def test_ep8_is_structural_only(self) -> None:
        q3 = json.loads((ROOT / "artifacts/q3_structural_oracle.json").read_text())
        verdict = json.loads((ROOT / "artifacts/final_verdict.json").read_text())
        self.assertEqual(q3["ep8_label"], "STRUCTURAL_EP8_PROJECTION_ONLY")
        self.assertEqual(verdict["ep8"], "STRUCTURAL_EP8_PROJECTION_ONLY")
        self.assertFalse(verdict["q5"]["speedup_claim"])

    def test_visual_gate_prevents_runtime_promotion(self) -> None:
        q2 = json.loads((ROOT / "artifacts/q2_vds_analysis.json").read_text())
        verdict = json.loads((ROOT / "artifacts/final_verdict.json").read_text())
        self.assertFalse(q2["gate"]["q2_pass"])
        self.assertEqual(verdict["q4"]["status"],
                         "NOT_RUN_BY_Q2_MLLM_GATE_AND_NO_MATCHED_SAFE_BATCH_LABELS")
        self.assertEqual(verdict["q5"]["status"], "NOT_RUN_BY_GATE")


if __name__ == "__main__":
    unittest.main()
