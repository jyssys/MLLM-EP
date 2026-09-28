from __future__ import annotations

from pathlib import Path

import numpy as np

from poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.contract import read_json
from poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.build_structured_probe_plan import (
    projection_energy, right_basis,
)


ROOT = Path(__file__).parents[1]


def test_subspace_energy_is_bounded() -> None:
    rows = np.eye(8, dtype=np.float64)[:4]
    basis = right_basis(rows, rank=2)
    assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-10)
    assert 0 <= projection_energy(rows[0], basis) <= 1.0000001


def test_frozen_manifest_contract_if_present() -> None:
    path = ROOT / "artifacts/manifests/frozen_visual_grounding_cohort.json"
    if not path.exists(): return
    payload = read_json(path); cases = payload["cases"]
    assert len([row for row in cases if row["split"] == "calibration"]) == 8
    assert len([row for row in cases if row["split"] == "heldout"]) == 24
    assert all(row["visual_token_relative_difference"] <= 0.10 for row in cases)
    assert len({row["original_image_sha256"] for row in cases}) == len(cases)
    all_hashes = [
        image_hash
        for row in cases
        for image_hash in (row["original_image_sha256"], row["counterfactual_image_sha256"])
    ]
    assert len(set(all_hashes)) == len(all_hashes)
    assert payload["selection_uses_routing_grounding_or_damage"] is False


def test_scope_excludes_latency_and_modes() -> None:
    spec = (ROOT / "research_spec.md").read_text()
    assert "No MoDES implementation or comparison" in spec
    assert "no TPOT/E2E/speedup claim" in spec
