from __future__ import annotations

import pytest
from pathlib import Path

from poc_mllm_rank_aware_active_expert_reduction.q2_2_causal_visual_path.contract import read_json

from poc_mllm_rank_aware_active_expert_reduction.q2_2_causal_visual_path.contract import (
    ALLOWED_GPUS, LAYERS, MODEL_REVISION, STEPS, TOP_K,
)
from poc_mllm_rank_aware_active_expert_reduction.q2_2_causal_visual_path.scripts.analyze_q2_2 import (
    separation,
)


def test_frozen_contract() -> None:
    assert ALLOWED_GPUS == (2, 3, 6, 7)
    assert MODEL_REVISION == "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
    assert TOP_K == 8
    assert LAYERS == (4, 12, 24, 36, 44, 47)
    assert STEPS == (8, 16, 32, 48)


def test_signed_visual_separation_is_not_clipped() -> None:
    rows = [
        {"pcs": 0.1, "d_visual": -0.4},
        {"pcs": 0.2, "d_visual": -0.2},
        {"pcs": 0.8, "d_visual": 0.3},
        {"pcs": 0.9, "d_visual": 0.5},
    ]
    result = separation(rows, "pcs", "d_visual", 0.2, 0.8)
    assert result["high_minus_low_mean"] == pytest.approx(0.7)


def test_nonactive_convention_is_zero_damage_not_substitution() -> None:
    d_orig, d_cf_nonactive = 0.12, 0.0
    assert d_orig - d_cf_nonactive == 0.12


def test_frozen_fresh_heldout_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = read_json(root / "artifacts/manifests/frozen_q2_2_cohort.json")
    old = read_json(
        Path(__file__).resolve().parents[3]
        / "poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/manifests/frozen_damage_cohort.json"
    )
    heldout = [row for row in manifest["cases"] if row["split"] == "heldout"]
    old_hashes = {row["original_image_sha256"] for row in old["anchors"]}
    assert len(heldout) >= 24
    assert not ({row["original_image_sha256"] for row in heldout} & old_hashes)
    assert all(row["visual_token_relative_difference"] <= 0.10 for row in manifest["cases"])
    assert all(len(row["forced_token_ids"]) == row["probe_step"] + 1 for row in manifest["cases"])
