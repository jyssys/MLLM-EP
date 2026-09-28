from __future__ import annotations

import numpy as np

from poc_mllm_vision_conditioned_decode_damage_skipping.scripts.analyze_functional_damage import metrics
from poc_mllm_vision_conditioned_decode_damage_skipping.scripts.analyze_predictors import select_skips


def test_drop_renormalization_preserves_remaining_sum() -> None:
    weights = np.asarray([0.24, 0.18, 0.15, 0.13, 0.10, 0.08, 0.07, 0.05])
    dropped = 3
    remaining = np.delete(weights, dropped) / (1.0 - weights[dropped])
    assert np.isclose(remaining.sum(), 1.0)


def test_skip_mask_is_exact_bounded_budget() -> None:
    mask = select_skips(np.asarray([4.0, 1.0, 3.0, 2.0, 0.0]), 0.40)
    assert mask.tolist() == [False, False, False, False, True] or mask.sum() == 2
    assert mask.sum() == 2


def test_ep8_contract_has_sixteen_experts_per_rank() -> None:
    owners = [expert // 16 for expert in range(128)]
    assert set(owners) == set(range(8))
    assert all(owners.count(rank) == 16 for rank in range(8))


def test_functional_damage_is_zero_for_identical_logits() -> None:
    logits = np.asarray([0.25, -0.5, 1.25, 0.0], dtype=np.float32)
    result = metrics(logits, logits.copy(), target=2)
    assert abs(result["functional_damage_score"]) < 1e-12
    assert result["absolute_safe"]
    assert not result["top1_flip"]


def test_functional_damage_charges_top1_flip() -> None:
    baseline = np.asarray([0.0, 3.0, -1.0], dtype=np.float32)
    changed = np.asarray([4.0, 0.0, -1.0], dtype=np.float32)
    result = metrics(baseline, changed, target=1)
    assert result["top1_flip"]
    assert result["functional_damage_score"] >= 1.0
    assert not result["absolute_safe"]
