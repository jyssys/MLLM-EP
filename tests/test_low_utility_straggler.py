import numpy as np

from scripts.analyze_low_utility_straggler import (
    canonical_mass_preserving,
    effective_slot_ranks,
    selection_mask,
)


def _fixture():
    weights = np.asarray([[[.10, .40, .05, .15, .04, .12, .08, .06]]], dtype=np.float32)
    normalized = weights / weights.sum(axis=-1, keepdims=True)
    ids = np.arange(8, dtype=np.int16).reshape(1, 1, 8)
    valid = np.ones((1, 1), dtype=np.bool_)
    return weights, normalized, ids, valid


def test_effective_slot_rank_does_not_assume_router_array_order():
    weights, _normalized, _ids, _valid = _fixture()
    ranks = effective_slot_ranks(weights)
    # Weight .40 is first; weight .04 is eighth, despite their array indices.
    assert ranks[0, 0, 1] == 1
    assert ranks[0, 0, 4] == 8
    assert sorted(ranks.reshape(-1).tolist()) == list(range(1, 9))


def test_mass_preserving_policy_respects_min_k():
    weights, normalized, _ids, valid = _fixture()
    removed = canonical_mass_preserving(
        normalized, effective_slot_ranks(weights), valid, .90, 4
    )
    assert int((~removed[0, 0]).sum()) >= 4
    assert float(normalized[~removed].sum()) >= .90


def test_budgeted_selection_only_removes_tail_slots():
    weights, normalized, ids, valid = _fixture()
    ranks = effective_slot_ranks(weights)
    pressure = np.ones((1, 4), dtype=np.float64)
    removed = selection_mask(
        normalized, ranks, ids, valid, 4, pressure, .30, 0, 4,
        np.zeros_like(normalized, dtype=np.bool_), False,
    )
    assert np.all(ranks[removed] > 4)
    assert int((~removed[0, 0]).sum()) >= 4
    assert float(normalized[removed].sum()) <= .30 + 1e-6
