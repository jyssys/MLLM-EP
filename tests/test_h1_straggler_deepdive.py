import numpy as np

from scripts.analyze_h1_straggler_deepdive import (
    batch_balance_and_traffic,
    excess_scores,
)


def test_excess_attribution_sums_to_rank_excess():
    histogram = np.zeros(256, dtype=np.int32)
    histogram[:4] = [40, 30, 20, 10]
    times = np.asarray([2.0, 1.0, 1.0, 1.0])
    scores = excess_scores(histogram, times, ep=4)
    assert np.isclose(scores.sum(), 0.75)
    assert np.allclose(scores[:4], [0.30, 0.225, 0.15, 0.075])
    assert np.count_nonzero(scores[64:]) == 0


def test_replica_split_preserves_assignments_and_moves_same_expert():
    histograms = np.zeros((1, 256), dtype=np.int32)
    histograms[0, 0] = 100
    histograms[0, 64] = 20
    assignment = np.ones((1, 4, 4), dtype=np.int32) * 10
    unique = np.ones((1, 4, 4), dtype=np.int32) * 5
    experts = np.full((1, 8), -1, dtype=np.int16)
    destinations = np.full((1, 8), -1, dtype=np.int8)
    experts[0, 0] = 0
    destinations[0, 0] = 2
    balanced, outgoing, incoming, moved = batch_balance_and_traffic(
        histograms, assignment, unique, np.asarray([32], dtype=np.int32),
        experts, destinations, 4, 1,
    )
    assert balanced.shape == (1, 4, 65)
    assert np.isclose(balanced.sum(), histograms.sum())
    assert moved[0] > 0
    assert outgoing.shape == incoming.shape == (1, 4)
    assert np.all(outgoing >= 0) and np.all(incoming >= 0)
