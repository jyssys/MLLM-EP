import numpy as np
import torch

from method_adapters.team_port import classify_hot_cold, team_speculative_candidates


def test_team_hot_cold_classification():
    masked = np.asarray([False, True, True, True, True, False])
    decoded = ~masked
    confidence = np.asarray([0, .2, .8, .1, .9, 0])
    hot, cold = classify_hot_cold(masked, confidence, decoded, threshold=.7, distance=0)
    assert np.array_equal(np.flatnonzero(hot), np.asarray([2, 4]))
    assert np.array_equal(np.flatnonzero(cold), np.asarray([1, 3]))


def test_team_speculative_candidate_construction():
    tokens = torch.tensor([[9, 9, 9, 9]])
    proposals = torch.tensor([[1, 2, 3, 4]])
    masked = torch.tensor([[False, True, False, True]])
    blocks, chosen = team_speculative_candidates(tokens, proposals, masked)
    assert chosen.tolist() == [1, 3]
    assert blocks[0].tolist() == [9, 9, 9, 9]
    assert blocks[1].tolist() == [9, 2, 9, 9]
    assert blocks[2].tolist() == [9, 9, 9, 4]
    assert blocks[3].tolist() == [9, 2, 9, 4]
