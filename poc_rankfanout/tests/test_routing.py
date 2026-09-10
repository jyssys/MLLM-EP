import numpy as np
import pytest

from poc_rankfanout.rankfanout.routing import (
    balanced_fanout_route,
    linear_expert_to_rank,
    summarize_route,
    token_fanout,
)


@pytest.mark.parametrize("fanout", [1, 2, 3, 4])
def test_exact_fanout(fanout):
    route = balanced_fanout_route(128, fanout, seed=7)
    mapping = linear_expert_to_rank()
    assert route.shape == (128, 8)
    assert np.all(token_fanout(route, mapping) == fanout)
    assert all(np.unique(row).size == 8 for row in route)
    summary = summarize_route(route, mapping)
    assert summary.mean_fanout == fanout
    assert summary.rank_load_max_over_mean == 1.0


def test_duplicates_count_unique_rank_once():
    mapping = linear_expert_to_rank()
    route = np.array([[0, 1, 2, 3, 32, 33, 34, 35]])
    assert token_fanout(route, mapping).tolist() == [2]
