import numpy as np

from route_pruning import RoutePruningConfig, select_pruned_routes
from scripts.replay_method_hist_compute import build_recv_ids


def fixture():
    ids = np.asarray([
        [0, 40, 70, 100, 130, 160, 200, 240],
        [1, 41, 71, 101, 131, 161, 201, 241],
    ], dtype=np.int16)
    weights = np.asarray([
        [.40, .30, .20, .10, .04, .03, .02, .01],
        [.35, .25, .20, .10, .04, .03, .02, .01],
    ], dtype=np.float32)
    return ids, weights


def test_zero_budget_is_exact_noop():
    ids, weights = fixture()
    removed, stats = select_pruned_routes(
        ids, weights, RoutePruningConfig("vanilla", 0),
    )
    assert not removed.any()
    assert stats["removed_mass"] == 0


def test_utility_policy_preserves_effective_top4_without_renormalizing():
    ids, weights = fixture()
    removed, stats = select_pruned_routes(
        ids, weights, RoutePruningConfig("p2", .05),
    )
    ranks = stats["ranks"]
    assert not np.any(removed & (ranks <= 4))
    assert np.all(8 - removed.sum(axis=1) >= 4)
    assert stats["removed_mass"] <= .05 * len(ids) + 1e-7


def test_p4_only_selects_above_mean_destination_ranks():
    ids, weights = fixture()
    ids[:, :5] = np.asarray([0, 1, 2, 3, 4])
    removed, stats = select_pruned_routes(
        ids, weights, RoutePruningConfig("p4", .10, target_ep=8),
    )
    owners = ids // 32
    if removed.any():
        assert np.all(stats["rank_pressure"][owners[removed]] > 0)


def test_replay_layout_preserves_histogram_and_unique_rows():
    counts = np.asarray([4, 2, 3, 0], dtype=np.int64)
    recv_ids = build_recv_ids(counts, unique_rows=5)
    assert recv_ids.shape == (5, 8)
    assert np.array_equal(np.bincount(recv_ids[recv_ids >= 0], minlength=4), counts)
    for row in recv_ids:
        valid = row[row >= 0]
        assert len(valid) == len(np.unique(valid))
