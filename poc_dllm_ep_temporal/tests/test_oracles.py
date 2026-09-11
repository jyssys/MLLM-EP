import numpy as np

from poc_dllm_ep_temporal.dllm_ep.batching_oracle import exact_first_batch, greedy_complementary
from poc_dllm_ep_temporal.dllm_ep.replica_oracle import perfect_replica_oracle
from poc_dllm_ep_temporal.dllm_ep.routing_metrics import linear_expert_ownership, rank_loads


def test_linear_ownership_and_conservation():
    ownership = linear_expert_ownership(8, 4)
    assert ownership.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    expert = np.arange(1, 9)
    ranks = rank_loads(expert, ownership, 4)
    assert ranks.tolist() == [3, 7, 11, 15]
    assert ranks.sum() == expert.sum()


def test_complementary_oracle_prefers_a_plus_b():
    vectors = np.array([[10, 1, 1, 1], [1, 10, 1, 1], [9, 1, 1, 1]], dtype=float)
    assert exact_first_batch(vectors, 2) == (0, 1)
    assert set(greedy_complementary(vectors, 2)[0]) == {0, 1}


def test_replica_oracle_charges_copy_cost():
    ownership = np.array([0, 1, 2, 3])
    future = np.array([[100, 1, 1, 1], [100, 1, 1, 1]], dtype=float)
    decision = perfect_replica_oracle(future, ownership, visible_copy_ms=10, latency_per_max_assignment_ms=1, lifetime=2)
    assert decision is not None
    assert decision.expert == 0
    assert decision.gross_latency_ms > 10
    assert decision.net_latency_ms == decision.gross_latency_ms - 10
