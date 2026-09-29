import torch

from poc_mllm_rank_aware_active_expert_reduction.p1_rank_aware_physical_bridge.instrumentation import (
    _collapse_same_pairs,
    _rank_k6,
    _tail_k6,
)


def test_policy_path_and_owner_contracts() -> None:
    ids = torch.tensor([[0, 1, 33, 34, 65, 66, 97, 98], [2, 3, 35, 36, 67, 68, 99, 100]])
    weights = torch.softmax(torch.arange(16, dtype=torch.float32).reshape(2, 8), dim=-1)
    tail_ids, tail_weights = _tail_k6(ids, weights)
    rank_ids, rank_weights = _rank_k6(ids, weights)
    collapsed_ids, collapsed_weights = _collapse_same_pairs(ids, weights)
    assert tail_ids.ge(0).sum().item() == rank_ids.ge(0).sum().item() == 12
    assert torch.allclose(tail_weights.sum(-1), torch.ones(2))
    assert torch.allclose(rank_weights.sum(-1), torch.ones(2))
    assert torch.equal(ids // 32, collapsed_ids // 32)
    assert torch.equal(weights, collapsed_weights)
