import torch

from method2.selection import select_merge_candidates


def test_non_straggler_tokens_are_excluded_even_with_low_importance():
    loads = torch.tensor([10.0, 4.0])
    expert_to_gpu = torch.tensor([0, 1])
    expert_assignment = torch.tensor([[0], [1], [0], [0]])
    importance = torch.tensor([0.1, 0.01, 0.9, 0.2])

    result = select_merge_candidates(
        loads,
        expert_assignment,
        importance,
        threshold=7.0,
        rho=0.67,
        expert_to_gpu=expert_to_gpu,
    )

    assert not result["candidate_mask"][1]
    assert result["candidate_mask"][0]
    assert result["candidate_mask"][3]


def test_high_importance_straggler_tokens_are_excluded():
    loads = torch.tensor([10.0, 4.0])
    expert_to_gpu = torch.tensor([0, 1])
    expert_assignment = torch.tensor([[0], [0], [0], [1]])
    importance = torch.tensor([0.1, 0.9, 0.2, 0.01])

    result = select_merge_candidates(
        loads,
        expert_assignment,
        importance,
        threshold=7.0,
        rho=0.67,
        expert_to_gpu=expert_to_gpu,
    )

    assert not result["candidate_mask"][1]
    assert result["candidate_mask"][0]
    assert result["candidate_mask"][2]


def test_candidate_count_tracks_rho_ratio():
    loads = torch.tensor([12.0])
    expert_to_gpu = torch.tensor([0])
    expert_assignment = torch.zeros(10, 1, dtype=torch.long)
    importance = torch.arange(10, dtype=torch.float32)

    result = select_merge_candidates(
        loads,
        expert_assignment,
        importance,
        threshold=7.0,
        rho=0.3,
        expert_to_gpu=expert_to_gpu,
    )

    assert result["candidate_indices"].numel() == 3
    assert result["candidate_indices"].tolist() == [0, 1, 2]

