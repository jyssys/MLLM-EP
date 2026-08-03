import pytest
import torch

from method2.importance import compute_cross_attention_importance, split_key_redundant


def test_raw_cross_attention_importance_matches_text_mean():
    attention = torch.zeros(5, 5)
    attention[0, 2:5] = torch.tensor([0.1, 0.2, 0.7])
    attention[1, 2:5] = torch.tensor([0.3, 0.2, 0.5])

    importance = compute_cross_attention_importance(attention, [0, 1], [2, 3, 4])

    assert torch.allclose(importance, torch.tensor([0.2, 0.2, 0.6]))


def test_key_redundant_split_uses_top_rho():
    importance = torch.tensor([0.1, 0.8, 0.4, 0.7, 0.2])

    key_mask, redundant_mask, key_indices = split_key_redundant(importance, rho_key=0.2)

    assert key_indices.tolist() == [1]
    assert key_mask.tolist() == [False, True, False, False, False]
    assert redundant_mask.sum().item() == 4


def test_derope_and_cls_are_phase2_only():
    attention = torch.eye(3)
    with pytest.raises(NotImplementedError):
        compute_cross_attention_importance(attention, [0], [1], derope=True)
    with pytest.raises(NotImplementedError):
        compute_cross_attention_importance(attention, [0], [1], lambda_cls=0.1)

