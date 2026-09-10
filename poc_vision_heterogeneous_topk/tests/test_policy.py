import numpy as np

from poc_vision_heterogeneous_topk.analyze_policy_oracle import (
    budget_keep,
    random_keep,
    semantic_keep,
    tail_aware_keep,
)


def fixture():
    ids = np.array([[0, 1, 32, 33, 64, 65, 96, 97],
                    [2, 3, 34, 35, 66, 67, 98, 99]])
    weights = np.array([[.40, .20, .10, .08, .07, .06, .05, .04],
                        [.35, .25, .12, .09, .07, .05, .04, .03]])
    modality = np.array(["vision", "text"])
    return ids, weights, modality


def test_exact_budget_and_text_preserved():
    ids, weights, modality = fixture()
    for fn in (lambda: random_keep(modality, 4, 7),
               lambda: semantic_keep(weights, modality, 4),
               lambda: tail_aware_keep(ids, weights, modality, 4, .02)):
        keep = fn()
        assert keep[0].sum() == 4
        assert keep[1].sum() == 8


def test_semantic_keeps_largest_weights():
    _, weights, modality = fixture()
    keep = semantic_keep(weights, modality, 4)
    assert keep[0].tolist() == [True, True, True, True, False, False, False, False]


def test_budget_policies_match_exact_work_and_keep_top1():
    ids = np.tile(np.arange(8), (10, 1)) * 16
    weights = np.tile(np.linspace(.3, .01, 8), (10, 1))
    modality = np.array(["vision"] * 10)
    for policy in ("random", "semantic", "tail"):
        keep = budget_keep(ids, weights, modality, .2, policy)
        assert (~keep).sum() == 16
        assert keep[:, 0].all()
