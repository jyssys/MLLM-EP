import numpy as np

from poc_semantic_first_topk.policy_core import (
    FULL_K,
    ep_same_budget_refinement,
    keep_from_k,
    semantic_allocation,
)


def test_semantic_allocation_obeys_prefix_and_text_k8():
    risk = np.tile(np.arange(1, 9, dtype=float), (4, 1))
    modality = np.asarray(["vision", "text", "vision", "other"])
    allocation = semantic_allocation(risk, modality, 4, FULL_K)
    assert allocation.dropped == 4
    assert allocation.k[1] == 8 and allocation.k[3] == 8
    keep = keep_from_k(allocation.k)
    assert np.all(keep[:, :-1] >= keep[:, 1:])


def test_ep_refinement_preserves_budget_and_never_increases_max_load():
    ids = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7],
                      [32, 33, 34, 35, 36, 37, 38, 39],
                      [0, 32, 64, 96, 1, 33, 65, 97]], dtype=np.int64)
    risk = np.tile(np.linspace(.01, .08, 8), (3, 1))
    modality = np.asarray(["vision", "vision", "vision"])
    initial = np.asarray([8, 4, 4], dtype=np.int16)
    refined = ep_same_budget_refinement(ids, risk, modality, initial, risk_slack=1.0)
    assert refined.k.sum() == initial.sum()
    assert refined.final_load.max() <= refined.initial_load.max()
