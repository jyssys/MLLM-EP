import numpy as np

from poc_semantic_first_topk.recovery_checks import (
    chartqa_relaxed,
    optimal_refinement,
    semantically_equivalent,
    virtual_rank_load,
)


def test_chartqa_and_semantic_equivalence_do_not_require_token_identity():
    assert chartqa_relaxed("100", "104")
    assert semantically_equivalent("2.870", "2.87")
    assert not semantically_equivalent("1.87", "2.87")


def test_optimal_refinement_preserves_work_and_balances_virtual_ranks():
    ids = np.asarray([
        [0, 1, 2, 3, 64, 65, 66, 67],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [64, 65, 66, 67, 68, 69, 70, 71],
    ], dtype=np.int64)
    risk = np.ones_like(ids, dtype=np.float64)
    modality = np.asarray(["vision", "vision", "vision"])
    semantic_k = np.asarray([8, 8, 1], dtype=np.int16)
    refined_k, metadata = optimal_refinement(
        ids, risk, modality, semantic_k, ep_size=2, risk_slack=0.0)
    semantic_load = virtual_rank_load(ids, semantic_k, 2)
    refined_load = virtual_rank_load(ids, refined_k, 2)
    assert refined_k.sum() == semantic_k.sum()
    assert metadata["risk_ratio"] <= 1.0 + 1e-9
    assert semantic_load.tolist() == [12, 5]
    assert refined_load.max() == 9
