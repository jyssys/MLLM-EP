import numpy as np

from poc_semantic_first_topk.policy_core import spatial_schedule_k


def test_schedule_k_maps_spatial_quadrants_and_preserves_text():
    modality = np.asarray(["vision"] * 16 + ["text"])
    coords = np.asarray([(row, col) for row in range(4) for col in range(4)] + [(-1, -1)])
    group_k = list(range(1, 9)) + list(range(8, 0, -1))
    result = spatial_schedule_k(modality, coords, group_k)
    assert result[:16].tolist() == group_k
    assert result[-1] == 8
