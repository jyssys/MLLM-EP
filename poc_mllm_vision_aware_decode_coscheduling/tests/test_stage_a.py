from __future__ import annotations

from poc_mllm_vision_aware_decode_coscheduling.contract import owner_rank
from poc_mllm_vision_aware_decode_coscheduling.stage_a import exact_partition


def test_owner_map() -> None:
    assert [owner_rank(value) for value in (0, 31, 32, 63, 64, 95, 96, 127)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_exact_partition_has_full_coverage() -> None:
    requests = [f"r{index}" for index in range(8)]
    preferred = {"r0", "r1", "r2", "r3"}

    def cost(group: tuple[str, ...]) -> float:
        overlap = len(set(group) & preferred)
        return float(min(overlap, 4 - overlap))

    _cost, groups = exact_partition(requests, cost)
    flattened = [request for group in groups for request in group]
    assert len(groups) == 2
    assert all(len(group) == 4 for group in groups)
    assert sorted(flattened) == requests
    assert len(set(flattened)) == len(requests)
