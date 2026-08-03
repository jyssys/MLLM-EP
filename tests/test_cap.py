import torch

from method2.cap import cap_merge_count, find_capped_merge_ratio


def test_cap_stops_when_straggler_reaches_second_place_load():
    loads = torch.tensor([10.0, 7.0, 4.0])

    result = cap_merge_count(loads, straggler_gpu=0, candidate_count=10, max_merge_ratio=1.0)

    assert result["actual_merge_count"] == 3
    assert result["straggler_load_after"] == 7.0


def test_cap_uses_less_merge_than_uncapped_overmerge():
    loads = torch.tensor([10.0, 7.0])

    result = cap_merge_count(loads, straggler_gpu=0, candidate_count=10, max_merge_ratio=1.0)

    uncapped_merge_count = 10
    assert result["actual_merge_count"] < uncapped_merge_count


def test_generic_rho_scan_finds_first_ratio_at_second_place():
    loads = torch.tensor([10.0, 7.0])

    result = find_capped_merge_ratio(
        loads,
        straggler_gpu=0,
        max_ratio=1.0,
        load_after_merge=lambda rho: 10.0 - 10.0 * rho,
        steps=10,
    )

    assert result["rho_eff"] == 0.3
    assert result["load_after"] == 7.0

