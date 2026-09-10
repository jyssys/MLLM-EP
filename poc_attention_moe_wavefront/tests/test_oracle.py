from poc_attention_moe_wavefront.wavefront.oracle import (
    SplitPoint,
    amdahl_adjusted_ttft,
    candidate_splits,
    compute_oracles,
    ideal_wave_ms,
)


def point(request: str, layer: int, split: int, wave_parts: tuple[float, float, float, float]) -> SplitPoint:
    return SplitPoint(request, layer, 100, split, *wave_parts)


def test_oracle_formula() -> None:
    assert ideal_wave_ms(2, 8, 5, 3) == 13


def test_candidates_include_boundary_neighbors_and_are_valid() -> None:
    values = candidate_splits(1000, [400])
    assert {144, 272, 400, 528, 656}.issubset(values)
    assert values == sorted(set(values))
    assert min(values) >= 1 and max(values) <= 999


def test_amdahl_clamps_noisy_stage_sum_to_observed_ttft() -> None:
    oracle, gain = amdahl_adjusted_ttft(100, 120, 80)
    assert oracle == 60
    assert gain == 40


def test_hierarchical_oracles_choose_expected_splits() -> None:
    points = [
        point("v", 0, 25, (1, 9, 2, 8)),
        point("v", 0, 50, (3, 7, 7, 3)),
        point("v", 1, 25, (2, 8, 2, 8)),
        point("v", 1, 50, (3, 7, 6, 4)),
        point("t", 0, 25, (1, 9, 2, 8)),
        point("t", 0, 50, (3, 7, 7, 3)),
    ]
    result = compute_oracles(
        points,
        {("v", 0): 20, ("v", 1): 20, ("t", 0): 20},
        {"v": 49, "t": None},
    )
    assert result["O1_choice"][("v", 0)].split == 50
    assert result["O3_choice"][("v", 0)].split == 50
    assert "t" not in result["O3"]

