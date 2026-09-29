from __future__ import annotations

import pytest

from poc_mllm_ep_collective_granularity.contract import (
    chunk_label,
    engine_token_budget,
    layered_stage_count,
)
from poc_mllm_ep_collective_granularity.scripts.analyze import (
    bootstrap,
    timeline_index,
)


def test_chunk_contract() -> None:
    assert chunk_label(0) == "FULL"
    assert chunk_label(512) == "CP512"
    assert engine_token_budget(512) == 516
    assert engine_token_budget(0) == 20_000


def test_exact_upstream_stage_policy() -> None:
    assert layered_stage_count(512) == 1
    assert layered_stage_count(1024) == 2
    assert layered_stage_count(2029) == 4
    assert layered_stage_count(4096) == 8
    assert layered_stage_count(4097) == 16
    with pytest.raises(ValueError):
        layered_stage_count(2029, stage_cap=8)


def test_bootstrap_keeps_paired_request_unit() -> None:
    result = bootstrap([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["ci95"][0] <= 2.0 <= result["ci95"][1]


def test_outer_and_internal_iteration_are_not_conflated() -> None:
    assert timeline_index({"iteration": 31, "outer_iteration": 19}) == 19
    assert timeline_index({"iteration": 31}) == 31
