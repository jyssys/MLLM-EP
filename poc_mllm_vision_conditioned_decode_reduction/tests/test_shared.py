from __future__ import annotations

from poc_mllm_vision_conditioned_decode_reduction.shared import (
    batch_regime_sweep,
    compact_decode_histogram,
    is_tp2_dummy_route,
    owner_rank,
    structural_ep_projection,
)


def test_owner_maps_and_dummy_contract() -> None:
    assert owner_rank(0, 2) == 0
    assert owner_rank(63, 2) == 0
    assert owner_rank(64, 2) == 1
    assert owner_rank(31, 4) == 0
    assert owner_rank(32, 4) == 1
    assert owner_rank(127, 8) == 7
    assert is_tp2_dummy_route([list(range(8))], [[0.125] * 8])
    assert not is_tp2_dummy_route([list(range(1, 9))], [[0.125] * 8])
    row = {"expert_count": {str(index): 1 for index in range(8)} | {"63": 1}}
    cleaned = compact_decode_histogram(row, tp2_dummy_copies=1)
    assert cleaned.sum() == 1
    assert cleaned[63] == 1


def test_structural_projection_deduplicates_consolidation_and_never_times_ep8() -> None:
    rows = [
        {
            "request_id": "r0",
            "step": 1,
            "layer": 0,
            "token_id": "t0",
            "expert_id": 1,
            "executed_expert_id": 1,
            "source_rank": 0,
        },
        {
            "request_id": "r0",
            "step": 1,
            "layer": 0,
            "token_id": "t0",
            "expert_id": 33,
            "executed_expert_id": 1,
            "source_rank": 0,
        },
        {
            "request_id": "r0",
            "step": 1,
            "layer": 0,
            "token_id": "t0",
            "expert_id": 127,
            "executed_expert_id": None,
            "source_rank": 0,
        },
    ]
    result = structural_ep_projection(rows, ep_size=8)
    assert result["evidence_label"] == "STRUCTURAL_EP8_PROJECTION_ONLY"
    assert result["baseline_expert_paths"] == 3
    assert result["residual_expert_paths"] == 1
    assert result["eliminated_expert_paths"] == 2
    assert result["baseline_remote_assignments"] == 2
    assert result["residual_remote_assignments"] == 0
    assert result["latency_ms"] is None
    assert result["tpot_gain"] is None
    assert result["e2e_gain"] is None


def test_structural_projection_refuses_to_invent_remote_bytes() -> None:
    result = structural_ep_projection(
        [{"request_id": "r", "step": 1, "layer": 0, "expert_id": 5}],
        ep_size=8,
    )
    assert result["source_rank_status"] == "UNAVAILABLE"
    assert result["baseline_remote_assignments"] is None
    assert result["baseline_logical_dispatch_bytes"] is None


def test_batch_sweep_removes_audited_dummy_and_marks_unsupported_batch() -> None:
    rows = []
    for pair, expert in (("a", 63), ("b", 64)):
        count = {str(index): 1 for index in range(8)}
        count[str(expert)] = count.get(str(expert), 0) + 1
        rows.append(
            {
                "record_type": "decode",
                "pair_id": pair,
                "layer": 0,
                "step": 1,
                "expert_count": count,
            }
        )
        for layer in range(1, 48):
            rows.append(
                {
                    "record_type": "decode",
                    "pair_id": pair,
                    "layer": layer,
                    "step": 1,
                    "expert_count": count,
                }
            )
    result = batch_regime_sweep(rows, ["a", "b"], batch_sizes=(1, 2, 4))
    assert result["results"]["1"]["mean_active_expert_union"] == 1
    assert result["results"]["2"]["mean_active_expert_union"] == 2
    assert result["results"]["4"]["status"] == "INSUFFICIENT_REQUEST_POOL"
