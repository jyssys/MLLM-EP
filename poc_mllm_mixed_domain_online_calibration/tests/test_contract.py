from __future__ import annotations

import numpy as np

from poc_mllm_mixed_domain_online_calibration.analysis import evaluate_stream
from poc_mllm_mixed_domain_online_calibration.contract import NUM_EXPERTS, NUM_LAYERS, owner_rank, warmup_count


def _sample(request_id: str, domain: str) -> dict:
    count = np.ones((NUM_LAYERS, NUM_EXPERTS))
    source = np.ones((NUM_LAYERS, 4, NUM_EXPERTS))
    target = {"count": count, "mass": count, "source": source, "observed_steps": [1]}
    return {
        "request_id": request_id,
        "domain": domain,
        "features": {
            "P_text": np.ones(4) / 2,
            "P_full": np.ones(4) / 2,
            "P_separated": np.ones(8) / np.sqrt(8),
            "P_vision": np.ones(4) / 2,
        },
        "decode": {window: target for window in ("1-32", "33-64", "65-96", "97-128", "1-128")},
    }


def test_balanced_rounding_and_owner_maps() -> None:
    assert [warmup_count(ratio, 40) for ratio in (0, .01, .05, .10, .20)] == [0, 1, 2, 4, 8]
    assert owner_rank(63, 2) == 0 and owner_rank(64, 2) == 1
    assert owner_rank(31, 4) == 0 and owner_rank(32, 4) == 1
    assert owner_rank(15, 8) == 0 and owner_rank(16, 8) == 1


def test_no_future_leakage_event_order() -> None:
    samples = {"past": _sample("past", "d"), "current": _sample("current", "d")}
    _, events = evaluate_stream(samples, ["current"], ["past"], "P_separated", "1-128", 4, True)
    assert events[0]["neighbor_request_ids"] == ["past"]
    assert not events[0]["current_truth_available_before_prediction"]
    assert events[0]["logical_prediction_timestamp"] < events[0]["logical_decode_completion_timestamp"]
    assert events[0]["memory_size_before_prediction"] == 1
    assert events[0]["memory_size_after_completion"] == 2


def test_current_request_in_memory_is_rejected() -> None:
    samples = {"current": _sample("current", "d")}
    try:
        evaluate_stream(samples, ["current"], ["current"], "P_full", "1-128", 4, False)
    except AssertionError as error:
        assert "leaked" in str(error)
    else:
        raise AssertionError("leak was accepted")
