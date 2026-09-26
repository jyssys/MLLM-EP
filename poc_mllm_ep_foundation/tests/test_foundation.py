from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from poc_mllm_ep_foundation.contract import parse_physical_gpus
from poc_mllm_ep_foundation.evaluation.metrics import chartqa_relaxed_correct, extract_mcq_letter
from poc_mllm_ep_foundation.io import append_jsonl_atomic, completed_ids
from poc_mllm_ep_foundation.virtual_ep.cost import NearestShapeCostModel
from poc_mllm_ep_foundation.virtual_ep.mapper import (
    expert_owner_map,
    source_rank,
    source_rank_for_target,
    validate_topk,
)
from poc_mllm_ep_foundation.scripts.analyze_physical_anchors_v4 import (
    FEATURE_NAMES,
    _request_arrays,
    _request_repeat_variability,
)
from poc_mllm_ep_foundation.scripts.predict_replay_v4 import _features, _predict
from poc_mllm_ep_foundation.scripts.project_ep8_sensitivity import _transfer_factors


def test_allowed_physical_gpu_subset() -> None:
    assert parse_physical_gpus("4,5") == [4, 5]
    assert parse_physical_gpus("4") == [4]
    with pytest.raises(ValueError):
        parse_physical_gpus("0,4")


def test_owner_maps() -> None:
    assert np.bincount(expert_owner_map(128, 2)).tolist() == [64] * 2
    assert np.bincount(expert_owner_map(128, 4)).tolist() == [32] * 4
    assert np.bincount(expert_owner_map(128, 8)).tolist() == [16] * 8
    assert [source_rank(dp, tp) for dp in range(2) for tp in range(2)] == [0, 1, 2, 3]
    assert [source_rank_for_target(1, tp, 2, 7) for tp in range(2)] == [0, 1]
    assert [source_rank_for_target(1, tp, 4, 7) for tp in range(2)] == [2, 3]
    assert [source_rank_for_target(0, tp, 8, 7) for tp in range(2)] == [6, 7]


def test_route_contract_and_duplicate_rejection() -> None:
    validate_topk(np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int16))
    with pytest.raises(ValueError):
        validate_topk(np.array([[0, 0]], dtype=np.int16))
    with pytest.raises(ValueError):
        validate_topk(np.array([[128]], dtype=np.int16))


def test_chartqa_and_mcq_metrics() -> None:
    assert chartqa_relaxed_correct("105", "100")
    assert not chartqa_relaxed_correct("106", "100")
    assert chartqa_relaxed_correct("yes", "Yes")
    assert extract_mcq_letter("The answer is C.") == "C"


def test_resume_is_atomic_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    append_jsonl_atomic(path, [{"request_id": "b", "v": 1}, {"request_id": "a", "v": 1}])
    append_jsonl_atomic(path, [{"request_id": "b", "v": 2}])
    assert completed_ids(path) == {"a", "b"}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"request_id": "a", "v": 1}, {"request_id": "b", "v": 2}]


def test_stale_calibration_rejected_and_serial_sum_not_exposed() -> None:
    stale = {"signature": {"hidden_size": 4096, "intermediate_size": 1024,
                            "num_experts": 128, "top_k": 8}, "observations": []}
    with pytest.raises(ValueError):
        NearestShapeCostModel(stale)
    good = {"signature": {"hidden_size": 2048, "intermediate_size": 768,
                           "num_experts": 128, "top_k": 8}, "observations": []}
    model = NearestShapeCostModel(good)
    assert not hasattr(model, "serial_rank_sum")


def test_virtual_cost_features_use_destination_assignment_load() -> None:
    assignment = np.array([[8, 0], [8, 0]], dtype=int)
    row = {
        "source_logical_rows": [1, 1],
        "assignment_matrix": assignment.tolist(),
        "unique_activation_matrix": [[1, 0], [1, 0]],
        "expert_histogram": [1] * 128,
        "layer": 0,
    }
    features = _features(row, 2)
    maximum = features[FEATURE_NAMES.index("log_assignments_max")]
    imbalance = features[FEATURE_NAMES.index("assignments_cv")]
    assert maximum == pytest.approx(np.log1p(16))
    assert imbalance == pytest.approx(1.0)


def test_request_target_uses_complete_repeat_totals() -> None:
    samples = [
        {
            "split": "development", "request_group": "a+b",
            "targets": np.array([100.0, 0.0, 0.0]),
            "repeat_targets": {
                0: np.array([0.0, 0.0, 0.0]),
                1: np.array([100.0, 0.0, 0.0]),
                2: np.array([100.0, 0.0, 0.0]),
            },
        },
        {
            "split": "development", "request_group": "a+b",
            "targets": np.array([100.0, 0.0, 0.0]),
            "repeat_targets": {
                0: np.array([100.0, 0.0, 0.0]),
                1: np.array([0.0, 0.0, 0.0]),
                2: np.array([100.0, 0.0, 0.0]),
            },
        },
    ]
    prediction = np.zeros((2, 3), dtype=float)
    legacy, _, _ = _request_arrays(samples, prediction, "development")
    coherent, _, _ = _request_arrays(
        samples, prediction, "development", coherent_repeat_target=True
    )
    assert legacy.tolist() == [200.0]
    assert coherent.tolist() == [100.0]
    variability = _request_repeat_variability(samples, "development")
    assert variability["request_groups"] == 1
    assert variability["repeat_count_min"] == 3


def test_ep8_sensitivity_factors_come_from_blind_transfer() -> None:
    transfer = {
        "request_values": {"untouched_test": [
            {"actual_ep4_ms": actual, "blind_ep2_model_prediction_ms": 10.0}
            for actual in (8.0, 10.0, 12.0, 14.0)
        ]}
    }
    factors = _transfer_factors(transfer)
    assert factors["endpoint_optimistic"] < factors["calibrated_base"]
    assert factors["calibrated_base"] < factors["concurrency_stressed"]


def test_componentwise_cost_state_selects_each_stage() -> None:
    children = []
    for values in ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]):
        children.append({
            "type": "standardized_ridge", "feature_indices": [0],
            "mean": [0.0], "scale": [1.0],
            "coef": [values, [0.0, 0.0, 0.0]],
        })
    state = {"type": "componentwise", "component_models": children}
    assert _predict(state, np.array([123.0])).tolist() == [1.0, 5.0, 9.0]
