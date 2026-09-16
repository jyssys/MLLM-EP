from dataclasses import replace

import numpy as np
import pytest

from virtual_ep.event_simulator import EventPolicy, critical_path_ms
from virtual_ep.mapping import ExpertOwnership, SourcePartition
from virtual_ep.schema import SCHEMA_VERSION, TraceBundle
from virtual_ep.simulator import simulate_trace, write_predictions
from virtual_ep.traffic import build_traffic
from virtual_ep.discovery_trace import (
    DISCOVERY_SCHEMA_VERSION,
    DiscoveryTrace,
    POSITION_CLASSES,
    from_heavy_trace,
)


def make_trace():
    expert_ids = np.asarray(
        [[0, 1], [2, 3], [4, 5], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]],
        dtype=np.int16,
    )
    rows = {
        "request_id": np.zeros(8, dtype=np.int32),
        "block_id": np.zeros(8, dtype=np.int16),
        "iteration_id": np.zeros(8, dtype=np.int16),
        "nfe": np.zeros(8, dtype=np.int32),
        "layer_id": np.zeros(8, dtype=np.int16),
        "token_position": np.arange(8, dtype=np.int16),
        "source_partition_key": np.arange(8, dtype=np.int16),
        "is_masked": np.ones(8, dtype=bool),
        "expert_ids": expert_ids,
        "router_weights": np.full_like(expert_ids, 0.5, dtype=np.float16),
    }
    iterations = {
        "request_id": np.asarray([0]),
        "block_id": np.asarray([0]),
        "iteration_id": np.asarray([0]),
        "nfe": np.asarray([0]),
        "masked_before": np.asarray([8]),
        "accepted": np.asarray([2]),
        "masked_after": np.asarray([6]),
        "confidence_mean": np.asarray([0.9]),
        "confidence_min": np.asarray([0.8]),
        "confidence_max": np.asarray([1.0]),
        "terminated": np.asarray([False]),
    }
    return TraceBundle(
        rows,
        iterations,
        {
            "schema_version": SCHEMA_VERSION,
            "physical_row_semantics": "vanilla_full_rows",
            "num_routed_experts": 8,
            "hidden_size": 4,
        },
    )


def test_ownership_and_source_partition():
    ownership = ExpertOwnership(256, 4)
    assert ownership.experts_per_rank == 64
    assert ownership.owner(np.asarray([0, 63, 64, 255])).tolist() == [0, 0, 1, 3]
    assert SourcePartition(4).ranks(np.arange(8)).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert SourcePartition(4).ranks(np.arange(10)).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]


def test_assignment_and_unique_payload_are_distinct():
    ownership = ExpertOwnership(8, 2)
    routes = np.asarray([[0, 1], [4, 5], [0, 4], [1, 5]])
    traffic = build_traffic(routes, np.arange(4), ownership, SourcePartition(2), 4)
    assert traffic.assignment_matrix.tolist() == [[2, 2], [2, 2]]
    assert traffic.unique_activation_matrix.tolist() == [[1, 1], [2, 2]]
    assert traffic.remote_assignments == 4
    assert traffic.remote_unique_activations == 3
    assert traffic.logical_dispatch_bytes == 3 * 4 * 2


def test_trace_roundtrip_and_structural_simulation(tmp_path):
    trace = make_trace()
    path = tmp_path / "trace.npz"
    trace.save(path)
    loaded = TraceBundle.load(path)
    predictions = simulate_trace(loaded, ep_size=2)
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.physical_rows == 8
    assert prediction.max_rank_load == 8
    assert prediction.moe_stage_ms is None
    assert not hasattr(prediction, "replicated_state_bridge_allgather_ms")


def test_event_policy_uses_only_measured_overlap_fraction():
    result = critical_path_ms(2.0, [3.0, 4.0], 1.0, EventPolicy(0.25, 0.5))
    assert result == 1.5 + 4.0 + 0.5


def test_timing_output_requires_ep2_calibration_verdict(tmp_path):
    prediction = simulate_trace(make_trace(), ep_size=2)[0]
    timed = replace(
        prediction,
        dispatch_ms=1.0,
        critical_expert_ms=2.0,
        combine_ms=1.0,
        moe_stage_ms=4.0,
    )
    with pytest.raises(ValueError, match="held-out EP2 validation"):
        write_predictions([timed], tmp_path / "invalid")
    write_predictions(
        [timed],
        tmp_path / "valid",
        timing_verdict="EP2-CALIBRATED",
    )


def test_discovery_trace_round_trip(tmp_path):
    n = 2
    arrays = {
        "request_id": np.asarray([0, 1], np.int32),
        "block_id": np.zeros(n, np.int16),
        "iteration_id": np.arange(n, dtype=np.int16),
        "nfe": np.arange(n, dtype=np.int32),
        "layer_id": np.ones(n, np.int16),
        "prompt_tokens": np.full(n, 10, np.int32),
        "physical_rows": np.full(n, 42, np.int32),
        "masked_current_block": np.full(n, 32, np.int16),
        "decoded_current_block": np.zeros(n, np.int16),
        "newly_accepted_current_block": np.zeros(n, np.int16),
        "accepted_this_iteration": np.ones(n, np.int16),
        "remaining_mask_after": np.full(n, 31, np.int16),
        "normalized_block_progress": np.zeros(n, np.float32),
        "confidence_mean": np.ones(n, np.float32),
        "confidence_min": np.ones(n, np.float32),
        "confidence_max": np.ones(n, np.float32),
        "terminated": np.zeros(n, bool),
        "expert_counts_by_class": np.zeros(
            (n, len(POSITION_CLASSES), 256), np.uint16
        ),
        "current_expert_ids": np.zeros((n, 32, 8), np.int16),
    }
    trace = DiscoveryTrace(
        arrays,
        {
            "schema_version": DISCOVERY_SCHEMA_VERSION,
            "model": "test",
            "revision": "x",
            "threshold": 0.95,
            "block_length": 32,
            "top_k": 8,
            "hidden_size": 2048,
        },
    )
    trace.validate()
    path = tmp_path / "discovery.npz"
    trace.save(path)
    loaded = DiscoveryTrace.load(path)
    np.testing.assert_array_equal(
        loaded.arrays["request_id"], arrays["request_id"]
    )


def test_heavy_conversion_keeps_prompt_tail_out_of_decoded_state():
    # The first physical generation block overlaps the unaligned prompt tail.
    # Those prompt rows must remain PROMPT_PREFIX across every refinement.
    physical = 4
    iterations = 2
    rows = {
        "request_id": np.repeat(np.asarray([0, 0], np.int32), physical),
        "block_id": np.zeros(physical * iterations, np.int16),
        "iteration_id": np.repeat(np.arange(iterations, dtype=np.int16), physical),
        "nfe": np.repeat(np.arange(iterations, dtype=np.int32), physical),
        "layer_id": np.ones(physical * iterations, np.int16),
        "token_position": np.tile(np.arange(physical, dtype=np.int32), iterations),
        "source_partition_key": np.tile(np.arange(physical, dtype=np.int32), iterations),
        "is_masked": np.asarray([False, False, False, True,
                                 False, False, False, False]),
        "expert_ids": np.zeros((physical * iterations, 8), np.int16),
        "router_weights": np.full((physical * iterations, 8), 1 / 8, np.float32),
    }
    iteration_rows = {
        "request_id": np.zeros(iterations, np.int32),
        "block_id": np.zeros(iterations, np.int16),
        "iteration_id": np.arange(iterations, dtype=np.int16),
        "nfe": np.arange(iterations, dtype=np.int32),
        "masked_before": np.asarray([1, 1], np.int16),
        "accepted": np.asarray([0, 1], np.int16),
        "masked_after": np.asarray([1, 0], np.int16),
        "confidence_mean": np.ones(iterations, np.float32),
        "confidence_min": np.ones(iterations, np.float32),
        "confidence_max": np.ones(iterations, np.float32),
        "terminated": np.asarray([False, True]),
    }
    heavy = TraceBundle(rows, iteration_rows, {
        "schema_version": SCHEMA_VERSION,
        "physical_row_semantics": "vanilla_full_rows",
        "num_routed_experts": 256,
        "hidden_size": 4,
        "block_length": 2,
        "top_k": 8,
    })
    converted = from_heavy_trace(heavy, {0: 3})
    np.testing.assert_array_equal(
        converted.arrays["current_position_class"],
        np.asarray([[0, 2], [0, 3]], dtype=np.int8),
    )
