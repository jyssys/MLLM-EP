from __future__ import annotations

import json
import gzip
from pathlib import Path

import numpy as np

from poc_mllm_vision_conditioned_decode_reduction.track_b import (
    RoutePoint,
    analyze_npz_vector_capture,
    load_shared_npz_capture,
    load_active_tp0_routes,
    load_compact_routes,
    recurrence_events,
    route_recurrence_ep8_projection,
)


def test_sparse_recurrence_uses_actual_step_distance() -> None:
    points = [
        RoutePoint("r", "discovery", 1, 0, (0, 1, 2, 3, 4, 5, 6, 7), (0.125,) * 8),
        RoutePoint("r", "discovery", 2, 0, (0, 1, 8, 9, 10, 11, 12, 13), (0.125,) * 8),
        RoutePoint("r", "discovery", 8, 0, (0, 8, 16, 17, 18, 19, 20, 21), (0.125,) * 8),
    ]
    events = recurrence_events(points)
    w1 = [row for row in events if row.window == 1]
    assert len(w1) == 8
    assert sum(row.recurred for row in w1) == 2
    w8_step8 = [row for row in events if row.window == 8 and row.step == 8]
    assert len(w8_step8) == 8
    assert {row.expert_id: row.distance for row in w8_step8 if row.recurred} == {0: 6, 8: 6}


def test_raw_loader_excludes_condition_b_and_tp1(tmp_path: Path) -> None:
    pool = {"r": "heldout"}
    root = tmp_path / "trace"
    root.mkdir()
    common = {
        "pair_id": "r", "condition": "A", "decode_step": 1, "layer": 0,
        "route_ids": [[11, 12, 13, 14, 15, 16, 17, 18]],
        "route_weights": [[0.125] * 8],
    }
    rows = [
        {**common, "tp_rank": 0},
        {**common, "tp_rank": 1, "route_ids": [[0, 1, 2, 3, 4, 5, 6, 7]]},
        {**common, "tp_rank": 0, "condition": "B", "route_ids": [[20, 21, 22, 23, 24, 25, 26, 27]]},
    ]
    (root / "route_summary.pid1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    points = load_active_tp0_routes([root], pool)
    assert len(points) == 1
    assert points[0].ids[0] == 11


def test_committed_compact_route_export(tmp_path: Path) -> None:
    path = tmp_path / "condition_a_tp0_routes.jsonl.gz"
    row = {
        "pair_id": "r", "split": "heldout", "step": 7, "layer": 3,
        "route_ids": [11, 12, 13, 14, 15, 16, 17, 18],
        "route_weights": [0.125] * 8,
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    points = load_compact_routes(path, {"r": "heldout"})
    assert len(points) == 1
    assert points[0].step == 7
    assert points[0].ids[-1] == 18


def test_ep8_projection_never_claims_latency() -> None:
    points = [
        RoutePoint("r", "discovery", 1, 0, (0, 16, 32, 48, 64, 80, 96, 112), (0.125,) * 8),
        RoutePoint("r", "discovery", 2, 0, (0, 16, 33, 49, 65, 81, 97, 113), (0.125,) * 8),
    ]
    result = route_recurrence_ep8_projection(recurrence_events(points), window=1)
    assert result["evidence_label"] == "STRUCTURAL_EP8_PROJECTION_ONLY"
    assert result["latency"] == "FORBIDDEN_NOT_COMPUTED"
    assert result["recurrent_paths"] == 2


def _write_npz_capture(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True)
    requests = [
        ("discovery_request_0", "discovery"),
        ("discovery_request_1", "discovery"),
        ("discovery_request_2", "discovery"),
        ("discovery_request_3", "discovery"),
        ("heldout_request", "heldout"),
    ]
    (root / "manifest.json").write_text(json.dumps({
        "requests": [
            {"pair_id": request, "analysis_split": split}
            for request, split in requests
        ]
    }), encoding="utf-8")
    ids = np.asarray([[0, 16, 32, 48, 64, 80, 96, 112]], dtype=np.int16)
    weights = np.full((1, 8), 0.125, dtype=np.float32)
    for request, _split in requests:
        if request == "discovery_request_0":
            np.savez_compressed(
                raw / f"router.{request}.rank0.layer4.step99.call0.npz",
                fingerprints=np.asarray([f"prefill_{index}" for index in range(200)]),
                hidden_states=np.zeros((200, 2048), dtype=np.float16),
                topk_ids=np.tile(ids, (200, 1)),
                topk_weights=np.tile(weights, (200, 1)),
            )
        for step in (1, 2):
            fingerprint = np.asarray([f"{request}_step{step}"])
            np.savez_compressed(
                raw / f"router.{request}.rank0.layer4.step{step}.call0.npz",
                fingerprints=fingerprint,
                hidden_states=np.full((1, 2048), step, dtype=np.float16),
                topk_ids=ids,
                topk_weights=weights,
            )
            for rank in range(4):
                mask = np.zeros((2, 8), dtype=bool)
                mask[0, rank * 2:(rank + 1) * 2] = True
                receiver_ids = np.vstack([ids, np.arange(8, dtype=np.int16)[None, :]])
                receiver_weights = np.vstack([weights, np.full((1, 8), 0.125, np.float32)])
                outputs = np.zeros((2, 8, 2048), dtype=np.float16)
                for slot in range(rank * 2, (rank + 1) * 2):
                    outputs[0, slot] = 1.0
                    if step == 2 and slot >= 2:
                        outputs[0, slot] = 1.5
                np.savez_compressed(
                    # Receiver call counters are deliberately offset.  The
                    # loader must recover canonical step from the router-side
                    # fingerprint rather than this filename.
                    raw / f"experts.{request}.rank{rank}.layer4.step{step + 5}.call0.npz",
                    fingerprints=np.asarray([fingerprint[0], f"dummy_{request}_{step}_{rank}"]),
                    expert_ids=receiver_ids,
                    router_weights=receiver_weights,
                    local_mask=mask,
                    raw_outputs=outputs,
                )


def test_shared_npz_temporal_oracle_filters_dummy_receiver_rows(tmp_path: Path) -> None:
    _write_npz_capture(tmp_path)
    pool = {
        "discovery_request_0": "discovery",
        "discovery_request_1": "discovery",
        "discovery_request_2": "discovery",
        "discovery_request_3": "discovery",
        "heldout_request": "heldout",
    }
    tokens, integrity = load_shared_npz_capture(tmp_path, pool)
    assert len(tokens) == 10
    assert integrity["filtered_non_source_receiver_rows"] == 40
    assert integrity["complete_top8_tokens"] == 10
    assert integrity["discarded_non_decode_router_rows"] == 200
    assert integrity["receiver_filename_step_mismatch_rows"] == 40

    result = analyze_npz_vector_capture(tmp_path, pool)
    assert result["status"] == "MEASURED_BOUNDED_VECTOR_ORACLE"
    assert result["gate"]["applicable"] is True
    assert result["gate"]["best_window"] == 1
    assert result["gate"]["best_reuse_rate"] == 0.25
    assert result["gate"]["pass_20_percent"] is True
