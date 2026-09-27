from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poc_mllm_vision_conditioned_decode_reduction.track_a import (
    HIDDEN_SIZE,
    PathVector,
    TokenGroup,
    audit_existing_evidence,
    load_capture_groups,
    oracle_frontier,
)


def _group(index: int = 0, scale: float = 1e-4) -> TokenGroup:
    base = np.linspace(-1.0, 1.0, HIDDEN_SIZE, dtype=np.float32)
    paths = []
    for expert in range(8):
        # Secondary outputs remain very close to primary expert zero.  The
        # synthetic case verifies accounting and the pre-frozen gate only.
        output = base + (scale * expert)
        paths.append(PathVector(expert, 0.20 - 0.01 * expert, output))
    return TokenGroup(
        f"request-{index}", "discovery", 0, 1, f"hash-{index}", tuple(paths)
    )


def test_oracle_frontier_accounts_actual_paths() -> None:
    result = oracle_frontier([_group(index) for index in range(10)])
    point = next(row for row in result["frontier"] if row["requested_reduction"] == 0.2)
    assert point["consolidated_paths"] == 16
    assert point["actual_path_reduction"] == 0.2
    assert 0.0 <= point["active_expert_union_reduction"] <= 1.0
    assert result["gate_pass"] is True
    assert result["evidence_label"] == "DIAGNOSTIC_VECTOR_ORACLE_NOT_FULL_MODEL_QUALITY"


def test_missing_vectors_are_not_promoted(tmp_path: Path) -> None:
    trace = tmp_path / "artifacts" / "trace.jsonl"
    trace.parent.mkdir()
    trace.write_text(json.dumps({
        "route_ids": [[1, 2, 3, 4, 5, 6, 7, 8]],
        "route_weights": [[0.2] * 8],
    }) + "\n")
    audit = audit_existing_evidence(tmp_path)
    assert audit["route_ids_available"] is True
    assert audit["expert_output_vectors_available"] is False
    assert audit["status"] == "BOUNDED_VECTOR_CAPTURE_REQUIRED"


def test_shared_npz_capture_is_recombined_across_ep_ranks(tmp_path: Path) -> None:
    fingerprints = np.asarray(["fingerprint"])
    expert_ids = np.arange(8, dtype=np.int16).reshape(1, 8)
    weights = np.linspace(0.2, 0.1, 8, dtype=np.float32).reshape(1, 8)
    np.savez_compressed(
        tmp_path / "router.request.rank0.layer4.step1.call0.npz",
        fingerprints=fingerprints,
        hidden_states=np.zeros((1, HIDDEN_SIZE), dtype=np.float16),
        token_ids=np.asarray([-1]),
        topk_ids=expert_ids,
        topk_weights=weights,
    )
    for rank in range(4):
        mask = np.zeros((1, 8), dtype=bool)
        mask[:, rank * 2 : rank * 2 + 2] = True
        outputs = np.zeros((1, 8, HIDDEN_SIZE), dtype=np.float16)
        for slot in range(rank * 2, rank * 2 + 2):
            outputs[0, slot] = slot + 1
        # One receiver rank intentionally uses a shifted call-counter step.
        # The source-side router row remains the canonical step.
        receiver_step = 9 if rank == 3 else 1
        np.savez_compressed(
            tmp_path / f"experts.request.rank{rank}.layer4.step{receiver_step}.call0.npz",
            fingerprints=fingerprints,
            expert_ids=expert_ids,
            router_weights=weights,
            local_mask=mask,
            raw_outputs=outputs,
        )
    groups, integrity = load_capture_groups(tmp_path)
    assert integrity["shards"] == 4
    assert integrity["source_fingerprints"] == 1
    assert integrity["complete_top8_groups"] == 1
    assert integrity["step_mismatch_shards"] == 1
    assert integrity["step_mismatch_rows"] == 1
    assert groups[0].step == 1
    assert [path.expert_id for path in groups[0].paths] == list(range(8))
