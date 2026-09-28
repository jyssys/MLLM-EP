from __future__ import annotations

import hashlib
import json
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import (
    DOMAINS,
    NUM_EXPERTS,
    UNIVERSAL_PROMPT,
    decode_window,
    owner_rank,
)

ROOT = Path(__file__).resolve().parents[1]


def test_model_shape_and_ep_owners() -> None:
    assert NUM_EXPERTS == 128
    for ep_size in (2, 4, 8):
        owners = [owner_rank(expert, ep_size) for expert in range(NUM_EXPERTS)]
        assert set(owners) == set(range(ep_size))
        assert all(owners.count(rank) == NUM_EXPERTS // ep_size for rank in range(ep_size))
    assert hashlib.sha256((ROOT / "research_spec.md").read_bytes()).hexdigest() == (
        "f725909d6c4238ae169b2c285c84e69916e2b1aa7eb8a64f669a16139b63f859"
    )


def test_decode_windows_cover_256() -> None:
    assert decode_window(1) == "1-32"
    assert decode_window(128) == "97-128"
    assert decode_window(129) == "129-192"
    assert decode_window(256) == "193-256"


def test_frozen_cohort_contract() -> None:
    path = ROOT / "artifacts/manifests/frozen_image_cohort.json"
    payload = json.loads(path.read_text())
    assert payload["selection_frozen_before_routing"] is True
    assert payload["prompt_sha256"] == hashlib.sha256(UNIVERSAL_PROMPT.encode()).hexdigest()
    assert len(payload["requests"]) == 200
    assert len({row["image_sha256"] for row in payload["requests"]}) == 200
    assert {row["domain"] for row in payload["requests"]} == set(DOMAINS)
    for domain in DOMAINS:
        rows = [row for row in payload["requests"] if row["domain"] == domain]
        assert len(rows) == 40
        assert all(row["question"] == UNIVERSAL_PROMPT for row in rows)


def test_ep8_is_structural_only() -> None:
    forbidden = {"latency", "tpot", "e2e", "speedup"}
    path = ROOT / "artifacts/ep8_structural_projection.json"
    if path.exists():
        lowered = path.read_text().lower()
        assert all(f'"{word}' not in lowered for word in forbidden)


def test_measured_execution_budget_and_mixed_fifo_order() -> None:
    ledger = ROOT / "artifacts/execution_ledger.jsonl"
    if ledger.exists():
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line]
        assert sum(int(row["request_executions"]) for row in rows) <= 400

    locality_path = ROOT / "artifacts/batch_locality.json"
    if not locality_path.exists():
        return
    locality = json.loads(locality_path.read_text())
    fifo = [item for wave in locality["8"]["assignments"]["S0_FIFO"] for item in wave]
    frozen = json.loads((ROOT / "artifacts/manifests/frozen_iid_stream.json").read_text())["request_ids"]
    fifo_set = set(fifo)
    assert fifo == [request for request in frozen if request in fifo_set]

    cohort = json.loads((ROOT / "artifacts/manifests/frozen_image_cohort.json").read_text())
    domains = {row["request_id"]: row["domain"] for row in cohort["requests"]}
    assert len({domains[request] for request in locality["8"]["assignments"]["S0_FIFO"][0]}) > 1
