from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from poc_mllm_rank_aware_active_expert_reduction.decode_modality_discovery_campaign.contract import (
    CONFIRMATION_REQUESTS,
    DISCOVERY_REQUESTS,
    PHYSICAL_GPUS,
    TOP_K,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/manifests/frozen_campaign_cohort.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_only_authorized_physical_gpus() -> None:
    assert PHYSICAL_GPUS == (2, 3, 6, 7)


def test_frozen_cohort_is_fresh_and_image_disjoint() -> None:
    payload = load_manifest()
    cases = payload["cases"]
    counts = Counter(case["split"] for case in cases)
    assert counts == {"discovery": DISCOVERY_REQUESTS, "confirmation": CONFIRMATION_REQUESTS}
    hashes = [case["original_image_sha256"] for case in cases]
    assert len(hashes) == len(set(hashes))
    discovery = {case["original_image_sha256"] for case in cases if case["split"] == "discovery"}
    confirmation = {case["original_image_sha256"] for case in cases if case["split"] == "confirmation"}
    assert discovery.isdisjoint(confirmation)
    assert all(value >= 0 for value in payload["prior_exclusion_hash_counts"].values())
    assert payload["frozen_before_routing_or_damage"] is True
    assert payload["confirmation_frozen_before_primary_selection"] is True


def test_causal_conditions_share_frozen_decode_sequence() -> None:
    for case in load_manifest()["cases"]:
        token_ids = case["forced_token_ids"]
        digest = hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()
        assert digest == case["forced_prefix_sha256"]
        assert len(token_ids) >= 116
        assert case["question"]


def test_probe_contract_uses_original_top8_without_replacement() -> None:
    assert TOP_K == 8
    source = (ROOT / "structured_probe.py").read_text(encoding="utf-8")
    assert "retained_mass" in source
    assert "weights[row, slot] /= retained_mass" in source
    assert "removed_expert_ids" in source


def test_confirmation_is_not_used_to_select_primary() -> None:
    frozen = ROOT / "artifacts/manifests/frozen_primary_candidate.json"
    if not frozen.exists():
        return
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    assert payload["frozen_before_confirmation_trace"] is True
    assert payload["confirmation_data_inspected"] is False
    assert payload["primary_hypothesis"].startswith("H")


def test_final_verdict_obeys_independent_gates() -> None:
    path = ROOT / "artifacts/final_verdict.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gates"] == {
        "G1_modality": True,
        "G2_skip_utility": False,
        "G3_robustness": True,
    }
    assert payload["verdict"] == "MODALITY_EFFECT_NOT_USEFUL_FOR_SKIPPING"
    assert payload["true_ep4_runtime_promoted"] is False


def test_required_summary_fields_are_present() -> None:
    path = ROOT / "reports/campaign_summary.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for field in (
        "CAMPAIGN_GPU_SET", "DISCOVERY_REQUESTS", "CONFIRMATION_REQUESTS",
        "H1_VISUAL_CONDITIONED_REDUNDANCY", "H2_TEMPORAL_LAYER_PERSISTENCE",
        "H3_EXPERT_CIRCUIT_STRUCTURE", "H4_MODALITY_RANK_OCCUPANCY",
        "H5_ROUTER_GEOMETRY", "PRIMARY_DISCOVERED_SIGNAL",
        "PRIMARY_MODALITY_EFFECT", "PRIMARY_SKIP_GAIN",
        "PRIMARY_RANK_CLOSURE_GAIN", "PRIMARY_HELDOUT_CI",
        "TRUE_EP4_RUNTIME_STATUS", "FINAL_VERDICT", "NEXT_ACTION",
    ):
        assert f"{field}:" in text
