from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_causal_plan_frozen_and_balanced() -> None:
    payload = json.loads((ROOT / "artifacts/manifests/frozen_causal_plan.json").read_text())
    assert payload["frozen_before_routing_analysis"] is True
    assert payload["anchor_count"] == 20
    domains = {}
    for row in payload["anchors"]:
        domains[row["domain"]] = domains.get(row["domain"], 0) + 1
        assert row["original_image_sha256"] != row["same_domain_replacement_image_sha256"]
        assert row["original_image_sha256"] != row["cross_domain_replacement_image_sha256"]
    assert set(domains.values()) == {4}

