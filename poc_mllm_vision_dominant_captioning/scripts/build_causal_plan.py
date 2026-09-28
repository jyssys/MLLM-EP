from __future__ import annotations

import argparse
import math
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import DOMAINS, atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]


def _distance(left: dict, right: dict) -> float:
    a = max(int(left["image_width"]) * int(left["image_height"]), 1)
    b = max(int(right["image_width"]) * int(right["image_height"]), 1)
    return abs(math.log(a) - math.log(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=ROOT / "artifacts/manifests/frozen_image_cohort.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manifests/frozen_causal_plan.json")
    args = parser.parse_args()
    rows = read_json(args.cohort)["requests"]
    by_domain = {domain: [row for row in rows if row["domain"] == domain] for domain in DOMAINS}
    anchors = []
    for domain_index, domain in enumerate(DOMAINS):
        candidates = sorted(
            [row for row in by_domain[domain] if row["analysis_split"] == "discovery"],
            key=lambda row: row["selection_key"],
        )[:4]
        cross_domain = DOMAINS[(domain_index + 1) % len(DOMAINS)]
        for anchor in candidates:
            same = min(
                [row for row in by_domain[domain] if row["request_id"] != anchor["request_id"]],
                key=lambda row: (_distance(anchor, row), row["selection_key"]),
            )
            cross = min(
                by_domain[cross_domain],
                key=lambda row: (_distance(anchor, row), row["selection_key"]),
            )
            anchors.append({
                "pair_id": anchor["request_id"],
                "domain": domain,
                "analysis_split": anchor["analysis_split"],
                "original_request_id": anchor["request_id"],
                "original_image_path": anchor["image_path"],
                "original_image_sha256": anchor["image_sha256"],
                "same_domain_replacement_request_id": same["request_id"],
                "same_domain_replacement_image_path": same["image_path"],
                "same_domain_replacement_image_sha256": same["image_sha256"],
                "cross_domain": cross_domain,
                "cross_domain_replacement_request_id": cross["request_id"],
                "cross_domain_replacement_image_path": cross["image_path"],
                "cross_domain_replacement_image_sha256": cross["image_sha256"],
                "selection_uses_routing": False,
                "selection_uses_generated_output": False,
            })
    atomic_json(args.output, {
        "schema_version": 1,
        "frozen_before_routing_analysis": True,
        "anchor_count": len(anchors),
        "same_domain_swap_count": len(anchors),
        "cross_domain_swap_count": len(anchors),
        "repeat_anchor_ids": [row["pair_id"] for row in anchors[::4]],
        "matching_proxy": "absolute log raw-image-area difference; actual visual token count reported post hoc",
        "anchors": anchors,
    })


if __name__ == "__main__":
    main()
