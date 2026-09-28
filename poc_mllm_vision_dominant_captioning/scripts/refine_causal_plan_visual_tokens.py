from __future__ import annotations

import argparse
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import atomic_json, read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=ROOT / "artifacts/manifests/frozen_image_cohort.json")
    parser.add_argument("--initial-plan", type=Path, default=ROOT / "artifacts/manifests/frozen_causal_plan.json")
    parser.add_argument("--natural-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manifests/frozen_causal_plan_visual_token_matched.json")
    args = parser.parse_args()
    cohort = {row["request_id"]: row for row in read_json(args.cohort)["requests"]}
    observed = {row["request_id"]: int(row["vision_token_count"]) for row in read_jsonl(args.natural_results)}
    initial = read_json(args.initial_plan)
    domains = sorted({row["domain"] for row in cohort.values()})
    anchors = []
    for frozen in initial["anchors"]:
        anchor = cohort[frozen["original_request_id"]]
        target = observed[anchor["request_id"]]
        same = min(
            [
                row for row in cohort.values()
                if row["domain"] == anchor["domain"] and row["request_id"] != anchor["request_id"]
            ],
            key=lambda row: (abs(observed[row["request_id"]] - target), row["selection_key"]),
        )
        cross_domain = frozen["cross_domain"]
        cross = min(
            [row for row in cohort.values() if row["domain"] == cross_domain],
            key=lambda row: (abs(observed[row["request_id"]] - target), row["selection_key"]),
        )
        anchors.append({
            **frozen,
            "same_domain_replacement_request_id": same["request_id"],
            "same_domain_replacement_image_path": same["image_path"],
            "same_domain_replacement_image_sha256": same["image_sha256"],
            "cross_domain_replacement_request_id": cross["request_id"],
            "cross_domain_replacement_image_path": cross["image_path"],
            "cross_domain_replacement_image_sha256": cross["image_sha256"],
            "original_visual_token_count": target,
            "same_domain_replacement_visual_token_count": observed[same["request_id"]],
            "cross_domain_replacement_visual_token_count": observed[cross["request_id"]],
        })
    atomic_json(args.output, {
        "schema_version": 2,
        "frozen_before_routing_analysis": True,
        "selection_inputs": ["dataset domain", "image hash", "processed visual token count"],
        "selection_uses_router_trace": False,
        "selection_uses_generated_text_or_quality": False,
        "anchor_count": len(anchors),
        "same_domain_swap_count": len(anchors),
        "cross_domain_swap_count": len(anchors),
        "repeat_anchor_ids": initial["repeat_anchor_ids"],
        "matching_proxy": "absolute processed visual-token-count difference",
        "domains": domains,
        "anchors": anchors,
    })


if __name__ == "__main__":
    main()
