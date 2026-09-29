#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from poc_mllm_rank_aware_active_expert_reduction.contract import atomic_json


def _jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--prior-causal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=8)
    parser.add_argument("--heldout", type=int, default=24)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))["requests"]
    natural = {row["request_id"]: row for row in _jsonl_gz(args.natural)}
    prior = json.loads(args.prior_causal.read_text(encoding="utf-8"))
    prior_ids = set()
    for row in prior["anchors"]:
        prior_ids.update({
            row["original_request_id"], row["same_domain_replacement_request_id"],
            row["cross_domain_replacement_request_id"],
        })
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cohort:
        request_id = row["request_id"]
        meta = natural.get(request_id)
        if request_id in prior_ids or meta is None or int(meta["output_token_count"]) < 128:
            continue
        by_domain[row["domain"]].append({
            **row,
            "vision_token_count": int(meta["vision_token_count"]),
            "forced_token_ids": [int(value) for value in meta["output_token_ids"][:128]],
            "natural_output_token_count": int(meta["output_token_count"]),
        })
    for rows in by_domain.values():
        rows.sort(key=lambda row: (row["selection_key"], row["request_id"]))
    domains = sorted(by_domain)
    needed = args.calibration + args.heldout
    anchors = []
    cursors = {domain: 0 for domain in domains}
    while len(anchors) < needed:
        for domain in domains:
            if len(anchors) >= needed:
                break
            rows = by_domain[domain]
            index = cursors[domain]
            if index >= len(rows):
                raise RuntimeError(f"insufficient fresh rows in {domain}")
            anchors.append(rows[index])
            cursors[domain] += 1
    anchor_ids = {row["request_id"] for row in anchors}
    replacement_used: set[str] = set()
    pairs = []
    for index, anchor in enumerate(anchors):
        candidates = [
            row for row in by_domain[anchor["domain"]]
            if row["request_id"] not in anchor_ids and row["request_id"] not in replacement_used
        ]
        replacement = min(candidates, key=lambda row: (
            abs(row["vision_token_count"] - anchor["vision_token_count"]),
            row["selection_key"],
        ))
        replacement_used.add(replacement["request_id"])
        split = "calibration" if index < args.calibration else "heldout"
        pairs.append({
            "pair_id": f"commit::{index:02d}::{anchor['request_id']}",
            "split": split, "domain": anchor["domain"],
            "original_request_id": anchor["request_id"],
            "original_image_path": anchor["image_path"],
            "original_image_sha256": anchor["image_sha256"],
            "original_visual_token_count": anchor["vision_token_count"],
            "counterfactual_request_id": replacement["request_id"],
            "counterfactual_image_path": replacement["image_path"],
            "counterfactual_image_sha256": replacement["image_sha256"],
            "counterfactual_visual_token_count": replacement["vision_token_count"],
            "visual_token_count_delta": abs(
                replacement["vision_token_count"] - anchor["vision_token_count"]
            ),
            "question": anchor["question"],
            "forced_token_ids": anchor["forced_token_ids"],
            "selection_uses_routing": False,
            "selection_uses_generated_quality": False,
        })
    # Four same-image controls estimate the deterministic/instrumentation floor.
    noise = []
    for index, anchor in enumerate(anchors[:4]):
        noise.append({
            "pair_id": f"repeat::{index:02d}::{anchor['request_id']}",
            "split": "repeat_noise", "domain": anchor["domain"],
            "original_request_id": anchor["request_id"],
            "original_image_path": anchor["image_path"],
            "original_image_sha256": anchor["image_sha256"],
            "original_visual_token_count": anchor["vision_token_count"],
            "counterfactual_request_id": anchor["request_id"],
            "counterfactual_image_path": anchor["image_path"],
            "counterfactual_image_sha256": anchor["image_sha256"],
            "counterfactual_visual_token_count": anchor["vision_token_count"],
            "visual_token_count_delta": 0, "question": anchor["question"],
            "forced_token_ids": anchor["forced_token_ids"],
            "selection_uses_routing": False,
            "selection_uses_generated_quality": False,
        })
    payload = {
        "schema_version": "visual_commitment_cohort.v1",
        "frozen_before_commitment_measurement": True,
        "source_cohort": str(args.cohort.resolve()),
        "source_natural": str(args.natural.resolve()),
        "excluded_prior_causal_request_ids": sorted(prior_ids),
        "selection_contract": "ROUND_ROBIN_DOMAIN_BALANCED_STABLE_KEY_OUTPUT_LENGTH_GE_128",
        "calibration_count": args.calibration, "heldout_count": args.heldout,
        "repeat_noise_count": len(noise), "domains": domains,
        "pairs": pairs + noise,
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        "calibration": args.calibration, "heldout": args.heldout,
        "repeat_noise": len(noise), "domains": domains,
    }, indent=2))


if __name__ == "__main__":
    main()
