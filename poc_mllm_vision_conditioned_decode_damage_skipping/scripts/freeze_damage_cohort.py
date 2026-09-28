#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from poc_mllm_vision_conditioned_decode_damage_skipping.contract import atomic_json, read_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-per-domain", type=int, default=4)
    args = parser.parse_args()

    pool = read_json(args.pool)
    by_id = {row["request_id"]: row for row in pool["requests"]}
    results = {row["request_id"]: row for row in read_jsonl(args.results)}
    warmup = read_json(args.warmup)["ratios"]["0.10"]
    chosen: list[dict] = []
    by_domain: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"calibration": [], "heldout": []})

    for domain in sorted(warmup):
        calibration = list(warmup[domain]["request_ids"])
        heldout = [
            row["request_id"] for row in pool["requests"]
            if row["domain"] == domain and row["analysis_split"] == "heldout"
            and results[row["request_id"]]["output_token_count"] >= 48
        ][: args.heldout_per_domain]
        if len(calibration) != 4 or len(heldout) != args.heldout_per_domain:
            raise RuntimeError((domain, len(calibration), len(heldout)))
        for split, identifiers in (("calibration", calibration), ("heldout", heldout)):
            for request_id in identifiers:
                source = by_id[request_id]
                result = results[request_id]
                tokens = [int(value) for value in result["output_token_ids"][:128]]
                if len(tokens) < 32:
                    raise RuntimeError(f"{request_id} has only {len(tokens)} natural tokens")
                row = {
                    "pair_id": request_id,
                    "request_id": request_id,
                    "domain": domain,
                    "damage_split": split,
                    # The shared bounded capture runner filters on this field;
                    # damage_split remains the scientific split of record.
                    "analysis_split": "discovery",
                    "question": source["question"],
                    "original_image_path": source["image_path"],
                    "original_image_sha256": source["image_sha256"],
                    "forced_token_ids": tokens,
                    "natural_output_token_count": int(result["output_token_count"]),
                    "finish_reason": result["finish_reason"],
                    "source_analysis_split": source["analysis_split"],
                }
                chosen.append(row)
                by_domain[domain][split].append(request_id)

    hashes = [row["original_image_sha256"] for row in chosen]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("damage cohort is not image-hash disjoint")
    atomic_json(args.output, {
        "schema_version": 1,
        "selection_frozen_before_damage_capture": True,
        "routing_or_damage_truth_used_for_selection": False,
        "calibration_contract": "10_PERCENT_PER_DOMAIN_FROM_FROZEN_WARMUP",
        "heldout_selection": "FIRST_FROZEN_POOL_ORDER_WITH_NATURAL_OUTPUT_GE48",
        "domains": sorted(by_domain),
        "by_domain": by_domain,
        "anchors": chosen,
        "request_count": len(chosen),
        "calibration_count": sum(row["damage_split"] == "calibration" for row in chosen),
        "heldout_count": sum(row["damage_split"] == "heldout" for row in chosen),
    })
    print(json.dumps({"requests": len(chosen), "domains": by_domain}, indent=2))


if __name__ == "__main__":
    main()
