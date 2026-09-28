from __future__ import annotations

import argparse
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import UNIVERSAL_PROMPT, atomic_json, read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "artifacts/manifests/frozen_causal_plan.json")
    parser.add_argument("--natural-results", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/manifests/causal_request_pool.json")
    parser.add_argument("--stream", type=Path, default=ROOT / "artifacts/manifests/causal_request_stream.json")
    args = parser.parse_args()
    natural = {row["request_id"]: row for row in read_jsonl(args.natural_results)}
    plan = read_json(args.plan)
    requests = []
    repeat_ids = set(plan["repeat_anchor_ids"])
    for anchor in plan["anchors"]:
        source = natural[anchor["original_request_id"]]
        forced = [int(value) for value in source["output_token_ids"][:128]]
        if not forced:
            continue
        conditions = [
            ("A", anchor["original_image_path"], anchor["original_image_sha256"]),
            ("B_same", anchor["same_domain_replacement_image_path"], anchor["same_domain_replacement_image_sha256"]),
            ("B_cross", anchor["cross_domain_replacement_image_path"], anchor["cross_domain_replacement_image_sha256"]),
        ]
        for condition, image_path, image_sha in conditions:
            requests.append({
                "request_id": f"{anchor['pair_id']}::{condition}::forced{len(forced)}",
                "pair_id": anchor["pair_id"],
                "condition": condition,
                "domain": anchor["domain"],
                "analysis_split": anchor["analysis_split"],
                "question": UNIVERSAL_PROMPT,
                "image_path": image_path,
                "image_sha256": image_sha,
                "forced_token_ids": forced,
                "natural_output_token_count": source["output_token_count"],
            })
        if anchor["pair_id"] in repeat_ids:
            for repeat in (0, 1):
                requests.append({
                    "request_id": f"{anchor['pair_id']}::A_repeat{repeat}::forced{len(forced)}",
                    "pair_id": anchor["pair_id"],
                    "condition": "A_repeat",
                    "repeat_id": repeat,
                    "domain": anchor["domain"],
                    "analysis_split": anchor["analysis_split"],
                    "question": UNIVERSAL_PROMPT,
                    "image_path": anchor["original_image_path"],
                    "image_sha256": anchor["original_image_sha256"],
                    "forced_token_ids": forced,
                    "natural_output_token_count": source["output_token_count"],
                })
    atomic_json(args.pool, {"schema_version": 1, "requests": requests})
    atomic_json(args.stream, {"schema_version": 1, "request_ids": [row["request_id"] for row in requests]})


if __name__ == "__main__":
    main()
