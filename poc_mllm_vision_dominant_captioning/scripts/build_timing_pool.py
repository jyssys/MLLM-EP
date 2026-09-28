from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import DOMAINS, UNIVERSAL_PROMPT, atomic_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-results", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/manifests/frozen_b8_timing_pool.json")
    parser.add_argument("--stream", type=Path, default=ROOT / "artifacts/manifests/frozen_b8_timing_stream.json")
    args = parser.parse_args()
    by_domain = defaultdict(list)
    for row in read_jsonl(args.natural_results):
        if row["output_token_count"] >= 128:
            by_domain[row["domain"]].append(row)
    selected = []
    cursor = {domain: 0 for domain in DOMAINS}
    while len(selected) < 16:
        progressed = False
        for domain in DOMAINS:
            if cursor[domain] < len(by_domain[domain]) and len(selected) < 16:
                selected.append(by_domain[domain][cursor[domain]])
                cursor[domain] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) < 8:
        raise RuntimeError("fewer than 8 natural responses have 128 tokens")
    primary = selected[:16]
    requests = []
    for warmup, rows in ((True, primary[:8]), (False, primary)):
        for row in rows:
            requests.append({
                "request_id": f"{row['request_id']}::fixed128::{('warmup' if warmup else 'measured')}",
                "source_request_id": row["request_id"],
                "domain": row["domain"],
                "analysis_split": row["analysis_split"],
                "question": UNIVERSAL_PROMPT,
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "forced_token_ids": row["output_token_ids"][:128],
                "is_warmup": warmup,
            })
    atomic_json(args.pool, {
        "schema_version": 1,
        "selection_uses_routing": False,
        "selection_criterion": "natural output has at least 128 tokens; round-robin domains",
        "batch_frozen_before_timing": True,
        "batch_size": 8,
        "requests": requests,
    })
    atomic_json(args.stream, {"schema_version": 1, "request_ids": [row["request_id"] for row in requests]})


if __name__ == "__main__":
    main()
