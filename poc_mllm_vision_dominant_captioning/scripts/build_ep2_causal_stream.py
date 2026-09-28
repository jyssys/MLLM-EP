from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/manifests/causal_request_pool.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manifests/frozen_ep2_causal_stream.json")
    args = parser.parse_args()
    rows = read_json(args.pool)["requests"]
    by_domain = defaultdict(list)
    for row in rows:
        if row["condition"] in ("A", "B_cross"):
            by_domain[row["domain"]].append(row)
    ids = []
    for domain in sorted(by_domain):
        pairs = sorted({row["pair_id"] for row in by_domain[domain]})[:1]
        for pair in pairs:
            for condition in ("A", "B_cross"):
                ids.append(next(row["request_id"] for row in by_domain[domain] if row["pair_id"] == pair and row["condition"] == condition))
    atomic_json(args.output, {
        "schema_version": 1,
        "request_count": len(ids),
        "frozen_before_ep2_routes": True,
        "composition": "1 causal anchor/domain x A and cross-domain replacement",
        "request_ids": ids,
    })


if __name__ == "__main__":
    main()
