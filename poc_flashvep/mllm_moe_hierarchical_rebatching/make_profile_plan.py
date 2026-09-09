#!/usr/bin/env python3
"""Pair frozen requests across DP ranks for per-request module profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.pool.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) % 2:
        raise ValueError("Profile pool must have an even request count")
    waves = []
    for index in range(0, len(rows), 2):
        waves.append({
            "label": f"profile:{index // 2:03d}", "policy": "single_request_profile",
            "repeat": 0, "batch_size": 2, "wave_index": index // 2,
            "request_ids": [rows[index]["request_id"], rows[index + 1]["request_id"]],
        })
    result = {
        "pool": str(args.pool.resolve()),
        "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
        "marginal_assertion": "Each selected request appears exactly once; one request per DP rank.",
        "waves": waves,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"requests": len(rows), "waves": len(waves)}, indent=2))


if __name__ == "__main__":
    main()
