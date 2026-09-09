#!/usr/bin/env python3
"""Create exact-marginal screening schedules for a frozen request pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def chunks(rows: list[dict], size: int) -> list[list[str]]:
    assert len(rows) % size == 0
    return [[row["request_id"] for row in rows[index:index + size]]
            for index in range(0, len(rows), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.pool.read_text().splitlines()]
    ids = sorted(row["request_id"] for row in rows)
    waves = []
    for repeat in range(args.repetitions):
        policies = {
            "P0_global_fixed": list(rows),
            "P1_vision_bucket": sorted(rows, key=lambda row: (row["image_count"], row["total_pixels"], row["request_id"])),
            "P2_lm_length_bucket": sorted(rows, key=lambda row: (row["question_words"], row["question_chars"], row["request_id"])),
            "text_image_split": sorted(rows, key=lambda row: (bool(row["image_count"]), row["request_id"])),
            "single_multi_split": sorted(rows, key=lambda row: (min(row["image_count"], 2), row["request_id"])),
        }
        shuffled = list(rows)
        random.Random(args.seed + repeat).shuffle(shuffled)
        policies["RANDOM"] = shuffled
        for batch_size in args.batch_sizes:
            if batch_size > len(rows) or len(rows) % batch_size:
                continue
            for policy, ordered in policies.items():
                assert sorted(row["request_id"] for row in ordered) == ids
                for wave_index, request_ids in enumerate(chunks(ordered, batch_size)):
                    waves.append({
                        "label": f"r{repeat}:b{batch_size}:{policy}:w{wave_index}",
                        "repeat": repeat, "batch_size": batch_size, "policy": policy,
                        "wave_index": wave_index, "request_ids": request_ids,
                    })
    result = {
        "pool": str(args.pool.resolve()),
        "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
        "marginal_assertion": "Every policy/repeat/batch-size traverses the identical request multiset exactly once.",
        "waves": waves,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"waves": len(waves), "policies": sorted({wave["policy"] for wave in waves})}, indent=2))


if __name__ == "__main__":
    main()
