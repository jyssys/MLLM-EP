#!/usr/bin/env python3
"""Create policy waves with actual-token-balanced, policy-independent DP assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def partition(ids: list[str], tokens: dict[str, int]) -> dict[str, list[str]]:
    # Deterministic membership independent of the policy's within-wave order.
    bins = [[], []]
    loads = [0, 0]
    for request_id in sorted(ids, key=lambda key: (-tokens[key], key)):
        rank = min(range(2), key=lambda index: (loads[index], index))
        bins[rank].append(request_id)
        loads[rank] += tokens[request_id]
    # Preserve each policy's original ordering inside its fixed membership.
    memberships = [set(value) for value in bins]
    ordered = [[request_id for request_id in ids if request_id in memberships[rank]] for rank in range(2)]
    return {"0": ordered[0], "1": ordered[1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.pool.read_text().splitlines()]
    tokens = json.loads(args.tokens.read_text())
    assert set(tokens) == {row["request_id"] for row in rows}
    ids = sorted(tokens)
    waves = []
    for repeat in range(args.repetitions):
        policies = {
            "P0_global_fixed": list(rows),
            "P1_vision_bucket": sorted(rows, key=lambda row: (row["image_count"], row["total_pixels"], row["request_id"])),
            "P2_lm_length_bucket": sorted(rows, key=lambda row: (tokens[row["request_id"]], row["request_id"])),
            "text_image_split": sorted(rows, key=lambda row: (bool(row["image_count"]), row["request_id"])),
            "single_multi_split": sorted(rows, key=lambda row: (min(row["image_count"], 2), row["request_id"])),
        }
        shuffled = list(rows); random.Random(args.seed + repeat).shuffle(shuffled)
        policies["RANDOM"] = shuffled
        for batch_size in args.batch_sizes:
            if batch_size > len(rows) or len(rows) % batch_size:
                continue
            for policy, ordered in policies.items():
                assert sorted(row["request_id"] for row in ordered) == ids
                for wave_index in range(0, len(ordered), batch_size):
                    wave = ordered[wave_index:wave_index + batch_size]
                    request_ids = [row["request_id"] for row in wave]
                    dp = partition(request_ids, tokens)
                    loads = {rank: sum(tokens[key] for key in members) for rank, members in dp.items()}
                    waves.append({
                        "label": f"r{repeat}:b{batch_size}:{policy}:w{wave_index // batch_size}",
                        "repeat": repeat, "batch_size": batch_size, "policy": policy,
                        "wave_index": wave_index // batch_size, "request_ids": request_ids,
                        "dp_request_ids": dp, "dp_token_loads": loads,
                        "dp_token_max_mean": max(loads.values()) / (sum(loads.values()) / 2),
                    })
    result = {
        "pool": str(args.pool.resolve()), "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
        "tokens": str(args.tokens.resolve()), "tokens_sha256": hashlib.sha256(args.tokens.read_bytes()).hexdigest(),
        "control": "Actual prompt tokens are greedily balanced; within-wave DP membership is independent of policy order.",
        "waves": waves,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"waves": len(waves), "max_dp_imbalance": max(w["dp_token_max_mean"] for w in waves)}, indent=2))


if __name__ == "__main__":
    main()
