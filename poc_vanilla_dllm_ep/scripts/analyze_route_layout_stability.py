#!/usr/bin/env python3
"""Quantify exact EP layout stability, distinct from expert-set stability."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank0", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.rank0.read_text())
    world = int(payload["topology"]["ep"])
    per_expert = 128 // world
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in payload["stage_rows"]:
        if row.get("forward_phase") != "refine" or "selected_experts" not in row:
            continue
        key = (str(row["request_id"]), int(row["block_id"]), int(row["layer"]))
        groups[key].append(row)
    rank_counts_equal = []
    ordered_owner_equal = []
    owner_multiset_equal = []
    stable_owner_branch = []
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["denoising_iteration"]))
        for current, previous in zip(rows[1:], rows[:-1]):
            rank_counts_equal.append(
                current["rank_assignment_counts"] == previous["rank_assignment_counts"]
            )
            for cur, prev in zip(current["selected_experts"], previous["selected_experts"]):
                cur_owner = [int(expert) // per_expert for expert in cur]
                prev_owner = [int(expert) // per_expert for expert in prev]
                ordered_owner_equal.append(cur_owner == prev_owner)
                owner_multiset_equal.append(sorted(cur_owner) == sorted(prev_owner))
                stable_owner_branch.append(
                    sum(a == b for a, b in zip(cur_owner, prev_owner)) / len(cur_owner)
                )
    result = {
        "world_size": world,
        "lag1_layer_iteration_pairs": len(rank_counts_equal),
        "lag1_token_pairs": len(ordered_owner_equal),
        "exact_rank_count_vector_equal_fraction": statistics.mean(rank_counts_equal),
        "ordered_owner_vector_equal_fraction": statistics.mean(ordered_owner_equal),
        "owner_multiset_equal_fraction": statistics.mean(owner_multiset_equal),
        "same_topk_slot_owner_fraction": statistics.mean(stable_owner_branch),
        "interpretation": (
            "These are exact metadata/layout stability metrics only. Stable owner "
            "metadata does not make hidden payloads or expert outputs reusable."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
