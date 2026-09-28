#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from poc_mllm_vision_conditioned_decode_damage_skipping.contract import atomic_json, read_jsonl


def summarize(rows: list[dict], skipped: np.ndarray, request_index: dict[str, int]) -> dict:
    before, after = Counter(), Counter()
    remote_before = remote_after = 0
    active_before = {rank: set() for rank in range(8)}
    active_after = {rank: set() for rank in range(8)}
    for index, row in enumerate(rows):
        expert = int(row["expert_id"])
        owner = expert // 16
        # Frozen TP2-DP4 structural source placement: TP0 source of DP=i%4.
        source = 2 * (request_index[row["request_id"]] % 4)
        before[owner] += 1
        active_before[owner].add(expert)
        remote_before += owner != source
        if not skipped[index]:
            after[owner] += 1
            active_after[owner].add(expert)
            remote_after += owner != source
    before_values = np.asarray([before[rank] for rank in range(8)], dtype=float)
    after_values = np.asarray([after[rank] for rank in range(8)], dtype=float)
    logical_bytes_per_path = 2 * 2048 * 2  # dispatch hidden + combine contribution, BF16.
    return {
        "paths_before": int(len(rows)),
        "paths_after": int((~skipped).sum()),
        "skipped_paths": int(skipped.sum()),
        "path_reduction": float(skipped.mean()),
        "assignments_per_rank_before": [int(value) for value in before_values],
        "assignments_per_rank_after": [int(value) for value in after_values],
        "active_experts_per_rank_before": [len(active_before[rank]) for rank in range(8)],
        "active_experts_per_rank_after": [len(active_after[rank]) for rank in range(8)],
        "rank_max_mean_before": float(before_values.max() / max(before_values.mean(), 1e-12)),
        "rank_max_mean_after": float(after_values.max() / max(after_values.mean(), 1e-12)),
        "remote_paths_before": int(remote_before),
        "remote_paths_after": int(remote_after),
        "removed_remote_paths": int(remote_before - remote_after),
        "logical_remote_dispatch_combine_bytes_removed": int((remote_before - remote_after) * logical_bytes_per_path),
        "byte_metric": "LOGICAL_BF16_PAYLOAD_PROXY_NO_DEEPEP_PADDING_OR_METADATA",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.labels) if row["split"] == "heldout"]
    request_ids = sorted({row["request_id"] for row in rows})
    request_index = {request: index for index, request in enumerate(request_ids)}
    prediction_rows = list(read_jsonl(args.predictions))
    lookup = {
        (row["request_id"], row["layer"], row["decode_step"], row["expert_id"], row["family"]): row["predicted_damage"]
        for row in prediction_rows
    }
    payload = {
        "status": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "scope": "BOUNDED_HELDOUT_FUNCTIONAL_PATH_SAMPLE",
        "mapping": "linear_128_experts_16_per_rank",
        "source_placement": "TP2_DP4_TP0_SOURCE_RANK_2_TIMES_FROZEN_REQUEST_INDEX_MOD4",
        "latency": None, "tpot": None, "e2e": None, "speedup": None,
        "policies": {},
    }
    for family in ("F1_MODES_STYLE_GENERIC", "F5_VISION_SEPARATED"):
        values = np.asarray([
            lookup[(row["request_id"], row["layer"], row["decode_step"], row["expert_id"], family)]
            for row in rows
        ])
        skipped = np.zeros(len(rows), dtype=bool)
        skipped[np.argsort(values, kind="stable")[: int(np.floor(0.20 * len(rows)))]] = True
        payload["policies"][family] = summarize(rows, skipped, request_index)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
