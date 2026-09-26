"""Interleave a promoted development cohort with a fresh untouched cohort."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from poc_mllm_ep_foundation.evaluation.workloads import load_manifest
from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_ep_foundation.scripts.build_anchor_manifest import _image_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--untouched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    development = [
        {**row, "anchor_split": "development"}
        for row in load_manifest(args.development)["requests"]
    ]
    untouched = [
        {**row, "anchor_split": "untouched_test"}
        for row in load_manifest(args.untouched)["requests"]
    ]
    if len(development) % 2 or len(untouched) % 2:
        raise ValueError("both cohorts must contain complete DP2 request pairs")
    development_images = {_image_key(row) for row in development}
    untouched_images = {_image_key(row) for row in untouched}
    overlap = development_images & untouched_images
    if overlap:
        raise ValueError(f"image leakage across revalidation splits: {len(overlap)}")

    queues = {
        "development": deque(
            development[index:index + 2]
            for index in range(0, len(development), 2)
        ),
        "untouched_test": deque(
            untouched[index:index + 2]
            for index in range(0, len(untouched), 2)
        ),
    }
    rows = []
    schedule = []
    while queues["development"] or queues["untouched_test"]:
        for split in ("development", "untouched_test"):
            if not queues[split]:
                continue
            pair = queues[split].popleft()
            schedule.append({
                "wave": len(schedule), "split": split,
                "request_ids": [str(row["request_id"]) for row in pair],
            })
            rows.extend(pair)

    atomic_json(args.output, {
        "schema_version": "mllm_ep_cost_revalidation_manifest.v1",
        "development_source": str(args.development.resolve()),
        "untouched_source": str(args.untouched.resolve()),
        "split_counts": {
            "development": len(development),
            "untouched_test": len(untouched),
        },
        "execution_order": "alternating DP2 pairs from development and untouched cohorts",
        "execution_schedule": schedule,
        "requests": rows,
    })
    print(json.dumps({
        "output": str(args.output), "requests": len(rows),
        "split_counts": {
            "development": len(development),
            "untouched_test": len(untouched),
        },
        "waves": len(schedule),
    }, indent=2))


if __name__ == "__main__":
    main()
