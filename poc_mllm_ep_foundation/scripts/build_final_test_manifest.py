"""Create a fresh image-disjoint final test after a predictor repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from poc_mllm_ep_foundation.constants import MODEL
from poc_mllm_ep_foundation.evaluation.workloads import load_manifest
from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_ep_foundation.scripts.build_anchor_manifest import (
    _area, _evenly, _image_key,
)
from poc_mllm_ep_foundation.scripts.run_full_model import _prepare


def _select(rows: list[dict], used: set[str], generation: int) -> list[dict]:
    unique = {}
    for row in rows:
        key = _image_key(row)
        if key not in used:
            unique.setdefault(key, row)
    ordered = sorted(unique.values(), key=lambda row: (_area(row), str(row["request_id"])))
    boundaries = (0, len(ordered) // 3, 2 * len(ordered) // 3, len(ordered))
    selected = []
    for bin_index, count in enumerate((3, 3, 2)):
        pool = ordered[boundaries[bin_index]:boundaries[bin_index + 1]]
        for row in _evenly(pool, count):
            item = dict(row)
            item.update({"anchor_split": "untouched_test",
                         "selection_visual_bin": ("low", "mid", "high")[bin_index],
                         "image_area": _area(row),
                         "final_test_generation": generation})
            selected.append(item)
            used.add(_image_key(row))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmstar", type=Path, required=True)
    parser.add_argument("--chartqa", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--generation", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument(
        "--skip-processing-metadata", action="store_true",
        help=("defer grid/token metadata to the physical runner; useful when "
              "the source manifests already contain the frozen image identity"),
    )
    args = parser.parse_args()
    excluded = [row for path in args.exclude
                for row in load_manifest(path)["requests"]]
    used = {_image_key(row) for row in excluded}
    rows = _select(load_manifest(args.mmstar)["requests"], used, args.generation)
    rows += _select(load_manifest(args.chartqa)["requests"], used, args.generation)
    if args.skip_processing_metadata:
        for row in rows:
            row["image_sha256"] = _image_key(row)
            row["processing_metadata_status"] = "DEFERRED_TO_PHYSICAL_RUNNER"
        rows.sort(key=lambda row: (row["task"], row["image_area"], row["request_id"]))
    else:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        for row in rows:
            _, meta = _prepare(processor, row)
            row["image_sha256"] = _image_key(row)
            row["image_grid_thw"] = meta["image_grid_thw"]
            row["vision_token_count"] = meta["vision_token_count"]
            row["prompt_token_count"] = meta["prompt_token_count"]
        rows.sort(key=lambda row: (row["task"], row["vision_token_count"], row["request_id"]))
    if len(rows) != 16 or len({_image_key(row) for row in rows}) != 16:
        raise AssertionError("fresh final test must contain 16 distinct images")
    payload = {
        "schema_version": f"mllm_ep_anchor_final_test.v{args.generation}",
        "split_counts": {"untouched_test": 16},
        "selection": ("fresh image-disjoint task-balanced area-stratified "
                      f"generation {args.generation}"),
        "excluded_manifests": [str(path.resolve()) for path in args.exclude],
        "requests": rows,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "requests": len(rows),
                      "ids": [row["request_id"] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
