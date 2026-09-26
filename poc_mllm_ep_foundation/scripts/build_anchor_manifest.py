from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from transformers import AutoProcessor

from poc_mllm_ep_foundation.constants import MODEL
from poc_mllm_ep_foundation.evaluation.workloads import load_manifest
from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_ep_foundation.scripts.run_full_model import _prepare


SPLIT_COUNTS = {
    "calibration": (6, 5, 5),
    "development": (3, 3, 2),
    "untouched_test": (2, 3, 3),
}


def _image_key(row: dict[str, Any]) -> str:
    value = row.get("image_sha256")
    if value:
        return str(value)
    return hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()


def _area(row: dict[str, Any]) -> int:
    with Image.open(row["image_path"]) as raw:
        image = ImageOps.exif_transpose(raw)
        return int(image.width * image.height)


def _evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError((count, len(rows)))
    if count == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))]
            for index in range(count)]


def _select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(_image_key(row), row)
    ordered = sorted(unique.values(), key=lambda row: (_area(row), str(row["request_id"])))
    bins = [ordered[: len(ordered) // 3],
            ordered[len(ordered) // 3: 2 * len(ordered) // 3],
            ordered[2 * len(ordered) // 3:]]
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for split, counts in SPLIT_COUNTS.items():
        for visual_bin, (pool, count) in enumerate(zip(bins, counts, strict=True)):
            available = [row for row in pool if _image_key(row) not in used]
            for row in _evenly(available, count):
                item = dict(row)
                item["anchor_split"] = split
                item["selection_visual_bin"] = ("low", "mid", "high")[visual_bin]
                item["image_area"] = _area(row)
                chosen.append(item)
                used.add(_image_key(row))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmstar", type=Path, required=True)
    parser.add_argument("--chartqa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=str(MODEL))
    args = parser.parse_args()

    selected = _select(load_manifest(args.mmstar)["requests"])
    selected += _select(load_manifest(args.chartqa)["requests"])
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    for row in selected:
        _, meta = _prepare(processor, row)
        row["image_sha256"] = _image_key(row)
        row["image_grid_thw"] = meta["image_grid_thw"]
        row["vision_token_count"] = meta["vision_token_count"]
        row["prompt_token_count"] = meta["prompt_token_count"]
    selected.sort(key=lambda row: (
        ("calibration", "development", "untouched_test").index(row["anchor_split"]),
        row["task"], row["vision_token_count"], row["request_id"],
    ))
    assert len(selected) == 64
    assert len({_image_key(row) for row in selected}) == 64
    atomic_json(args.output, {
        "schema_version": "mllm_ep_anchor_manifest.v1",
        "model": args.model,
        "selection": "image-disjoint task-balanced area-stratified; actual processor grids recorded",
        "split_counts": {
            split: sum(row["anchor_split"] == split for row in selected)
            for split in SPLIT_COUNTS
        },
        "requests": selected,
    })
    print(json.dumps({"output": str(args.output), "requests": len(selected),
                      "vision_min": min(row["vision_token_count"] for row in selected),
                      "vision_max": max(row["vision_token_count"] for row in selected)}))


if __name__ == "__main__":
    main()
