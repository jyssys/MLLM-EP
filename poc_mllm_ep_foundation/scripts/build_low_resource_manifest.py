"""Freeze an image-disjoint, task-balanced semantic fallback cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_ep_foundation.io import atomic_json


def _select(rows: list[dict], count: int) -> list[dict]:
    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(str(row["image_sha256"]), row)
    ordered = sorted(unique.values(), key=lambda row: int(row["source_row"]))
    if count > len(ordered):
        raise ValueError((count, len(ordered)))
    indexes = [round(index * (len(ordered) - 1) / (count - 1))
               for index in range(count)]
    chosen = [dict(ordered[index]) for index in indexes]
    if len({row["image_sha256"] for row in chosen}) != count:
        raise AssertionError("selection is not image-disjoint")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=96)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    requests = []
    for task in ("mmstar", "chartqa"):
        requests.extend(_select(
            [row for row in source["requests"] if row["task"] == task],
            args.per_task,
        ))
    requests.sort(key=lambda row: (row["task"], int(row["source_row"])))
    payload = {
        "schema_version": "foundation.low_resource_semantic.v1",
        "parent_manifest": str(args.input.resolve()),
        "selection": "96 evenly spaced image-disjoint public samples per task; frozen before evaluation",
        "protocol_status": source.get("protocol_status", "PUBLIC_PROTOCOL_PARTIAL"),
        "requests": requests,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()),
                      "requests": len(requests),
                      "unique_images": len({row["image_sha256"] for row in requests})}))


if __name__ == "__main__":
    main()
