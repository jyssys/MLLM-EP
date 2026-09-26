from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_ep_foundation.io import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-per-task", type=int, default=8)
    parser.add_argument("--validation-per-task", type=int, default=4)
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    selected = []
    for split in ("calibration", "development", "untouched_test"):
        for task in ("chartqa", "mmstar"):
            rows = sorted((row for row in value["requests"]
                           if row["anchor_split"] == split and row["task"] == task),
                          key=lambda row: (row["vision_token_count"], row["request_id"]))
            count = (args.calibration_per_task if split == "calibration"
                     else args.validation_per_task)
            indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
            selected.extend(rows[index] for index in indices)
    split_counts = {
        split: sum(row["anchor_split"] == split for row in selected)
        for split in ("calibration", "development", "untouched_test")
    }
    atomic_json(args.output, {
        **{key: item for key, item in value.items() if key != "requests"},
        "schema_version": "mllm_ep_anchor_timing_subset.v2",
        "parent_manifest": str(args.input.resolve()),
        "selection": "four evenly spaced actual visual-token shapes per split and task",
        "split_counts": split_counts,
        "requests": selected,
    })
    print(json.dumps({"output": str(args.output), "requests": len(selected),
                      "split_counts": split_counts}))


if __name__ == "__main__":
    main()
