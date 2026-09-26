"""Interleave frozen split pairs to remove execution-order confounding."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from poc_mllm_ep_foundation.io import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    by_split = {}
    for split in ("calibration", "development", "untouched_test"):
        rows = [row for row in value["requests"] if row["anchor_split"] == split]
        if len(rows) % 2:
            raise ValueError((split, len(rows)))
        by_split[split] = deque([rows[index:index + 2] for index in range(0, len(rows), 2)])
    selected = []
    schedule = []
    # The frozen split ratio is 2:1:1.  A plain three-way round robin drains
    # development/test first and leaves a calibration-only tail, recreating
    # the thermal/order confound this manifest is meant to remove.  Consume
    # pairs in the same 2:1:1 ratio so every split spans the full run.
    split_cycle = ("calibration", "development", "calibration", "untouched_test")
    while any(by_split.values()):
        progressed = False
        for split in split_cycle:
            if not by_split[split]:
                continue
            pair = by_split[split].popleft()
            schedule.append({"wave": len(schedule), "split": split,
                             "request_ids": [row["request_id"] for row in pair]})
            selected.extend(pair)
            progressed = True
        if not progressed:  # pragma: no cover - defensive against malformed input
            raise RuntimeError("interleave schedule made no progress")
    payload = {
        **{key: item for key, item in value.items()
           if key not in ("requests", "execution_schedule")},
        "schema_version": "mllm_ep_anchor_timing_interleaved.v1",
        "parent_manifest": str(args.input.resolve()),
        "execution_order": "frozen within-split pairs; 2:1:1 calibration/development/calibration/untouched cycle",
        "execution_schedule": schedule,
        "requests": selected,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "requests": len(selected),
                      "waves": schedule}, indent=2))


if __name__ == "__main__":
    main()
