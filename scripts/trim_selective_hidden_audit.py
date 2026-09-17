#!/usr/bin/env python3
"""Enforce the 16-request cap on adjacent hidden-state audit records.

Policy traces keep routing/work arrays and F0/F1 cache-drift records. Only the
extra ``adjacent_hidden`` diagnostic is removed beyond fixed request ID 15.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


DRIFT_KEYS = (
    "drift_request_id", "drift_block_id", "drift_iteration_id", "drift_layer_id",
    "drift_position", "drift_freeze_age", "drift_boundary", "drift_cosine",
    "drift_relative_l2",
)


def trim(path: Path, maximum_request: int) -> bool:
    request_id = int(path.stem.split("_")[-1])
    if request_id <= maximum_request:
        return False
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    if "drift_boundary" not in payload:
        return False
    keep = payload["drift_boundary"] != "adjacent_hidden"
    if np.all(keep):
        return False
    if np.any(keep):
        for key in DRIFT_KEYS:
            payload[key] = payload[key][keep]
    else:
        for key in DRIFT_KEYS:
            payload.pop(key, None)
    metadata = json.loads(str(payload["metadata_json"]))
    metadata["adjacent_hidden_capture"] = False
    metadata["hidden_audit_cap_enforced_post_collection"] = True
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--maximum-request", type=int, default=15)
    args = parser.parse_args()
    changed = 0
    for root in args.roots:
        for path in sorted((root / "traces").glob("request_*.npz")):
            changed += int(trim(path, args.maximum_request))
    print(json.dumps({"trimmed_files": changed, "maximum_request": args.maximum_request}))


if __name__ == "__main__":
    main()
