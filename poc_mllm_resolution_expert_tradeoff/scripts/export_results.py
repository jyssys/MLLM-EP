from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

def _read(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_request_id") or row["request_id"].split("__", 1)[0])


def _config(row: dict[str, Any]) -> str:
    return f"{row['resolution_label']}_K{int(row['k_visual'])}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, action="append", default=[])
    parser.add_argument("--heldout", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str, str]] = set()
    for split, paths in (("discovery", args.discovery), ("heldout", args.heldout)):
        for path in paths:
            for source in _read(path):
                row = dict(source)
                row["part2_result_split"] = split
                row["part2_config"] = _config(row)
                row["base_request_id"] = _base_id(row)
                key = (split, row["part2_config"], row["base_request_id"])
                if key in keys:
                    raise RuntimeError(f"duplicate consolidated result: {key}")
                keys.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (
        row["part2_result_split"], row["part2_config"], row["base_request_id"]
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=args.output.name + ".", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
