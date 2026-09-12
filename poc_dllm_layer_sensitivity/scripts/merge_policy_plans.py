#!/usr/bin/env python3
"""Merge generated intervention plans while keeping one shared baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("plans", type=Path, nargs="+")
    args = parser.parse_args()

    merged: list[dict] = []
    seen: set[str] = set()
    for plan_path in args.plans:
        for policy in json.loads(plan_path.read_text()):
            name = policy["name"]
            if name in seen:
                continue
            seen.add(name)
            merged.append(policy)

    if "baseline" not in seen:
        merged.insert(
            0,
            {
                "name": "baseline",
                "mode": "none",
                "layers": "",
                "phases": "early,middle,late",
                "waves": "",
            },
        )
    else:
        merged.sort(key=lambda policy: policy["name"] != "baseline")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "policies": len(merged)}))


if __name__ == "__main__":
    main()
