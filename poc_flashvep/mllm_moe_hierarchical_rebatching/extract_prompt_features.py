#!/usr/bin/env python3
"""Extract stable actual model-input token counts from a completed run."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    values: dict[str, set[int]] = {}
    for name in glob.glob(str(args.run / "requests_dp*.jsonl")):
        for line in open(name):
            row = json.loads(line)
            if row["warmup"]:
                continue
            values.setdefault(row["source_request_id"], set()).add(int(row["prompt_tokens"]))
    unstable = {key: sorted(value) for key, value in values.items() if len(value) != 1}
    if unstable:
        raise AssertionError(f"Prompt token counts changed: {unstable}")
    result = {key: next(iter(value)) for key, value in sorted(values.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"requests": len(result), "min": min(result.values()), "max": max(result.values())}))


if __name__ == "__main__":
    main()
