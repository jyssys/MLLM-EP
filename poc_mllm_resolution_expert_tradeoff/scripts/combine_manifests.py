from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from poc_mllm_ep_foundation.io import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["requests"]:
            request_id = str(row["request_id"])
            if request_id in seen:
                raise RuntimeError(f"duplicate request id: {request_id}")
            seen.add(request_id)
            requests.append(row)
    atomic_json(args.output, {
        "schema_version": "mllm_resolution_expert_tradeoff.manifest.v1",
        "split": args.name,
        "requests": requests,
        "sources": [str(path.resolve()) for path in args.input],
    })


if __name__ == "__main__":
    main()
