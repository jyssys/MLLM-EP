"""Combine frozen request manifests without changing request contents/order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_ep_foundation.io import atomic_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-input", type=int)
    args = parser.parse_args()
    requests = []
    sources = []
    seen = set()
    for path in args.input:
        value = json.loads(path.read_text(encoding="utf-8"))
        selected = value["requests"][:args.limit_per_input]
        sources.append({"path": str(path.resolve()), "sha256": sha256(path),
                        "task": value.get("task"), "source_requests": len(value["requests"]),
                        "selected_requests": len(selected)})
        for row in selected:
            request_id = str(row["request_id"])
            if request_id in seen:
                raise ValueError(f"duplicate request ID: {request_id}")
            seen.add(request_id)
            requests.append(row)
    payload = {
        "schema_version": "foundation.combined_manifest.v1",
        "protocol_status": "PUBLIC_PROTOCOL_PARTIAL",
        "sources": sources,
        "requests": requests,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "requests": len(requests),
                      "sources": sources}, indent=2))


if __name__ == "__main__":
    main()
