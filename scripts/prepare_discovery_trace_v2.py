#!/usr/bin/env python3
"""Convert v1 audit traces and merge per-request discovery-v2 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from virtual_ep.discovery_trace import DiscoveryTrace, from_heavy_trace
from virtual_ep.schema import TraceBundle


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert-heavy")
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--requests", type=int, nargs="*")
    convert.add_argument("--prompt-records", type=Path,
                         help="JSONL containing sample_id and prompt_tokens")
    merge = sub.add_parser("merge")
    merge.add_argument("--shard-dir", type=Path, required=True)
    merge.add_argument("--extra", type=Path, nargs="*", default=[])
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "convert-heavy":
        trace = TraceBundle.load(args.input)
        if args.requests:
            trace = trace.select_requests(set(args.requests))
        prompt_lengths = None
        if args.prompt_records:
            rows = [json.loads(line) for line in args.prompt_records.read_text().splitlines()
                    if line.strip()]
            prompt_lengths = {int(row["sample_id"]): int(row["prompt_tokens"]) for row in rows}
        result = from_heavy_trace(trace, prompt_lengths)
        result.save(args.output)
        print(f"converted {len(result.metadata['requests'])} requests, "
              f"{len(result.arrays['request_id'])} invocations")
    else:
        paths = sorted(args.shard_dir.glob("request_*.npz")) + list(args.extra)
        traces = [DiscoveryTrace.load(path) for path in paths]
        result = DiscoveryTrace.concatenate(traces)
        result.save(args.output)
        print(f"merged {len(result.metadata['requests'])} requests, "
              f"{len(result.arrays['request_id'])} invocations")


if __name__ == "__main__":
    main()
