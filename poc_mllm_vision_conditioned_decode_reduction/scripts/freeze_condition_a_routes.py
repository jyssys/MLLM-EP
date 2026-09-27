#!/usr/bin/env python3
"""Export one exact condition-A TP0 route row per request/layer/checkpoint."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--trace-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pool_data = json.loads(args.pool.read_text())
    pool = {
        str(row["pair_id"]): str(
            row.get("reduction_split") or row.get("coscheduling_split") or row["analysis_split"]
        )
        for row in pool_data["anchors"]
    }
    rows = {}
    for root in args.trace_dir:
        for path in sorted(root.glob("route_summary.pid*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                pair = str(raw.get("pair_id") or "")
                step = raw.get("decode_step")
                if (
                    pair not in pool
                    or raw.get("condition") != "A"
                    or int(raw.get("tp_rank", -1)) != 0
                    or step is None
                    or not 1 <= int(step) <= 128
                ):
                    continue
                ids, weights = raw.get("route_ids") or [], raw.get("route_weights") or []
                if len(ids) != 1 or len(weights) != 1 or len(ids[0]) != 8 or len(weights[0]) != 8:
                    raise ValueError((path, pair, step, raw.get("layer")))
                row = {
                    "pair_id": pair,
                    "split": pool[pair],
                    "step": int(step),
                    "layer": int(raw["layer"]),
                    "route_ids": [int(value) for value in ids[0]],
                    "route_weights": [float(value) for value in weights[0]],
                    "source_ep_rank": int(raw.get("ep_rank", 0)),
                    "evidence": "MEASURED_TRUE_EP4_CONDITION_A_TP0",
                }
                key = (pair, row["layer"], row["step"])
                if key in rows and rows[key] != row:
                    raise ValueError(f"conflicting duplicate {key}")
                rows[key] = row
    expected = len(pool) * 48 * 22
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        for key in sorted(rows):
            handle.write(json.dumps(rows[key], separators=(",", ":"), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
