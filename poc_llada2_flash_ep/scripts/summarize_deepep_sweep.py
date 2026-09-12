#!/usr/bin/env python3
"""Collapse DeepEP rank rows into critical-rank dispatch/combine timings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    summaries = []
    for directory in sorted(args.root.glob("source_tokens_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])):
        rank_rows = []
        for rank in range(4):
            rank_rows.append([json.loads(line) for line in (directory / f"rank{rank}.jsonl").open()])
        repeats = min(map(len, rank_rows))
        dispatch = np.asarray([max(rank_rows[rank][i]["dispatch_ms"] for rank in range(4)) for i in range(repeats)])
        combine = np.asarray([max(rank_rows[rank][i]["combine_ms"] for rank in range(4)) for i in range(repeats)])
        source_tokens = int(rank_rows[0][0]["source_tokens"])
        summaries.append(
            {
                "source_tokens_per_rank": source_tokens,
                "global_physical_m": source_tokens * 4,
                "assignments": source_tokens * 4 * 8,
                "repeats": repeats,
                "dispatch_p50_ms": float(np.median(dispatch)),
                "dispatch_p95_ms": float(np.percentile(dispatch, 95)),
                "combine_p50_ms": float(np.median(combine)),
                "combine_p95_ms": float(np.percentile(combine, 95)),
                "communication_p50_ms": float(np.median(dispatch + combine)),
                "communication_p95_ms": float(np.percentile(dispatch + combine, 95)),
                "relative_l2_max": max(row["relative_l2"] for rows in rank_rows for row in rows),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
