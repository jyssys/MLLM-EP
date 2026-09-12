#!/usr/bin/env python3
"""Summarize model-forward microbatch geometry at a fixed submitted pool."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open()))
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        match = re.fullmatch(r"dynamic(\d+)", row["mode"])
        if match and int(row["batch"]) == 16:
            groups[(row["topology"], int(match.group(1)), int(row["batch"]))].append(row)
    output = []
    for (topology, mini, submitted), values in sorted(groups.items()):
        wall = np.asarray([float(row["wall_seconds"]) for row in values])
        per_forward = np.asarray([float(row["mean_ms_per_forward"]) for row in values])
        tps = np.asarray([float(row["generated_tokens_per_second"]) for row in values])
        output.append(
            {
                "topology": topology,
                "submitted_batch": submitted,
                "model_forward_microbatch": mini,
                "physical_m_decode": mini * 32,
                "restarts": len(values),
                "wall_seconds_median": float(np.median(wall)),
                "mean_ms_per_forward_median": float(np.median(per_forward)),
                "tokens_per_second_median": float(np.median(tps)),
            }
        )
    if not output:
        raise RuntimeError("no submitted-batch-16 dynamic microbatch records found")
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
