#!/usr/bin/env python3
"""Summarize bounded quality/performance runs and paired topology effects."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


PATTERN = re.compile(
    r"Forward:\s*(?P<nfe>\d+),\s*Time:\s*(?P<wall>[0-9.]+),.*?TPS:\s*(?P<tps>[0-9.]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for quality in args.root.glob("results/quality/*/*/r*/quality.json"):
        task, topology, repeat_dir = quality.parts[-4], quality.parts[-3], quality.parts[-2]
        repeat = int(repeat_dir.removeprefix("r"))
        log = args.root / "logs" / f"quality_{task}_{topology}_r{repeat}.log"
        matches = list(PATTERN.finditer(log.read_text(errors="replace")))
        if not matches:
            continue
        match = matches[-1]
        result = json.loads(quality.read_text())
        nfe, wall = int(match["nfe"]), float(match["wall"])
        rows.append(
            {
                "task": task,
                "topology": topology,
                "repeat": repeat,
                "correct": result["correct"],
                "total": result["total"],
                "accuracy": result["accuracy"],
                "nfe": nfe,
                "wall_seconds": wall,
                "ms_per_forward": wall * 1000 / nfe,
                "tokens_per_second": float(match["tps"]),
                "source_log": str(log.relative_to(args.root)),
            }
        )
    rows.sort(key=lambda row: (row["task"], row["repeat"], row["topology"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    paired = []
    by_key = {(row["task"], row["repeat"], row["topology"]): row for row in rows}
    for task, repeat in sorted({(row["task"], row["repeat"]) for row in rows}):
        if (task, repeat, "tp4") not in by_key or (task, repeat, "ep4") not in by_key:
            continue
        tp, ep = by_key[(task, repeat, "tp4")], by_key[(task, repeat, "ep4")]
        paired.append(
            {
                "task": task,
                "repeat": repeat,
                "wall_gain_ep_over_tp_percent": 100 * (tp["wall_seconds"] - ep["wall_seconds"]) / tp["wall_seconds"],
                "forward_normalized_gain_percent": 100 * (tp["ms_per_forward"] - ep["ms_per_forward"]) / tp["ms_per_forward"],
                "throughput_gain_percent": 100 * (ep["tokens_per_second"] - tp["tokens_per_second"]) / tp["tokens_per_second"],
            }
        )
    paired_path = args.output.with_name(args.output.stem + "_PAIRED.csv")
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    for task in sorted({row["task"] for row in paired}):
        selected = [row for row in paired if row["task"] == task]
        print(task, {key: float(np.median([row[key] for row in selected])) for key in selected[0] if key.endswith("percent")})


if __name__ == "__main__":
    main()
