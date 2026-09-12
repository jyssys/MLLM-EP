#!/usr/bin/env python3
"""Extract dLLM liveness/acceptance from trace-only benchmark logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = []
    prefix = "[LLADA_DENOISE]"
    for line in args.log.read_text(errors="replace").splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        record = json.loads(line[position + len(prefix) :])
        if not record["request_id"].startswith("measured_"):
            continue
        physical = int(record["physical_rows"])
        live = int(sum(record["remaining_before"]))
        accepted = int(sum(record["accepted"]))
        rows.append(
            {
                "request_id": record["request_id"],
                "iteration": record["iteration"],
                "selected_sequences": len(record["sequence_ids"]),
                "block_starts": json.dumps(record["block_starts"], separators=(",", ":")),
                "physical_rows": physical,
                "live_positions_before": live,
                "accepted_positions": accepted,
                "live_to_physical_ratio": live / physical,
                "accepted_per_forward": accepted,
                "phase": "early" if live / physical > 2 / 3 else "middle" if live / physical > 1 / 3 else "late",
            }
        )
    if not rows:
        raise RuntimeError("no measured denoising trace rows found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for phase in ("all", "early", "middle", "late"):
        subset = rows if phase == "all" else [row for row in rows if row["phase"] == phase]
        summary[phase] = {
            "forwards": len(subset),
            "live_to_physical_ratio_mean": float(np.mean([row["live_to_physical_ratio"] for row in subset])),
            "live_to_physical_ratio_median": float(np.median([row["live_to_physical_ratio"] for row in subset])),
            "accepted_per_forward_mean": float(np.mean([row["accepted_per_forward"] for row in subset])),
        }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
