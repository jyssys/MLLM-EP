#!/usr/bin/env python3
"""Summarize the live-row sensitivity of each static block size.

This is a counterfactual shape calculation, not a measured Epoch result.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with (ROOT / "EP_REGIME_MAP.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        if int(row["sample_count"]) != 8 or int(row["mini_batch_size"]) != 8:
            continue
        physical = float(row["physical_rows_median"])
        live_ratio = float(row["decision_live_ratio_median"])
        output.append(
            {
                "task": row["task"],
                "block_length": row["block_length"],
                "physical_rows_median": physical,
                "decision_live_ratio_median": live_ratio,
                "modeled_live_rows_median": physical * live_ratio,
                "modeled_dead_rows_median": physical * (1 - live_ratio),
                "modeled_row_reduction_percent": 100 * (1 - live_ratio),
                "active_experts_before_median": row["active_experts_median"],
                "tiny_expert_fraction_before_median": row["tiny_expert_fraction_le4_median"],
                "evidence": (
                    "modeled live-row count from measured mask state; "
                    "not measured Epoch latency and no compacted routing claim"
                ),
            }
        )
    with (ROOT / "EPOCH_COMPACTION_SENSITIVITY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} sensitivity rows")


if __name__ == "__main__":
    main()
