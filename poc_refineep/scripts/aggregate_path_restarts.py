#!/usr/bin/env python3
"""Aggregate existing-path summaries with restart as the statistical unit."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--restart-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for restart, path in enumerate(args.inputs, 1):
        frame = pd.read_csv(path)
        frame["restart"] = restart
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    args.restart_output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.restart_output, index=False, float_format="%.9f")
    keys = ["case_id", "dataset", "wave", "layer", "phase", "shape_class",
            "global_fresh_m", "max_tokens_contract", "policy"]
    numerical = ["layout_ms", "dispatch_ms", "weight_prep_ms", "combine_ms",
                 "comm_semantics_ms", "p90_comm_semantics_ms", "max_relative_l2",
                 "max_relative_l2_vs_identity", "source_assignments_global",
                 "received_assignments_global"]
    output = rows.groupby(keys, as_index=False)[numerical].median()
    output["assignment_count_match"] = output.source_assignments_global == output.received_assignments_global
    output["restarts"] = len(frames)
    output["repeats_per_restart"] = rows.repeats.min()
    output["timing_unit"] = "restart-median of critical-rank same-GPU CUDA-event medians"
    output.to_csv(args.output, index=False, float_format="%.9f")
    print(f"{len(output)} case-policy rows from {len(frames)} independent restarts")


if __name__ == "__main__":
    main()
