#!/usr/bin/env python3
"""Aggregate per-rank transformer-block traces onto logical critical paths."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FIELDS = ("prepare_attn_ms", "attention_ms", "prepare_mlp_ms", "mlp_ms", "postprocess_ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(errors="replace")
    nfe = {
        f"measured_{int(offset)}": int(count)
        for offset, count in re.findall(r"\[iter\s+(\d+)\]nfe=\s*(\d+)", text)
    }
    denoise: dict[str, list[dict]] = defaultdict(list)
    prefix = "[LLADA_DENOISE]"
    for line in text.splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        record = json.loads(line[position + len(prefix) :])
        if record.get("request_id") in nfe:
            denoise[record["request_id"]].append(record)
    if set(denoise) != set(nfe):
        raise RuntimeError("block analysis requires the logical denoising trace")
    rank_rows: list[dict] = []
    for rank in range(4):
        with (args.trace / f"block_rank{rank}.jsonl").open() as handle:
            rank_rows.extend(
                row for line in handle
                if (row := json.loads(line)).get("request_id") in nfe
            )

    selected: dict[tuple[int, int, str, int], dict] = {}
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in rank_rows:
        groups[(row["rank"], row["layer"], row["request_id"])].append(row)
    for (rank, layer, request_id), rows in groups.items():
        rows.sort(key=lambda row: row["invocation"])
        count = len(denoise[request_id])
        if len(rows) < count:
            raise RuntimeError(f"incomplete block trace: {rank=} {layer=} {request_id=}")
        for iteration, row in enumerate(rows[-count:]):
            selected[(rank, layer, request_id, iteration)] = row

    logical: list[dict] = []
    for layer in range(32):
        for request_id, records in denoise.items():
            count = len(records)
            for iteration in range(count):
                rows = [selected[(rank, layer, request_id, iteration)] for rank in range(4)]
                record = records[iteration]
                physical_rows = int(record["physical_rows"])
                live = int(sum(record["remaining_before"]))
                out = {
                    "request_id": request_id,
                    "layer": layer,
                    "iteration": iteration,
                    "q_len": rows[0]["q_len"],
                    "batch": rows[0]["batch"],
                    "physical_rows": physical_rows,
                    "live_positions": live,
                    "live_ratio": live / physical_rows,
                    "phase": "early" if live / physical_rows > 2 / 3 else "middle" if live / physical_rows > 1 / 3 else "late",
                }
                for field in FIELDS:
                    out[field.replace("_ms", "_critical_ms")] = max(float(row[field]) for row in rows)
                out["block_critical_sum_ms"] = sum(
                    out[field.replace("_ms", "_critical_ms")] for field in FIELDS
                )
                logical.append(out)

    summary: list[dict] = []
    for phase in ("all", "early", "middle", "late"):
        phase_rows = logical if phase == "all" else [row for row in logical if row["phase"] == phase]
        if not phase_rows:
            continue
        denominator = sum(row["block_critical_sum_ms"] for row in phase_rows)
        for field in FIELDS + ("block_critical_sum_ms",):
            key = field.replace("_ms", "_critical_ms") if field in FIELDS else field
            values = np.asarray([row[key] for row in phase_rows])
            summary.append(
                {
                    "phase": phase,
                    "stage": key.removesuffix("_critical_ms"),
                    "logical_layer_invocations": len(values),
                    "median_critical_ms": float(np.median(values)),
                    "p95_critical_ms": float(np.percentile(values, 95)),
                    "sum_critical_ms": float(values.sum()),
                    "share_of_observed_block_sum": float(values.sum() / denominator),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("block_logical_invocations.csv", logical), ("block_stage_summary.csv", summary)):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "block_trace_manifest.json").write_text(
        json.dumps(
            {
                "nfe_by_request": nfe,
                "denoise_forwards_by_request": {key: len(value) for key, value in denoise.items()},
                "raw_rank_rows": len(rank_rows),
                "logical_layer_invocations": len(logical),
                "critical_rank_aggregation": "maximum rank duration per sequential stage",
                "observer_heavy": True,
            }, indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
