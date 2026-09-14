#!/usr/bin/env python3
"""Add deterministic controlled EP4 route shapes and summarize the corpus.

The real cases retain measured LLaDA2 routes.  Controlled cases hold M/top-k
fixed while varying load skew, token destination fanout, and remote fraction.
They are diagnostic replay inputs, not model-generated routing evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


WORLD = 4
EXPERTS_PER_RANK = 64
TOPK = 8


def experts_for_rank(rank: int, offset: int, count: int) -> list[int]:
    return [rank * EXPERTS_PER_RANK + ((offset + i) % EXPERTS_PER_RANK) for i in range(count)]


def route_row(source: int, row: int, variant: str) -> list[int]:
    if variant == "uniform_fanout4":
        return sum((experts_for_rank(rank, row * 2, 2) for rank in range(WORLD)), [])
    if variant == "mild_skew_fanout4":
        return (
            experts_for_rank(0, row * 3, 3)
            + experts_for_rank(1, row * 2, 2)
            + experts_for_rank(2, row * 2, 2)
            + experts_for_rank(3, row, 1)
        )
    if variant == "strong_skew_fanout2":
        return experts_for_rank(0, row * 7, 7) + experts_for_rank(1, row, 1)
    if variant == "fanout1_local":
        return experts_for_rank(source, row * TOPK, TOPK)
    if variant == "fanout1_remote":
        return experts_for_rank((source + 1) % WORLD, row * TOPK, TOPK)
    raise ValueError(variant)


def build_case(m: int, variant: str) -> dict:
    base, remainder = divmod(m, WORLD)
    local_m = [base + (rank < remainder) for rank in range(WORLD)]
    rows = []
    global_row = 0
    for source, count in enumerate(local_m):
        source_rows = []
        for _ in range(count):
            source_rows.append(route_row(source, global_row, variant))
            global_row += 1
        rows.append(source_rows)
    return {
        "case_id": f"control_m{m}_{variant}",
        "dataset": "controlled",
        "wave": -1,
        "layer": 1,
        "phase": "controlled",
        "shape_class": (
            "small" if m <= 32 else "medium-small" if m <= 128 else "medium"
        ),
        "physical_dense_m": m,
        "fresh_m": m,
        "source_fresh_m": local_m,
        "topk": TOPK,
        "hidden": 4096,
        "experts": 256,
        "experts_per_rank": EXPERTS_PER_RANK,
        "source_topk_ids": rows,
        "provenance": "deterministic controlled route shape; not model generated",
        "epoch_status": "not_applicable_control",
        "case_kind": "controlled",
        "control_axis": variant,
    }


def summarize(case: dict) -> dict:
    histogram = [0] * 256
    rank_load = [0] * WORLD
    remote = 0
    fanouts = []
    for source, source_rows in enumerate(case["source_topk_ids"]):
        for ids in source_rows:
            owners = set()
            for expert in ids:
                owner = int(expert) // EXPERTS_PER_RANK
                histogram[int(expert)] += 1
                rank_load[owner] += 1
                remote += owner != source
                owners.add(owner)
            fanouts.append(len(owners))
    pairs = sum(histogram)
    active_counts = [value for value in histogram if value]
    mean = sum(rank_load) / WORLD
    rank_cv = math.sqrt(sum((x - mean) ** 2 for x in rank_load) / WORLD) / mean if mean else 0
    case["histogram"] = histogram
    case["rank_load"] = rank_load
    case["remote_assignments"] = remote
    case["destination_fanout_mean"] = sum(fanouts) / len(fanouts) if fanouts else 0
    case["rank_load_cv"] = rank_cv
    return {
        "case_id": case["case_id"],
        "case_kind": case.get("case_kind", "real"),
        "dataset": case["dataset"],
        "fresh_m": case["fresh_m"],
        "shape_class": case["shape_class"],
        "control_axis": case.get("control_axis", "measured_route"),
        "token_expert_pairs": pairs,
        "remote_assignments": remote,
        "remote_fraction": remote / pairs if pairs else 0,
        "destination_fanout_mean": case["destination_fanout_mean"],
        "active_experts": len(active_counts),
        "rows_per_active_expert_mean": pairs / len(active_counts) if active_counts else 0,
        "max_rows_per_expert": max(active_counts, default=0),
        "rank0_load": rank_load[0],
        "rank1_load": rank_load[1],
        "rank2_load": rank_load[2],
        "rank3_load": rank_load[3],
        "rank_load_cv": rank_cv,
        "provenance": case["provenance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    source = [json.loads(line) for line in args.replay.read_text().splitlines() if line.strip()]
    # Idempotent when rerun on a corpus that already contains our controls.
    real = [case for case in source if case.get("case_kind", "real") == "real"]
    for case in real:
        case.setdefault("case_kind", "real")
        case.setdefault("control_axis", "measured_route")
    controls = [
        build_case(m, variant)
        for m in (32, 128, 512)
        for variant in (
            "uniform_fanout4",
            "mild_skew_fanout4",
            "strong_skew_fanout2",
            "fanout1_local",
            "fanout1_remote",
        )
    ]
    corpus = real + controls
    rows = [summarize(case) for case in corpus]
    args.replay.write_text("".join(json.dumps(case) + "\n" for case in corpus))
    with args.summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"real": len(real), "controlled": len(controls), "total": len(corpus)}))


if __name__ == "__main__":
    main()
