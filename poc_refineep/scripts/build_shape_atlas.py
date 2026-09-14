#!/usr/bin/env python3
"""Build a faithful post-compaction EP shape atlas and replay corpus.

The source trace is a measured LLaDA2.0-Flash true-EP4 trajectory.  For each
source rank, mask_before identifies rows that were live when entering the
forward.  We filter that rank's captured top-k rows with the exact mask slice,
then join the four sources.  This is an Epoch/FreshLane-like *sensitivity
worklist*, not a measured Epoch implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


PHYSICAL_GPUS = [4, 5, 6, 7]
EXPERTS = 256
EXPERTS_PER_RANK = 64
TOPK = 8
HIDDEN = 4096


def flatten(rows):
    return [value for row in rows for value in row]


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - position) + ordered[hi] * (position - lo))


def shape_class(m):
    if m <= 8:
        return "very-small"
    if m <= 32:
        return "small"
    if m <= 128:
        return "medium-small"
    if m <= 512:
        return "medium"
    return "large"


def load_raw(raw_root: Path):
    cases = {}
    for dataset in ("gsm8k", "humaneval"):
        rank_files = sorted((raw_root / dataset / "r1").glob("shape_rank*.jsonl"))
        if len(rank_files) != 4:
            raise RuntimeError(f"expected four rank files for {dataset}, got {rank_files}")
        for path in rank_files:
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("request_id") != "measured_baseline_0" or row.get("wave", -1) < 0:
                        continue
                    source_rank = int(row["rank"])
                    key = (dataset, int(row["wave"]), int(row["layer"]))
                    case = cases.setdefault(
                        key,
                        {
                            "dataset": dataset,
                            "wave": int(row["wave"]),
                            "layer": int(row["layer"]),
                            "phase": row["phase"],
                            "physical_m": int(row["physical_m_global"]),
                            "source_topk_ids": [None] * 4,
                            "source_full_topk_ids": [None] * 4,
                        },
                    )
                    global_mask = flatten(row["mask_before"])
                    start = int(row["global_row_start"])
                    local_m = int(row["physical_m_local"])
                    local_mask = global_mask[start : start + local_m]
                    local_topk = row["topk_ids_local_source"]
                    if len(local_mask) != len(local_topk):
                        raise RuntimeError(f"mask/topk mismatch at {key} rank {source_rank}")
                    case["source_full_topk_ids"][source_rank] = local_topk
                    case["source_topk_ids"][source_rank] = [
                        ids for ids, is_live in zip(local_topk, local_mask) if is_live
                    ]
    return cases


def summarize(case):
    histogram = [0] * EXPERTS
    rank_load = [0] * 4
    remote_pairs = 0
    fanouts = []
    for source_rank, source_rows in enumerate(case["source_topk_ids"]):
        for ids in source_rows:
            owners = set()
            for expert in ids:
                expert = int(expert)
                histogram[expert] += 1
                owner = expert // EXPERTS_PER_RANK
                rank_load[owner] += 1
                owners.add(owner)
                if owner != source_rank:
                    remote_pairs += 1
            fanouts.append(len(owners))
    live_m = sum(len(rows) for rows in case["source_topk_ids"])
    pairs = sum(histogram)
    active_counts = [count for count in histogram if count]
    active = len(active_counts)
    mean_rows = pairs / active if active else 0.0
    mean_load = sum(rank_load) / 4 if rank_load else 0.0
    load_cv = (
        math.sqrt(sum((x - mean_load) ** 2 for x in rank_load) / 4) / mean_load
        if mean_load
        else 0.0
    )
    return {
        "fresh_m": live_m,
        "token_expert_pairs": pairs,
        "remote_assignments": remote_pairs,
        "remote_bytes_one_way_bf16": remote_pairs * HIDDEN * 2,
        "active_experts": active,
        "rows_per_active_expert_mean": mean_rows,
        "rows_per_active_expert_p50": median(active_counts) if active_counts else 0.0,
        "rows_per_active_expert_p90": percentile(active_counts, 0.9),
        "max_rows_per_expert": max(active_counts, default=0),
        "tiny_expert_fraction_le4": (
            sum(count <= 4 for count in active_counts) / active if active else 0.0
        ),
        "destination_fanout_mean": sum(fanouts) / len(fanouts) if fanouts else 0.0,
        "destination_fanout_max": max(fanouts, default=0),
        "rank_load": rank_load,
        "rank_load_cv": load_cv,
        "critical_rank": max(range(4), key=rank_load.__getitem__) if pairs else -1,
        "histogram": histogram,
    }


def select_replay(cases):
    """Select five real quantile cases per size class, including endpoints."""
    grouped = defaultdict(list)
    for key, case in cases.items():
        metrics = summarize(case)
        if metrics["fresh_m"]:
            grouped[shape_class(metrics["fresh_m"])].append((key, case, metrics))

    selected = []
    order = ["very-small", "small", "medium-small", "medium", "large"]
    for label in order:
        items = grouped[label]
        if not items:
            continue
        ordered = sorted(items, key=lambda item: (item[2]["fresh_m"], item[0]))
        chosen = []
        for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
            target_index = round((len(ordered) - 1) * quantile)
            target_m = ordered[target_index][2]["fresh_m"]
            ranked = sorted(
                ordered,
                key=lambda item: (
                    abs(item[2]["fresh_m"] - target_m),
                    item in chosen,
                    item[0][0], item[0][1], item[0][2],
                ),
            )
            for item in ranked:
                if item not in chosen:
                    chosen.append(item)
                    break
        for item in ordered:
            if item not in chosen:
                chosen.append(item)
            if len(chosen) == 5:
                break
        selected.extend(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_raw(args.raw_root)
    atlas_rows = []
    for key, case in sorted(cases.items()):
        metrics = summarize(case)
        atlas_rows.append(
            {
                "dataset": case["dataset"],
                "wave": case["wave"],
                "layer": case["layer"],
                "phase": case["phase"],
                "physical_dense_m": case["physical_m"],
                "fresh_m": metrics["fresh_m"],
                "fresh_fraction": metrics["fresh_m"] / case["physical_m"],
                "shape_class": shape_class(metrics["fresh_m"]),
                "token_expert_pairs": metrics["token_expert_pairs"],
                "remote_assignments": metrics["remote_assignments"],
                "remote_bytes_one_way_bf16": metrics["remote_bytes_one_way_bf16"],
                "active_experts": metrics["active_experts"],
                "rows_per_active_expert_mean": metrics["rows_per_active_expert_mean"],
                "rows_per_active_expert_p50": metrics["rows_per_active_expert_p50"],
                "rows_per_active_expert_p90": metrics["rows_per_active_expert_p90"],
                "max_rows_per_expert": metrics["max_rows_per_expert"],
                "tiny_expert_fraction_le4": metrics["tiny_expert_fraction_le4"],
                "destination_fanout_mean": metrics["destination_fanout_mean"],
                "destination_fanout_max": metrics["destination_fanout_max"],
                "rank0_load": metrics["rank_load"][0],
                "rank1_load": metrics["rank_load"][1],
                "rank2_load": metrics["rank_load"][2],
                "rank3_load": metrics["rank_load"][3],
                "rank_load_cv": metrics["rank_load_cv"],
                "critical_rank": metrics["critical_rank"],
                "source_trace_kind": "measured_route_plus_oracle_live_row_filter",
                "epoch_status": "sensitivity_only_not_measured_epoch",
            }
        )
    atlas_path = args.output_dir / "REFINEMENT_EP_SHAPES.csv"
    with atlas_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(atlas_rows[0]))
        writer.writeheader()
        writer.writerows(atlas_rows)

    replay_path = args.output_dir / "EP_SHAPE_REPLAY.jsonl"
    with replay_path.open("w") as stream:
        for index, (_, case, metrics) in enumerate(select_replay(cases)):
            row = {
                "case_id": f"real_{index:02d}_{case['dataset']}_w{case['wave']}_l{case['layer']}",
                "dataset": case["dataset"],
                "wave": case["wave"],
                "layer": case["layer"],
                "phase": case["phase"],
                "shape_class": shape_class(metrics["fresh_m"]),
                "physical_dense_m": case["physical_m"],
                "fresh_m": metrics["fresh_m"],
                "source_fresh_m": [len(rows) for rows in case["source_topk_ids"]],
                "topk": TOPK,
                "hidden": HIDDEN,
                "experts": EXPERTS,
                "experts_per_rank": EXPERTS_PER_RANK,
                "source_topk_ids": case["source_topk_ids"],
                "histogram": metrics["histogram"],
                "rank_load": metrics["rank_load"],
                "remote_assignments": metrics["remote_assignments"],
                "destination_fanout_mean": metrics["destination_fanout_mean"],
                "rank_load_cv": metrics["rank_load_cv"],
                "provenance": "measured LLaDA2 EP4 routes filtered by future-known live mask",
                "epoch_status": "compaction sensitivity; no Epoch implementation claimed",
            }
            stream.write(json.dumps(row) + "\n")

    print(json.dumps({"atlas_rows": len(atlas_rows), "replay_cases": sum(1 for _ in replay_path.open()), "atlas": str(atlas_path), "replay": str(replay_path)}))


if __name__ == "__main__":
    main()
