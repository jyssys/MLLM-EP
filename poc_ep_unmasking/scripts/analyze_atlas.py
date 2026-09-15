#!/usr/bin/env python3
"""Analyze stock confidence choices and sampled-layer rank signatures.

The future route in the one-step screen comes from the *baseline* next
forward.  It is a frozen-state upper bound, not a counterfactual rollout or
measured request speedup.
"""

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


DELTAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def token_signature(record, routes, layers, position):
    batch = len(record["block_starts"])
    global_row = record["batch_row"] * record["block_length"] + position
    rows_per_rank = batch * record["block_length"] // 4
    source_rank = global_row // rows_per_rank
    source_row = global_row % rows_per_rank
    loads = [0] * 4
    expert_ids = []
    for layer in layers:
        route = routes.get(
            (record["request_id"], source_rank, layer, record["ep_latest_invocation"])
        )
        if route is None or source_row >= len(route["topk_ids"]):
            return None
        ids = route["topk_ids"][source_row]
        expert_ids.extend(ids)
        for expert in ids:
            loads[expert // 64] += 1
    return loads, expert_ids, source_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--layers", default="1,8,16,24,31")
    args = parser.parse_args()
    layers = tuple(int(value) for value in args.layers.split(","))
    atlas = [
        record
        for record in read_jsonl(args.result_dir / "unmask" / "unmask_rank0.jsonl")
        if record["request_id"].startswith("measured_")
        and record["remaining_before"] > 0
        and record["selected_positions"]
    ]
    atlas.sort(key=lambda row: (row["call_index"], row["batch_row"]))
    iteration = defaultdict(int)
    by_lifetime = defaultdict(list)
    for record in atlas:
        seq_ids = record["sequence_ids"]
        seq_id = seq_ids[record["batch_row"]] if seq_ids else record["batch_row"]
        record["sequence_id"] = seq_id
        key = (record["request_id"], seq_id, record["block_starts"][record["batch_row"]])
        record["iteration"] = iteration[key]
        iteration[key] += 1
        by_lifetime[key].append(record)

    routes = {}
    for rank in range(4):
        path = args.result_dir / "ep" / f"ep_rank{rank}.jsonl"
        if not path.exists():
            continue
        for route in read_jsonl(path):
            if "topk_ids" in route and route["layer"] in layers:
                routes[
                    (route["request_id"], rank, route["layer"], route["invocation"])
                ] = route

    slack_rows = []
    signature_rows = []
    one_step_rows = []
    near_tie_pairs = []
    temporal_cost = []
    for record in atlas:
        selected = set(record["selected_positions"])
        confidences = dict(zip(record["eligible_positions"], record["confidences"]))
        cutoff = record["selected_cutoff"]
        best_unselected = record["best_unselected"]
        masked_fraction = record["remaining_before"] / record["block_length"]
        phase = "early" if masked_fraction > 2 / 3 else (
            "middle" if masked_fraction > 1 / 3 else "late"
        )
        key = (
            record["request_id"], record["sequence_id"],
            record["block_starts"][record["batch_row"]],
        )
        lifetime = by_lifetime[key]
        next_record = lifetime[record["iteration"] + 1] if (
            record["iteration"] + 1 < len(lifetime)
        ) else None
        current_signature = {}
        future_signature = {}
        positions_to_trace = set(record["eligible_positions"])
        if next_record:
            positions_to_trace.update(next_record["masked_positions"])
        for position in sorted(positions_to_trace):
            current = (
                token_signature(record, routes, layers, position)
                if position in record["eligible_positions"] else None
            )
            future = token_signature(next_record, routes, layers, position) if next_record else None
            if current:
                current_signature[position] = current
            if future:
                future_signature[position] = future
            if current and future:
                before_load, before_ids, before_source = current
                after_load, after_ids, after_source = future
                dot = sum(a * b for a, b in zip(before_load, after_load))
                before_norm = math.sqrt(sum(value * value for value in before_load))
                after_norm = math.sqrt(sum(value * value for value in after_load))
                temporal_cost.append({
                    "request_id": record["request_id"],
                    "call_index": record["call_index"],
                    "position": position,
                    "rank_cosine": dot / (before_norm * after_norm) if before_norm * after_norm else 0,
                    "rank_l1_change": sum(abs(a - b) for a, b in zip(before_load, after_load)),
                    "expert_set_overlap": len(set(before_ids) & set(after_ids)) / len(set(before_ids) | set(after_ids)) if set(before_ids) | set(after_ids) else 0,
                    "source_rank_same": int(before_source == after_source),
                })
            if current:
                loads, experts, source_rank = current
                signature_rows.append({
                    "request_id": record["request_id"],
                    "call_index": record["call_index"],
                    "sequence_id": record["sequence_id"],
                    "block_start": key[2],
                    "iteration": record["iteration"],
                    "phase": phase,
                    "position": position,
                    "confidence": confidences[position],
                    "selected": int(position in selected),
                    "near_tie_0p01": int(cutoff is not None and confidences[position] >= cutoff - 0.01),
                    "source_rank": source_rank,
                    "rank0": loads[0], "rank1": loads[1],
                    "rank2": loads[2], "rank3": loads[3],
                    "remote_assignments": sum(load for r, load in enumerate(loads) if r != source_rank),
                    "max_rank_contribution": max(loads),
                    "expert_ids": json.dumps(experts),
                })

        for delta in DELTAS:
            alternatives = [
                position for position, confidence in confidences.items()
                if position not in selected and cutoff is not None
                and confidence >= cutoff - delta
            ]
            removable_selected = [
                position for position in selected
                if cutoff is not None and confidences[position] <= cutoff + delta
            ]
            common = {
                "request_id": record["request_id"],
                "call_index": record["call_index"],
                "sequence_id": record["sequence_id"],
                "block_start": key[2],
                "iteration": record["iteration"],
                "phase": phase,
                "masked_before": record["remaining_before"],
                "transfer_count": len(selected),
                "selected_cutoff": cutoff,
                "best_unselected": best_unselected,
                "cutoff_gap": (cutoff - best_unselected) if (
                    cutoff is not None and best_unselected is not None
                ) else None,
                "delta": delta,
                "alternatives": len(alternatives),
                "swappable_selected": len(removable_selected),
            }
            slack_rows.append(common)
            if delta == 0.02:
                for old, new in itertools.product(removable_selected, alternatives):
                    if old in current_signature and new in current_signature:
                        old_load, old_ids, old_source = current_signature[old]
                        new_load, new_ids, new_source = current_signature[new]
                        near_tie_pairs.append({
                            "request_id": record["request_id"],
                            "call_index": record["call_index"],
                            "selected_position": old,
                            "alternative_position": new,
                            "confidence_gap": confidences[old] - confidences[new],
                            "rank_l1_difference": sum(abs(a - b) for a, b in zip(old_load, new_load)),
                            "max_rank_contribution_difference": abs(max(old_load) - max(new_load)),
                            "remote_assignment_difference": abs(
                                sum(v for r, v in enumerate(old_load) if r != old_source)
                                - sum(v for r, v in enumerate(new_load) if r != new_source)
                            ),
                            "expert_sets_differ": int(set(old_ids) != set(new_ids)),
                        })

            # Freeze the next baseline route.  Exact next-wave full-MoE
            # service time cannot be credited without compacted GPU replay.
            if not next_record or not future_signature:
                continue
            next_masked = set(next_record["masked_positions"])
            base_remaining = next_masked
            base_load = [
                sum(future_signature[p][0][rank] for p in base_remaining if p in future_signature)
                for rank in range(4)
            ]
            best_load = base_load
            best_swap = None
            for old, new in itertools.product(removable_selected, alternatives):
                if old not in future_signature or new not in future_signature:
                    continue
                # The baseline next live set excludes old and includes new.
                # Under the swap, old remains live while new is finalized.
                candidate = [
                    base_load[r] + future_signature[old][0][r]
                    - future_signature[new][0][r]
                    for r in range(4)
                ]
                if max(candidate) < max(best_load):
                    best_load, best_swap = candidate, (old, new)
            one_step_rows.append({
                **common,
                "future_step_available": 1,
                "future_positions_traced": len(future_signature),
                "base_rank_load": json.dumps(base_load),
                "best_rank_load": json.dumps(best_load),
                "base_max_rank": max(base_load),
                "best_max_rank": max(best_load),
                "max_rank_reduction_pct": (
                    100 * (max(base_load) - max(best_load)) / max(base_load)
                    if max(base_load) else 0
                ),
                "best_swap": json.dumps(best_swap),
                "one_swap_only": 1,
                "frozen_baseline_route_not_rollout": 1,
            })

    write_csv(args.result_dir / "CONFIDENCE_SLACK.csv", slack_rows)
    write_csv(args.result_dir / "ONE_STEP_ORACLES.csv", one_step_rows)
    write_csv(args.result_dir / "TOKEN_EP_SIGNATURES.csv", signature_rows)
    write_csv(args.result_dir / "NEAR_TIE_EP_PAIR_DIFFERENCES.csv", near_tie_pairs)
    write_csv(args.result_dir / "TEMPORAL_POSITION_EP_COST.csv", temporal_cost)
    try:
        import pandas as pd
        pd.DataFrame(signature_rows).to_parquet(
            args.result_dir / "TOKEN_EP_SIGNATURES.parquet", index=False
        )
    except (ImportError, ValueError) as exc:
        print("parquet unavailable:", exc)

    summary = {
        "decision_rows": len(atlas),
        "measured_route_records": len(routes),
        "sampled_layers": layers,
        "slack": {},
        "slack_by_phase": {},
        "one_step": {},
        "near_tie_pair_count_0p02": len(near_tie_pairs),
        "near_tie_pair_median_rank_l1_difference": median(row["rank_l1_difference"] for row in near_tie_pairs) if near_tie_pairs else 0,
        "near_tie_pair_median_remote_assignment_difference": median(row["remote_assignment_difference"] for row in near_tie_pairs) if near_tie_pairs else 0,
        "near_tie_pair_fraction_expert_sets_differ": sum(row["expert_sets_differ"] for row in near_tie_pairs) / len(near_tie_pairs) if near_tie_pairs else 0,
        "temporal_position_pairs": len(temporal_cost),
        "temporal_rank_cosine_median": median(row["rank_cosine"] for row in temporal_cost) if temporal_cost else None,
        "temporal_rank_l1_change_median": median(row["rank_l1_change"] for row in temporal_cost) if temporal_cost else None,
    }
    for delta in DELTAS:
        subset = [row for row in slack_rows if row["delta"] == delta]
        one = [row for row in one_step_rows if row["delta"] == delta]
        summary["slack"][str(delta)] = {
            "fraction_alternatives_ge1": sum(row["alternatives"] >= 1 for row in subset) / len(subset) if subset else 0,
            "fraction_alternatives_ge2": sum(row["alternatives"] >= 2 for row in subset) / len(subset) if subset else 0,
            "fraction_alternatives_ge4": sum(row["alternatives"] >= 4 for row in subset) / len(subset) if subset else 0,
            "fraction_alternatives_ge8": sum(row["alternatives"] >= 8 for row in subset) / len(subset) if subset else 0,
            "median_alternatives": median(row["alternatives"] for row in subset) if subset else 0,
        }
        summary["one_step"][str(delta)] = {
            "matched_next_steps": len(one),
            "fraction_positive": sum(row["max_rank_reduction_pct"] > 0 for row in one) / len(one) if one else 0,
            "median_max_rank_reduction_pct": median(row["max_rank_reduction_pct"] for row in one) if one else 0,
            "mean_max_rank_reduction_pct": sum(row["max_rank_reduction_pct"] for row in one) / len(one) if one else 0,
        }
        if delta == 0.02:
            for phase in ("early", "middle", "late"):
                phase_rows = [row for row in subset if row["phase"] == phase]
                summary["slack_by_phase"][phase] = {
                    "decisions": len(phase_rows),
                    "fraction_alternatives_ge1_0p02": sum(row["alternatives"] >= 1 for row in phase_rows) / len(phase_rows) if phase_rows else 0,
                    "median_transfer_count": median(row["transfer_count"] for row in phase_rows) if phase_rows else 0,
                }
    (args.result_dir / "atlas_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
