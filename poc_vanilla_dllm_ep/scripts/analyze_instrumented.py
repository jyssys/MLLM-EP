#!/usr/bin/env python3
"""Analyze stage timing and temporal structure without cross-GPU timestamps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


STAGES = ("router", "prepare", "dispatch", "expert", "combine", "moe")


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else float("nan")


def route_metrics(current: list[list[int]], previous: list[list[int]], world: int):
    ordered = []
    set_equal = []
    stable = []
    owner_equal = []
    for cur, prev in zip(current, previous):
        ordered.append(cur == prev)
        set_equal.append(set(cur) == set(prev))
        stable.append(len(set(cur) & set(prev)) / max(len(cur), 1))
        owner_equal.append(
            {expert * world // 128 for expert in cur}
            == {expert * world // 128 for expert in prev}
        )
    return {
        "route_order_equal_fraction": mean(ordered),
        "route_set_equal_fraction": mean(set_equal),
        "stable_branch_fraction": mean(stable),
        "owner_set_equal_fraction": mean(owner_equal),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instrumented_dir", type=Path)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_by_ep_request: dict[tuple[int, str], list[float]] = defaultdict(list)
    for path in args.clean_dir.glob("vanilla_ep*_restart*_rank0.json"):
        match = re.search(r"ep(\d+)", path.name)
        payload = json.loads(path.read_text())
        ep = int(match.group(1))
        for record in payload["records"]:
            clean_by_ep_request[(ep, str(record["id"]))].append(float(record["elapsed_s"]))

    stage_summaries = []
    temporal_export = []
    lag_export = []
    for rank0_path in sorted(args.instrumented_dir.glob("*_rank0.json")):
        match = re.search(r"ep(\d+)", rank0_path.name)
        if match is None:
            continue
        ep = int(match.group(1))
        rank_paths = sorted(args.instrumented_dir.glob(rank0_path.name.replace("_rank0.json", "_rank*.json")))
        payloads = [json.loads(path.read_text()) for path in rank_paths]
        rank0 = payloads[0]
        by_call: dict[int, list[dict]] = defaultdict(list)
        for payload in payloads:
            for row in payload["stage_rows"]:
                by_call[int(row["call_id"])].append(row)
        stage_totals = {stage: 0.0 for stage in STAGES}
        for rows in by_call.values():
            for stage in STAGES:
                stage_totals[stage] += max(float(row[f"{stage}_ms"]) for row in rows)
        request_wall_ms = float(rank0["request_wall_s"]) * 1000
        request_ids = [str(record["id"]) for record in rank0["records"]]
        clean_matched = [
            statistics.median(clean_by_ep_request[(ep, request_id)])
            for request_id in request_ids
        ]
        instrumented_latencies = [float(record["elapsed_s"]) for record in rank0["records"]]
        modules = rank0["module_event_summary"]
        summary = {
            "ep": ep,
            "rank_files": len(payloads),
            "requests": len(request_ids),
            "request_wall_ms": request_wall_ms,
            "instrumented_request_mean_ms": mean(instrumented_latencies) * 1000,
            "matched_clean_request_mean_ms": mean(clean_matched) * 1000,
            "observer_tax_fraction": mean(instrumented_latencies) / mean(clean_matched) - 1,
            "attention_sum_ms_rank0": float(modules["attention"]["sum_ms"]),
            "model_forward_sum_ms_rank0": float(modules["model_forward"]["sum_ms"]),
        }
        for stage in STAGES:
            summary[f"critical_{stage}_sum_ms"] = stage_totals[stage]
            summary[f"critical_{stage}_request_share"] = stage_totals[stage] / request_wall_ms
        stage_summaries.append(summary)

        route_rows = [
            row for row in rank0["stage_rows"]
            if row.get("forward_phase") == "refine" and "selected_experts" in row
        ]
        for row in route_rows:
            exported = {
                "ep": ep,
                "request_id": row["request_id"],
                "block_id": row["block_id"],
                "iteration": row["denoising_iteration"],
                "layer": row["layer"],
                "masked_positions": row["masked_positions"],
                "accepted_since_previous": row["accepted_since_previous"],
                "assignments": row["assignments"],
                "fanout": row["destination_rank_fanout"],
                "max_over_mean_rank_load": max(row["rank_assignment_counts"])
                / max(sum(row["rank_assignment_counts"]) / ep, 1e-12),
            }
            for lag in (1, 2, 4, 8):
                for output_name, source_name in (
                    ("route_set_equal_fraction", "route_set_equal_fraction"),
                    ("stable_branch_fraction", "stable_branch_fraction"),
                    ("owner_set_equal_fraction", "owner_set_equal_fraction"),
                    ("hidden_cosine_mean", "hidden_cosine_mean"),
                    ("hidden_rel_l2_mean", "hidden_rel_l2_mean"),
                    ("branch_output_cosine_mean", "stable_branch_output_cosine_mean"),
                    ("branch_output_rel_l2_mean", "stable_branch_output_rel_l2_mean"),
                    ("combined_output_cosine_mean", "combined_output_cosine_mean"),
                    ("combined_output_rel_l2_mean", "combined_output_rel_l2_mean"),
                    ("rank_load_cosine", "rank_load_cosine"),
                    ("same_critical_rank", "same_critical_rank"),
                ):
                    exported[f"lag{lag}_{output_name}"] = row.get(
                        f"lag{lag}_{source_name}"
                    )
            temporal_export.append(exported)
        grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
        for row in route_rows:
            grouped[(str(row["request_id"]), int(row["block_id"]), int(row["layer"]))].append(row)
        for (request_id, block_id, layer), rows in grouped.items():
            rows.sort(key=lambda row: int(row["denoising_iteration"]))
            for lag in (1, 2, 4, 8):
                for index in range(lag, len(rows)):
                    current = rows[index]
                    previous = rows[index - lag]
                    metrics = route_metrics(
                        current["selected_experts"], previous["selected_experts"], ep
                    )
                    current_hot = {
                        expert for expert, _ in Counter(
                            expert for route in current["selected_experts"] for expert in route
                        ).most_common(8)
                    }
                    previous_hot = {
                        expert for expert, _ in Counter(
                            expert for route in previous["selected_experts"] for expert in route
                        ).most_common(8)
                    }
                    lag_export.append({
                        "ep": ep,
                        "request_id": request_id,
                        "block_id": block_id,
                        "layer": layer,
                        "iteration": current["denoising_iteration"],
                        "lag": lag,
                        **metrics,
                        "top8_hot_expert_jaccard": len(current_hot & previous_hot)
                        / max(len(current_hot | previous_hot), 1),
                    })

    write_csv(args.output_dir / "instrumented_stage_summary.csv", stage_summaries)
    write_csv(args.output_dir / "temporal_rows.csv", temporal_export)
    write_csv(args.output_dir / "route_lag_metrics.csv", lag_export)
    temporal_summary = {
        "stage": stage_summaries,
        "lag": {
            str(lag): {
                metric: mean(float(row[metric]) for row in lag_export if row["lag"] == lag)
                for metric in (
                    "route_order_equal_fraction",
                    "route_set_equal_fraction",
                    "stable_branch_fraction",
                    "owner_set_equal_fraction",
                    "top8_hot_expert_jaccard",
                )
            }
            for lag in (1, 2, 4, 8)
        },
        "hidden_and_output_by_lag": {
            str(lag): {
                metric: mean(
                    float(row[f"lag{lag}_{metric}"]) for row in temporal_export
                    if row.get(f"lag{lag}_{metric}") is not None
                    and not math.isnan(float(row[f"lag{lag}_{metric}"]))
                )
                for metric in (
                    "hidden_cosine_mean",
                    "hidden_rel_l2_mean",
                    "branch_output_cosine_mean",
                    "branch_output_rel_l2_mean",
                    "combined_output_cosine_mean",
                    "combined_output_rel_l2_mean",
                    "rank_load_cosine",
                )
            }
            for lag in (1, 2, 4, 8)
        },
    }
    (args.output_dir / "temporal_summary.json").write_text(
        json.dumps(temporal_summary, indent=2)
    )
    print(json.dumps(temporal_summary, indent=2))


if __name__ == "__main__":
    main()
