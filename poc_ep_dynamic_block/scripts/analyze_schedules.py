#!/usr/bin/env python3
"""Aggregate actual-trajectory block schedules and quality gates."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIR_RE = re.compile(
    r"schedule_(?P<plan>.+)_(?P<task>gsm8k|humaneval)_n(?P<n>\d+)_m(?P<mini>\d+)_"
    r"L(?P<target>\d+)_t(?P<threshold>[0-9.]+)_r(?P<repeat>\d+)"
    r"(?:_(?P<barrier>barrier1))?$"
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def is_fixed(schedule: list[int]) -> bool:
    return len(set(int(value) for value in schedule)) == 1


def main() -> None:
    rows = []
    for directory in sorted((ROOT / "results" / "schedules").glob("schedule_*")):
        match = DIR_RE.fullmatch(directory.name)
        quality_path = directory / "schedule_quality.json"
        if not match or not quality_path.exists():
            continue
        metadata = match.groupdict()
        for policy in json.loads(quality_path.read_text()):
            schedule = [int(value) for value in policy.get("block_schedule", [])]
            details = policy.get("details", [])
            rows.append(
                {
                    **metadata,
                    "n": int(metadata["n"]),
                    "mini": int(metadata["mini"]),
                    "target": int(metadata["target"]),
                    "threshold": float(metadata["threshold"]),
                    "repeat": int(metadata["repeat"]),
                    "schedule_barrier": metadata.get("barrier") == "barrier1",
                    "policy": policy["policy"],
                    "block_schedule": json.dumps(schedule, separators=(",", ":")),
                    "fixed": is_fixed(schedule),
                    "model_seconds": float(policy["model_seconds"]),
                    "elapsed_seconds": float(policy["elapsed_seconds"]),
                    "nfe": int(policy["nfe"]),
                    "generated_tokens": int(policy["generated_tokens"]),
                    "correct": int(policy["correct"]),
                    "total": int(policy["total"]),
                    "accuracy": float(policy["accuracy"]),
                    "passing_ids": json.dumps(
                        [int(detail["id"]) for detail in details if detail["pass"]],
                        separators=(",", ":"),
                    ),
                    "evidence": "clean actual trajectory; no independent-block summation",
                    "result_dir": str(directory.relative_to(ROOT.parent)),
                }
            )
    write_csv(ROOT / "POLICY_COMPARISON.csv", rows)

    oracle_rows = []
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task and row["n"] >= 8]
        for plan, repeat in sorted({(row["plan"], row["repeat"]) for row in task_rows}):
            run = [row for row in task_rows if row["plan"] == plan and row["repeat"] == repeat]
            fixed = [row for row in run if row["fixed"]]
            dynamic = [row for row in run if not row["fixed"]]
            if not fixed or not dynamic:
                continue
            fastest = min(fixed, key=lambda row: row["model_seconds"])
            max_quality = max(row["correct"] for row in fixed)
            quality_static = min(
                (row for row in fixed if row["correct"] == max_quality),
                key=lambda row: row["model_seconds"],
            )
            baseline_pass = set(json.loads(quality_static["passing_ids"]))

            def choose(candidates: list[dict]) -> dict | None:
                return min(candidates, key=lambda row: row["model_seconds"]) if candidates else None

            o0 = choose(dynamic)
            o1 = choose([row for row in dynamic if row["correct"] >= quality_static["correct"]])
            o2 = choose(
                [
                    row for row in dynamic
                    if baseline_pass.issubset(set(json.loads(row["passing_ids"])))
                ]
            )
            o3 = choose([row for row in dynamic if row["correct"] >= quality_static["correct"] - 1])
            for oracle, candidate, baseline in (
                ("O0_latency_only", o0, fastest),
                ("O1_dataset_score_preserving", o1, quality_static),
                ("O2_per_request_safe", o2, quality_static),
                ("O3_one_sample_epsilon", o3, quality_static),
            ):
                oracle_rows.append(
                    {
                        "task": task,
                        "plan": plan,
                        "repeat": repeat,
                        "mini_batch_size": baseline["mini"],
                        "oracle": oracle,
                        "baseline_policy": baseline["policy"],
                        "baseline_schedule": baseline["block_schedule"],
                        "baseline_seconds": baseline["model_seconds"],
                        "baseline_correct": baseline["correct"],
                        "candidate_policy": candidate["policy"] if candidate else "NONE",
                        "candidate_schedule": candidate["block_schedule"] if candidate else "[]",
                        "candidate_seconds": candidate["model_seconds"] if candidate else "",
                        "candidate_correct": candidate["correct"] if candidate else "",
                        "gain_percent": (
                            100 * (baseline["model_seconds"] - candidate["model_seconds"])
                            / baseline["model_seconds"]
                            if candidate else ""
                        ),
                        "actual_trajectory": True,
                    }
                )
    write_csv(ROOT / "DYNAMIC_SCHEDULE_ORACLE.csv", oracle_rows)

    request_rows = []
    for directory in sorted((ROOT / "results" / "schedules").glob("schedule_*")):
        match = DIR_RE.fullmatch(directory.name)
        if not match:
            continue
        for path in sorted(directory.glob("policy_*.json")):
            payload = json.loads(path.read_text())
            for row in payload.get("rows", []):
                if int(row.get("physical_batch_size", 0)) != 1:
                    continue
                request_rows.append(
                    {
                        **match.groupdict(),
                        "policy": payload["policy"].get("name", path.stem),
                        "block_schedule": json.dumps(payload["policy"].get("block_schedule", []), separators=(",", ":")),
                        "request_id": int(row["original_index"]),
                        "request_model_seconds": float(row["physical_batch_model_seconds"]),
                        "request_nfe": int(row["physical_batch_nfe"]),
                        "generated_length": int(row["generated_length"]),
                    }
                )
    write_csv(ROOT / "REQUEST_SCHEDULE_RESULTS.csv", request_rows)
    print(f"wrote {len(rows)} policy rows, {len(oracle_rows)} oracle rows, {len(request_rows)} request rows")


if __name__ == "__main__":
    main()
