"""Blindly compare EP2-derived virtual EP4 workload with physical EP4 routes.

The physical EP4 trace contains one source shard per EP rank.  This script
reconstructs the source/destination matrices from those source-local router
outputs and compares them with the frozen EP2-to-EP4 replay.  It deliberately
does not compare latency or fit a correction; this is the structural transfer
test that precedes EP4 cost-model fitting.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.constants import NUM_EXPERTS, TOP_K
from poc_mllm_ep_foundation.io import atomic_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _quantiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0, "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _physical_rows(root: Path) -> dict[tuple[tuple[str, ...], int], dict[str, Any]]:
    raw = root / "profile/raw_live"
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(raw.glob("rank[0-9]*.jsonl")):
        for row in _jsonl(path):
            if bool(row.get("measured", True)) and int(row.get("iteration", 0)) == 0:
                grouped[(int(row["wave"]), int(row["layer"]))].append(row)

    output = {}
    for (wave, layer), sources in grouped.items():
        if len(sources) != 4:
            raise AssertionError((wave, layer, len(sources)))
        assignment = np.zeros((4, 4), dtype=np.int64)
        unique = np.zeros_like(assignment)
        expert_histogram = np.zeros(NUM_EXPERTS, dtype=np.int64)
        request_ids = set()
        source_rows = np.zeros(4, dtype=np.int64)
        for source in sources:
            source_rank = int(source["ep_rank"])
            request_ids.add(str(source["request_id"]))
            route_path = raw / str(source["route_file"])
            with np.load(route_path) as archive:
                ids = archive["topk_ids"].astype(np.int64)
            if ids.ndim != 2 or ids.shape[1] != TOP_K:
                raise AssertionError((route_path, ids.shape))
            if np.any(ids < 0) or np.any(ids >= NUM_EXPERTS):
                raise AssertionError((route_path, int(ids.min()), int(ids.max())))
            source_rows[source_rank] = len(ids)
            destinations = ids // (NUM_EXPERTS // 4)
            np.add.at(expert_histogram, ids.reshape(-1), 1)
            for index in range(len(ids)):
                np.add.at(assignment[source_rank], destinations[index], 1)
                np.add.at(unique[source_rank], np.unique(destinations[index]), 1)
        key = (tuple(sorted(request_ids)), layer)
        output[key] = {
            "wave": wave,
            "layer": layer,
            "request_ids": sorted(request_ids),
            "source_rows": source_rows,
            "assignment": assignment,
            "unique": unique,
            "expert_histogram": expert_histogram,
        }
    return output


def _relative_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).sum() / max(1, int(right.sum())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--virtual-replay", type=Path, required=True)
    parser.add_argument("--physical-ep4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replay = json.loads(args.virtual_replay.read_text())
    if int(replay["target_ep"]) != 4:
        raise ValueError("virtual replay must target EP4")
    physical = _physical_rows(args.physical_ep4_root)
    rows = []
    missing = []
    for predicted in replay["rows"]:
        key = (tuple(sorted(map(str, predicted["request_ids"]))), int(predicted["layer"]))
        actual = physical.get(key)
        if actual is None:
            missing.append({"request_ids": list(key[0]), "layer": key[1]})
            continue
        predicted_assignment = np.asarray(predicted["assignment_matrix"], dtype=np.int64)
        predicted_unique = np.asarray(predicted["unique_activation_matrix"], dtype=np.int64)
        predicted_histogram = np.asarray(predicted["expert_histogram"], dtype=np.int64)
        actual_histogram = actual["expert_histogram"]
        rows.append({
            "request_ids": list(key[0]),
            "layer": key[1],
            "predicted_wave": int(predicted["wave"]),
            "physical_wave": int(actual["wave"]),
            "assignment_relative_l1": _relative_l1(predicted_assignment, actual["assignment"]),
            "unique_row_relative_l1": _relative_l1(predicted_unique, actual["unique"]),
            "expert_histogram_relative_l1": _relative_l1(predicted_histogram, actual_histogram),
            "assignment_count_delta": int(predicted_assignment.sum() - actual["assignment"].sum()),
            "source_row_delta": (
                predicted_assignment.sum(axis=1) // TOP_K - actual["source_rows"]
            ).astype(int).tolist(),
        })

    result = {
        "schema_version": "mllm_ep2_to_ep4_structural_transfer.v1",
        "evidence": {
            "prediction": "EP2_SOURCE_TRACE_REMAPPED_TO_TP2_DP2_EP4",
            "truth": "ACTUAL_PHYSICAL_TP2_DP2_EP4_SOURCE_ROUTER_TRACE",
            "latency_scope": "NOT_INCLUDED",
        },
        "matched_invocations": len(rows),
        "missing_invocations": missing,
        "metrics": {
            name: _quantiles([float(row[name]) for row in rows])
            for name in (
                "assignment_relative_l1",
                "unique_row_relative_l1",
                "expert_histogram_relative_l1",
            )
        },
        "exact_assignment_count_fraction": (
            float(np.mean([row["assignment_count_delta"] == 0 for row in rows])) if rows else 0.0
        ),
        "rows": rows,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "matched_invocations", "missing_invocations", "metrics",
        "exact_assignment_count_fraction",
    )}, indent=2))


if __name__ == "__main__":
    main()
