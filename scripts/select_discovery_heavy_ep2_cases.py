#!/usr/bin/env python3
"""Select full-physical-row H1/H4 cases from retained heavy traces.

The aggregate discovery trace deliberately retains only exact routes for the
current 32-position block.  Physical validation, however, should replay the
full vanilla routed workload.  This utility loads the small retained heavy
audit cohort and emits representative full-row route states for true EP2.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from virtual_ep.compute_model import ComputeModel
from virtual_ep.schema import TraceBundle, invocation_slices


NUM_EXPERTS = 256
LOCAL_EXPERTS = 128


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def rank_times(routes: np.ndarray, model: ComputeModel) -> np.ndarray:
    histogram = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
    return np.asarray([
        model.predict(histogram[rank * LOCAL_EXPERTS:(rank + 1) * LOCAL_EXPERTS])[0]
        for rank in range(2)
    ])


def block_number_map(trace: TraceBundle) -> dict[int, dict[int, int]]:
    blocks = defaultdict(set)
    for request, block in zip(trace.rows["request_id"], trace.rows["block_id"]):
        blocks[int(request)].add(int(block))
    return {
        request: {block: ordinal for ordinal, block in enumerate(sorted(values))}
        for request, values in blocks.items()
    }


def choose_h1(invocations: dict, model: ComputeModel) -> list[tuple[dict, np.ndarray]]:
    groups = defaultdict(list)
    for (request, block, iteration, layer), routes in invocations.items():
        if layer == 10:
            times = rank_times(routes, model)
            groups[(request, block)].append((iteration, routes, times))
    scored = []
    for block, rows in groups.items():
        rows.sort(key=lambda item: item[0])
        if len(rows) < 3:
            continue
        critical = [int(np.argmax(item[2])) for item in rows]
        persistence = float(np.mean(np.asarray(critical[1:]) == np.asarray(critical[:-1])))
        imbalance = float(np.mean([
            item[2].max() / max(item[2].mean(), 1e-9) for item in rows
        ]))
        scored.append((persistence, imbalance, block, rows))
    if not scored:
        raise RuntimeError("no H1 heavy blocks with at least three refinements")
    choices = [
        ("low_persistence", min(scored, key=lambda row: (row[0], row[1]))),
        ("high_persistence", max(scored, key=lambda row: (row[0], row[1]))),
        ("near_balanced", min(scored, key=lambda row: row[1])),
        ("high_imbalance", max(scored, key=lambda row: row[1])),
    ]
    result = []
    for category, (persistence, imbalance, block, rows) in choices:
        for iteration, routes, times in rows[:3]:
            result.append(({
                "hypothesis": "H1", "category": category,
                "request": block[0], "block": block[1],
                "iteration": iteration, "layer": 10,
                "block_predicted_persistence": persistence,
                "block_predicted_max_mean": imbalance,
                "physical_rows": len(routes),
                "structural_critical_rank_ep2": int(np.argmax(times)),
            }, routes))
    return result


def choose_h4(invocations: dict, model: ComputeModel) -> list[tuple[dict, np.ndarray]]:
    # The online signature uses only the first two refinements, aggregated over
    # all routed layers.  Validation replays a matched full-row layer-10 state.
    early = defaultdict(lambda: defaultdict(lambda: np.zeros(2, dtype=np.float64)))
    candidate_state = {}
    for (request, block, iteration, layer), routes in invocations.items():
        if iteration < 2:
            owners = routes // LOCAL_EXPERTS
            early[(request, block)][iteration] += np.bincount(owners.reshape(-1), minlength=2)
        if layer == 10 and iteration >= 2:
            candidate_state.setdefault((request, block), (iteration, routes))
    signatures = {key: sum(values.values(), np.zeros(2)) for key, values in early.items()
                  if key in candidate_state and len(values) >= 2}
    pairs = []
    keys = sorted(signatures)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            if left[0] == right[0]:
                continue
            left_routes = candidate_state[left][1]; right_routes = candidate_state[right][1]
            rows = len(left_routes) + len(right_routes)
            similarity = cosine(signatures[left], signatures[right])
            aggregate = signatures[left] + signatures[right]
            complement = float(aggregate.max() / max(aggregate.mean(), 1e-9))
            pairs.append({
                "blocks": (left, right), "rows": rows, "similarity": similarity,
                "complement": complement, "routes": np.concatenate([left_routes, right_routes]),
                "iteration": (candidate_state[left][0], candidate_state[right][0]),
            })
    if len(pairs) < 3:
        raise RuntimeError("not enough cross-request heavy pairs for H4")
    # Restrict to the central row-count band so policy timing is not a proxy
    # for sequence length.  All selected cases use identical layer identity.
    row_values = np.asarray([row["rows"] for row in pairs])
    lower, upper = np.percentile(row_values, [35, 65])
    matched = [row for row in pairs if lower <= row["rows"] <= upper]
    if len(matched) < 3:
        matched = pairs
    target_rows = float(np.median([row["rows"] for row in matched]))
    row_scale = max(target_rows, 1.0)
    similar = sorted(
        matched,
        key=lambda row: row["similarity"] - abs(row["rows"] - target_rows) / row_scale,
        reverse=True,
    )[:3]
    complementary = sorted(
        matched,
        key=lambda row: row["complement"] + abs(row["rows"] - target_rows) / row_scale,
    )[:3]
    rng = np.random.default_rng(20260917)
    random = [matched[index] for index in rng.choice(
        len(matched), size=min(3, len(matched)), replace=False
    )]
    result = []
    for policy, selected in (("random", random), ("similar", similar),
                             ("complementary", complementary)):
        for row in selected:
            times = rank_times(row["routes"], model)
            result.append(({
                "hypothesis": "H4", "policy": policy,
                "blocks": [list(key) for key in row["blocks"]],
                "iterations": list(row["iteration"]), "layer": 10,
                "physical_rows": row["rows"], "early_signature_cosine": row["similarity"],
                "early_signature_max_mean": row["complement"],
                "structural_critical_rank_ep2": int(np.argmax(times)),
            }, row["routes"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heavy", type=Path, required=True)
    parser.add_argument("--compute-ep2", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requested = set(args.requests)
    trace = TraceBundle.load(args.heavy).select_requests(requested)
    mapping = block_number_map(trace)
    invocations = {}
    for key, selected in invocation_slices(trace.rows):
        request, old_block, iteration, _nfe, layer = key
        invocations[(request, mapping[request][old_block], iteration, layer)] = (
            trace.rows["expert_ids"][selected].astype(np.int16, copy=False)
        )
    model = ComputeModel.from_csv(args.compute_ep2)
    cases = choose_h1(invocations, model) + choose_h4(invocations, model)
    offsets = [0]; chunks = []; labels = []
    for label, routes in cases:
        chunks.append(routes); offsets.append(offsets[-1] + len(routes)); labels.append(label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, expert_ids=np.concatenate(chunks), offsets=np.asarray(offsets),
        labels_json=np.asarray(json.dumps(labels), dtype=np.str_),
    )
    args.output.with_suffix(".json").write_text(json.dumps({
        "source": str(args.heavy), "heavy_requests": sorted(requested),
        "row_semantics": "full vanilla physical rows", "cases": labels,
    }, indent=2) + "\n")
    print(json.dumps({"cases": len(labels), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
