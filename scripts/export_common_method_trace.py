#!/usr/bin/env python3
"""Export compact per-method routes into one common fresh/reuse schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEMANTIC = {"CURRENT_BLOCK_MASKED": 0, "NEWLY_ACCEPTED": 1, "DECODED": 2}
EXECUTION = {"FRESH": 0, "REUSED": 1}


def load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files if key != "metadata_json"}
        metadata = json.loads(str(z["metadata_json"]))
    return arrays, metadata


def selective_rows(path: Path, method: str):
    z, metadata = load(path); request = int(metadata["request_id"])
    weights_available = "route_weights" in z
    weights = z.get("route_weights", np.full(z["routes"].shape, np.nan, dtype=np.float16))
    rows = []
    last_route = np.full(z["routes"].shape[1:], -1, dtype=np.int16)
    last_weight = np.zeros(z["routes"].shape[1:], dtype=np.float16)
    last_ref = np.full(32, -1, dtype=np.int16); ages = np.zeros(32, dtype=np.int16)
    prior_block = None
    for index in range(len(z["block_id"])):
        block = int(z["block_id"][index]); refinement = int(z["iteration_id"][index])
        if block != prior_block:
            last_route.fill(-1); last_weight.fill(0); last_ref.fill(-1); ages.fill(0)
        masked = z["masked"][index]; active = z["active"][index]
        newly = z["newly_decoded"][index]
        reused = masked & ~active
        fresh = ~reused
        current_route = z["routes"][index].copy(); current_weight = weights[index].copy()
        if np.any(reused):
            current_route[:, reused] = last_route[:, reused]
            current_weight[:, reused] = last_weight[:, reused]
        semantic = np.full(32, SEMANTIC["DECODED"], dtype=np.int8)
        semantic[masked] = SEMANTIC["CURRENT_BLOCK_MASKED"]
        semantic[newly] = SEMANTIC["NEWLY_ACCEPTED"]
        for layer_offset, layer_id in enumerate(metadata["routed_layer_ids"]):
            for position in range(32):
                rows.append((
                    request, block, refinement, int(layer_id), block * 32 + position,
                    semantic[position], EXECUTION["REUSED"] if reused[position] else EXECUTION["FRESH"],
                    method, current_route[layer_offset, position], current_weight[layer_offset, position],
                    bool(fresh[position]), bool(reused[position]),
                    int(last_ref[position]) if reused[position] else -1,
                    int(ages[position]),
                ))
        last_route[:, fresh] = z["routes"][index][:, fresh]
        last_weight[:, fresh] = weights[index][:, fresh]
        last_ref[fresh] = refinement
        ages[reused] += 1; ages[fresh] = 0
        prior_block = block
    return rows, weights_available


def team_rows(path: Path, method: str):
    z, metadata = load(path); request = int(metadata.get("request_id", path.stem.split("_")[-1]))
    rows = []; last_ref = np.full(32, -1, dtype=np.int16); prior_block = None
    for index in range(len(z["block_id"])):
        block = int(z["block_id"][index]); refinement = int(z["iteration_id"][index])
        if block != prior_block: last_ref.fill(-1)
        fresh = z["fresh"][index]; reused = z["reused"][index]
        for layer_offset, layer_id in enumerate(metadata["routed_layer_ids"]):
            for position in range(32):
                rows.append((
                    request, block, refinement, int(layer_id), block * 32 + position,
                    int(z["semantic_state"][index, position]),
                    EXECUTION["REUSED"] if reused[position] else EXECUTION["FRESH"],
                    method, z["routes"][index, layer_offset, position],
                    z["route_weights"][index, layer_offset, position],
                    bool(fresh[position]), bool(reused[position]),
                    int(last_ref[position]) if reused[position] else -1, 0,
                ))
        last_ref[fresh] = refinement; prior_block = block
    return rows, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("selective", "team"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_rows = []; weight_flags = []
    reader = selective_rows if args.format == "selective" else team_rows
    for path in sorted(args.trace_dir.glob("request_*.npz")):
        rows, weights = reader(path, args.method)
        all_rows.extend(rows); weight_flags.append(weights)
    if not all_rows: raise RuntimeError("no request traces found")
    scalars = list(zip(*[row[:8] for row in all_rows]))
    payload = {
        "request_id": np.asarray(scalars[0], dtype=np.int16),
        "block_id": np.asarray(scalars[1], dtype=np.int16),
        "refinement_id": np.asarray(scalars[2], dtype=np.int16),
        "layer_id": np.asarray(scalars[3], dtype=np.int8),
        "token_position": np.asarray(scalars[4], dtype=np.int16),
        "semantic_state": np.asarray(scalars[5], dtype=np.int8),
        "execution_state": np.asarray(scalars[6], dtype=np.int8),
        "method": np.asarray(scalars[7], dtype="U32"),
        "selected_expert_ids": np.stack([row[8] for row in all_rows]).astype(np.int16),
        "selected_router_weights": np.stack([row[9] for row in all_rows]).astype(np.float16),
        "is_fresh_route": np.asarray([row[10] for row in all_rows], dtype=bool),
        "is_reused_route": np.asarray([row[11] for row in all_rows], dtype=bool),
        "reuse_source_refinement": np.asarray([row[12] for row in all_rows], dtype=np.int16),
        "freeze_age": np.asarray([row[13] for row in all_rows], dtype=np.int8),
    }
    metadata = {
        "method_schema_version": 2, "method": args.method,
        "semantic_state_encoding": SEMANTIC, "execution_state_encoding": EXECUTION,
        "router_weights_complete": bool(all(weight_flags)),
        "variable_k_padding": {"expert_id": -1, "router_weight": 0.0},
        "rows": len(all_rows), "requests": len(set(payload["request_id"].tolist())),
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
