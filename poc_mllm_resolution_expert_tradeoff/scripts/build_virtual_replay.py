from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


ROUTE_RE = re.compile(
    r"route_pid(?P<pid>\d+)_f(?P<forward>\d+)_ep(?P<source>\d+)_layer(?P<layer>\d+)\.npy"
)


def _cv(values: np.ndarray) -> float:
    mean = float(values.mean())
    return float(values.std() / mean) if mean else 0.0


def _summary(trace_dir: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows = {}
    for path in trace_dir.glob("route_summary.pid*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[(int(row["pid"]), int(row["forward_id"]), int(row["layer"]))] = row
    return rows


def build(trace_dir: Path, target_ep: int, request_id: str) -> dict[str, Any]:
    if target_ep not in (4, 8):
        raise ValueError("Part 2 replays target EP4 or EP8")
    summaries = _summary(trace_dir)
    grouped: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for path in sorted((trace_dir / "raw" / "routes").glob("*.npy")):
        match = ROUTE_RE.fullmatch(path.name)
        if match is None:
            continue
        key = (int(match.group("forward")), int(match.group("layer")))
        grouped.setdefault(key, []).append((
            int(match.group("source")), np.load(path, allow_pickle=False).astype(np.int64)
        ))
    if not grouped:
        raise RuntimeError(f"no exact route tensors under {trace_dir}")

    # The primary Part-2 question is visual prefill. Keep complete TP source
    # shards for any forward whose companion summary observes visual rows.
    visual_forwards = {
        (forward, layer)
        for (_, forward, layer), row in summaries.items()
        if int(row.get("visual_rows", 0)) > 0
    }
    local = 128 // target_ep
    output = []
    for (forward, layer), shards in sorted(grouped.items()):
        if (forward, layer) not in visual_forwards:
            continue
        expert_hist = np.zeros(128, dtype=np.int64)
        assignment = np.zeros((target_ep, target_ep), dtype=np.int64)
        unique = np.zeros((target_ep, target_ep), dtype=np.int64)
        source_rows = np.zeros(target_ep, dtype=np.int64)
        for physical_source, ids in shards:
            # For the fixed batch-one diagnostic, the TP2 source shard remains
            # rank 0/1 in the target TP2/DP2 or TP2/DP4 topology.
            source = physical_source
            valid = ids >= 0
            flat = ids[valid]
            expert_hist += np.bincount(flat, minlength=128)
            owners = np.where(valid, ids // local, -1)
            source_rows[source] += len(ids)
            for destination in range(target_ep):
                assignment[source, destination] += int(np.count_nonzero(owners == destination))
                unique[source, destination] += int(
                    np.count_nonzero(np.any(owners == destination, axis=1))
                )
        rank_load = assignment.sum(0)
        remote = unique.copy()
        np.fill_diagonal(remote, 0)
        remote_rows = int(remote.sum())
        outgoing = np.count_nonzero(remote, axis=1)
        mean = float(rank_load.mean())
        ordered = np.sort(rank_load)
        output.append({
            "wave": forward,
            "layer": layer,
            "request_ids": [request_id],
            "logical_rows": int(source_rows.sum()),
            "source_logical_rows": source_rows.astype(int).tolist(),
            "expert_histogram": expert_hist.astype(int).tolist(),
            "assignment_matrix": assignment.astype(int).tolist(),
            "unique_activation_matrix": unique.astype(int).tolist(),
            "active_experts": int(np.count_nonzero(expert_hist)),
            "active_source_ranks": np.flatnonzero(source_rows).astype(int).tolist(),
            "idle_source_ranks": np.flatnonzero(source_rows == 0).astype(int).tolist(),
            "assignment_invariant": int(expert_hist.sum()),
            "rank_assignment_load": rank_load.astype(int).tolist(),
            "rank_load_stats": {
                "mean": mean,
                "max": int(rank_load.max(initial=0)),
                "second_max": int(ordered[-2]) if len(ordered) > 1 else int(ordered[-1]),
                "max_mean": float(rank_load.max(initial=0) / mean) if mean else 0.0,
                "cv": _cv(rank_load),
            },
            "remote_unique_rows": remote_rows,
            "logical_dispatch_bytes": remote_rows * 2048 * 2,
            "logical_combine_bytes": remote_rows * 2048 * 2,
            "outgoing_peer_count": outgoing.astype(int).tolist(),
            "max_fanout": int(outgoing.max(initial=0)),
        })
    return {
        "schema_version": "mllm_ep_replay.v4",
        "status": "STRUCTURAL_VERIFIED_FROM_POLICY_SPECIFIC_ACTUAL_ROUTES",
        "trace_root": str(trace_dir.resolve()),
        "target_ep": target_ep,
        "target_topology": (
            "TP2_DP2_EP4_PP1" if target_ep == 4 else "TP2_DP4_EP8_PP1"
        ),
        "source_rule": "fixed batch-one request uses TP source ranks 0/1; remaining DP ranks idle",
        "owner_rule": f"contiguous {local} experts/rank",
        "logical_activation_bytes_per_row": 4096,
        "transport_warning": (
            None if target_ep == 4
            else "EP8 communication and E2E are unvalidated extrapolation"
        ),
        "rows": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--target-ep", type=int, choices=(4, 8), required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.trace_dir, args.target_ep, args.request_id)
    atomic_json(args.output, result)
    print(json.dumps({"rows": len(result["rows"]), "target_ep": args.target_ep}, indent=2))


if __name__ == "__main__":
    main()
