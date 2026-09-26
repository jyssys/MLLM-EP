from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.constants import DTYPE_BYTES, HIDDEN_SIZE, NUM_EXPERTS, TOP_K
from poc_mllm_ep_foundation.io import atomic_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _source_parts(rows: int, dp_rank: int, target_ep: int) -> np.ndarray:
    if target_ep == 2:
        return np.full(rows, dp_rank, dtype=np.int16)
    # Target family is TP2 with DP=target_ep/2.  The fixed anchor wave has two
    # requests, so DP0 and DP1 are active while later DP ranks are explicitly
    # idle for EP8.  Within a request, sequence-parallel TP source ownership is
    # reconstructed as contiguous balanced row shards.
    midpoint = (rows + 1) // 2
    source = np.empty(rows, dtype=np.int16)
    source[:midpoint] = dp_rank * 2
    source[midpoint:] = dp_rank * 2 + 1
    return source


def _stats(values: np.ndarray) -> dict[str, float | int]:
    mean = float(values.mean()) if len(values) else 0.0
    ordered = np.sort(values)
    return {
        "mean": mean,
        "max": int(values.max(initial=0)),
        "second_max": int(ordered[-2]) if len(ordered) > 1 else int(ordered[-1]),
        "max_mean": float(values.max(initial=0) / mean) if mean else 0.0,
        "cv": float(values.std() / mean) if mean else 0.0,
    }


def replay(trace_root: Path, target_ep: int) -> dict[str, Any]:
    if target_ep not in (2, 4, 8):
        raise ValueError(target_ep)
    raw = trace_root / "profile/raw_live"
    selected: dict[tuple[int, int, int], dict[str, Any]] = {}
    for rank_file in sorted(raw.glob("rank[0-9]*.jsonl")):
        for row in _jsonl(rank_file):
            if not row.get("measured", True) or int(row.get("iteration", 0)) != 0:
                continue
            key = (int(row["wave"]), int(row["layer"]), int(row["worker_dp_rank"]))
            selected[key] = row
    invocations: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for (wave, layer, _), row in selected.items():
        invocations[(wave, layer)].append(row)

    local_experts = NUM_EXPERTS // target_ep
    output_rows = []
    for (wave, layer), sources in sorted(invocations.items()):
        assignment = np.zeros((target_ep, target_ep), dtype=np.int64)
        unique = np.zeros_like(assignment)
        source_rows = np.zeros(target_ep, dtype=np.int64)
        expert_hist = np.zeros(NUM_EXPERTS, dtype=np.int64)
        request_ids = []
        for source in sources:
            request_ids.append(str(source["request_id"]))
            with np.load(raw / source["route_file"]) as archive:
                ids = archive["topk_ids"].astype(np.int64)
            if ids.ndim != 2 or ids.shape[1] != TOP_K:
                raise AssertionError(ids.shape)
            src = _source_parts(len(ids), int(source["worker_dp_rank"]), target_ep)
            source_rows += np.bincount(src, minlength=target_ep)
            destinations = ids // local_experts
            np.add.at(expert_hist, ids.reshape(-1), 1)
            for row_index in range(len(ids)):
                s = int(src[row_index])
                for destination in destinations[row_index]:
                    assignment[s, int(destination)] += 1
                for destination in np.unique(destinations[row_index]):
                    unique[s, int(destination)] += 1
        rank_load = np.add.reduceat(expert_hist, np.arange(0, NUM_EXPERTS, local_experts))
        offdiag = ~np.eye(target_ep, dtype=bool)
        outgoing_peers = (unique > 0).sum(1) - (np.diag(unique) > 0)
        output_rows.append({
            "wave": wave, "layer": layer, "request_ids": sorted(request_ids),
            "assignment_matrix": assignment.tolist(),
            "unique_activation_matrix": unique.tolist(),
            "source_logical_rows": source_rows.tolist(),
            "logical_rows": int(source_rows.sum()),
            "rank_assignment_load": rank_load.tolist(),
            "rank_load_stats": _stats(rank_load),
            "expert_histogram": expert_hist.tolist(),
            "active_experts": int((expert_hist > 0).sum()),
            "logical_dispatch_bytes": int(unique[offdiag].sum() * HIDDEN_SIZE * DTYPE_BYTES),
            "logical_combine_bytes": int(unique[offdiag].sum() * HIDDEN_SIZE * DTYPE_BYTES),
            "remote_unique_rows": int(unique[offdiag].sum()),
            "outgoing_peer_count": outgoing_peers.tolist(),
            "max_fanout": int(outgoing_peers.max(initial=0)),
            "active_source_ranks": sorted(np.flatnonzero(assignment.sum(1)).astype(int).tolist()),
            "idle_source_ranks": sorted(np.flatnonzero(assignment.sum(1) == 0).astype(int).tolist()),
            "assignment_invariant": int(assignment.sum()),
        })
    if not output_rows:
        raise RuntimeError(f"no measured iteration-0 routes under {trace_root}")
    return {
        "schema_version": "mllm_virtual_ep_replay.v2",
        "trace_root": str(trace_root.resolve()),
        "target_ep": target_ep,
        "target_topology": {"tp": 1 if target_ep == 2 else 2,
                            "dp": target_ep if target_ep == 2 else target_ep // 2,
                            "ep": target_ep, "pp": 1, "sp": 1},
        "owner_rule": f"contiguous {local_experts} experts/rank",
        "source_rule": "TP2 contiguous balanced row shards; anchor DP0/DP1 active; EP8 DP2/DP3 idle",
        "logical_activation_bytes_per_row": HIDDEN_SIZE * DTYPE_BYTES,
        "transport_warning": "logical bytes only; backend metadata, padding and physical copies require calibrated cost model",
        "status": ("PHYSICAL_TRACE_REMAP" if target_ep in (2, 4)
                   else "STRUCTURAL_VERIFIED_UNVALIDATED_EP8_COMM"),
        "rows": output_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--target-ep", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = replay(args.trace_root, args.target_ep)
    atomic_json(args.output, value)
    print(json.dumps({"output": str(args.output), "target_ep": args.target_ep,
                      "invocations": len(value["rows"])}))


if __name__ == "__main__":
    main()
