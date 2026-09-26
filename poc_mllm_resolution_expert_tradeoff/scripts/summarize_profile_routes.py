"""Summarize exact Part-2 routes captured by the low-overhead EP profiler.

The EP profiler records the actual post-adapter expert IDs for every physical
source rank.  This converter uses one deterministic measured repeat, verifies
the mixed-k accounting against the full-model result metadata, and emits both
the physical EP4 structural summary and target EP4/EP8 replay contracts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _cv(values: np.ndarray) -> float:
    mean = float(values.mean()) if values.size else 0.0
    return float(values.std() / mean) if mean else 0.0


def _replay_row(
    items: list[tuple[int, dict[str, Any], np.ndarray]],
    *,
    target_ep: int,
) -> dict[str, Any]:
    local_experts = 128 // target_ep
    histogram = np.zeros(128, dtype=np.int64)
    assignment = np.zeros((target_ep, target_ep), dtype=np.int64)
    unique = np.zeros((target_ep, target_ep), dtype=np.int64)
    source_rows = np.zeros(target_ep, dtype=np.int64)
    for source, _, ids in items:
        if source >= target_ep:
            raise RuntimeError(f"source rank {source} is outside target EP{target_ep}")
        valid = ids >= 0
        selected = ids[valid]
        histogram += np.bincount(selected, minlength=128)
        owners = np.where(valid, ids // local_experts, -1)
        source_rows[source] += int(ids.shape[0])
        for destination in range(target_ep):
            assignment[source, destination] += int(np.count_nonzero(owners == destination))
            unique[source, destination] += int(
                np.count_nonzero(np.any(owners == destination, axis=1))
            )
    loads = assignment.sum(axis=0)
    ordered = np.sort(loads)
    remote = unique.copy()
    np.fill_diagonal(remote, 0)
    remote_rows = int(remote.sum())
    mean = float(loads.mean())
    return {
        "source_logical_rows": source_rows.astype(int).tolist(),
        "expert_histogram": histogram.astype(int).tolist(),
        "assignment_matrix": assignment.astype(int).tolist(),
        "unique_activation_matrix": unique.astype(int).tolist(),
        "active_experts": int(np.count_nonzero(histogram)),
        "active_source_ranks": np.flatnonzero(source_rows).astype(int).tolist(),
        "idle_source_ranks": np.flatnonzero(source_rows == 0).astype(int).tolist(),
        "assignment_invariant": int(histogram.sum()),
        "rank_assignment_load": loads.astype(int).tolist(),
        "rank_load_stats": {
            "mean": mean,
            "max": int(loads.max(initial=0)),
            "second_max": int(ordered[-2]) if len(ordered) > 1 else int(ordered[-1]),
            "max_mean": float(loads.max(initial=0) / mean) if mean else 0.0,
            "cv": _cv(loads),
        },
        "remote_unique_rows": remote_rows,
        "logical_dispatch_bytes": remote_rows * 2048 * 2,
        "logical_combine_bytes": remote_rows * 2048 * 2,
        "outgoing_peer_count": np.count_nonzero(remote, axis=1).astype(int).tolist(),
        "max_fanout": int(np.count_nonzero(remote, axis=1).max(initial=0)),
    }


def summarize(profile_root: Path, config: str, iteration: int) -> dict[str, Any]:
    result_rows = _jsonl(profile_root / "results.jsonl")
    metadata = {str(row["request_id"]): row for row in result_rows}
    raw = profile_root / "profile" / "raw_live"
    grouped: dict[tuple[int, int], list[tuple[int, dict[str, Any], np.ndarray]]] = defaultdict(list)
    for rank in range(4):
        path = raw / f"rank{rank}.jsonl"
        if not path.exists():
            raise RuntimeError(f"missing physical rank trace: {path}")
        for row in _jsonl(path):
            if not bool(row.get("measured", True)) or int(row.get("iteration", 0)) != iteration:
                continue
            route_path = raw / str(row["route_file"])
            ids = np.load(route_path, allow_pickle=False)["topk_ids"].astype(np.int64)
            grouped[(int(row["wave"]), int(row["layer"]))].append((rank, row, ids))
    if not grouped:
        raise RuntimeError(f"no measured route rows under {profile_root}")

    physical_rows: list[dict[str, Any]] = []
    replay: dict[int, list[dict[str, Any]]] = {4: [], 8: []}
    for (wave, layer), items in sorted(grouped.items()):
        if len(items) != 4 or {rank for rank, _, _ in items} != set(range(4)):
            raise RuntimeError(f"wave={wave}, layer={layer}: expected physical ranks 0..3")
        request_ids = sorted({str(row["request_id"]) for _, row, _ in items})
        if any(request_id not in metadata for request_id in request_ids):
            raise RuntimeError(f"missing result metadata for {request_ids}")
        k_visual_values = {int(metadata[request_id]["k_visual"]) for request_id in request_ids}
        if len(k_visual_values) != 1:
            raise RuntimeError("one physical wave mixed multiple visual-k policies")
        k_visual = k_visual_values.pop()
        visual_rows = sum(int(metadata[request_id]["vision_token_count"])
                          for request_id in request_ids)
        logical_rows = sum(int(ids.shape[0]) for _, _, ids in items)
        valid_pairs = sum(int(np.count_nonzero(ids >= 0)) for _, _, ids in items)
        invalid_slots = sum(int(np.count_nonzero(ids < 0)) for _, _, ids in items)
        expected_invalid = visual_rows * (8 - k_visual)
        expected_valid = logical_rows * 8 - expected_invalid
        if invalid_slots != expected_invalid or valid_pairs != expected_valid:
            raise RuntimeError(
                f"mixed-k accounting mismatch wave={wave} layer={layer}: "
                f"valid {valid_pairs}/{expected_valid}, invalid {invalid_slots}/{expected_invalid}"
            )
        ep4 = _replay_row(items, target_ep=4)
        physical_rows.append({
            "config": config,
            "wave": wave,
            "layer": layer,
            "iteration": iteration,
            "request_ids": request_ids,
            "logical_rows": logical_rows,
            "visual_rows": visual_rows,
            "text_special_rows": logical_rows - visual_rows,
            "all_valid_pairs": valid_pairs,
            "valid_visual_pairs": visual_rows * k_visual,
            "invalid_visual_slots": invalid_slots,
            "all_rank_pair_histogram": ep4["rank_assignment_load"],
            "logical_remote_dispatch_bf16_bytes": ep4["logical_dispatch_bytes"],
            "logical_remote_combine_bf16_bytes": ep4["logical_combine_bytes"],
            "assignment_matrix": ep4["assignment_matrix"],
            "unique_activation_matrix": ep4["unique_activation_matrix"],
            "rank_load_stats": ep4["rank_load_stats"],
            "max_fanout": ep4["max_fanout"],
            "evidence": "MEASURED_TRUE_EP4_EXACT_POST_ADAPTER_ROUTE_IDS",
        })
        for target_ep in (4, 8):
            converted = _replay_row(items, target_ep=target_ep)
            replay[target_ep].append({
                "wave": wave,
                "layer": layer,
                "request_ids": request_ids,
                "logical_rows": logical_rows,
                **converted,
            })

    return {"physical_rows": physical_rows, "replay": replay}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.profile_root, args.config, args.iteration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "route_summary.jsonl"
    summary_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n"
                for row in result["physical_rows"]),
        encoding="utf-8",
    )
    for target_ep in (4, 8):
        payload = {
            "schema_version": "mllm_ep_replay.v4",
            "status": "STRUCTURAL_VERIFIED_FROM_POLICY_SPECIFIC_ACTUAL_ROUTES",
            "trace_root": str(args.profile_root.resolve()),
            "target_ep": target_ep,
            "target_topology": "TP2_DP2_EP4_PP1" if target_ep == 4 else "TP2_DP4_EP8_PP1",
            "source_rule": (
                "global batch two uses TP source ranks 0/1 and 2/3; "
                "target EP8 ranks 4..7 remain idle for the same global request count"
            ),
            "owner_rule": f"contiguous {128 // target_ep} experts/rank",
            "logical_activation_bytes_per_row": 4096,
            "transport_warning": None if target_ep == 4 else (
                "EP8 communication and E2E are unvalidated extrapolation"
            ),
            "rows": result["replay"][target_ep],
        }
        atomic_json(args.output_dir / f"virtual_ep{target_ep}.json", payload)
    print(json.dumps({
        "config": args.config,
        "physical_rows": len(result["physical_rows"]),
        "ep4_rows": len(result["replay"][4]),
        "ep8_rows": len(result["replay"][8]),
    }, indent=2))


if __name__ == "__main__":
    main()
