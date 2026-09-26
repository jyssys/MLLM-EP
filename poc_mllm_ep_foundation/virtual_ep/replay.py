from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..constants import DTYPE_BYTES, HIDDEN_SIZE, NUM_EXPERTS
from ..io import atomic_json
from .mapper import expert_owner_map, source_rank_for_target, validate_topk

ROUTE_RE = re.compile(r"wave(?P<wave>\d+)_dp(?P<dp>\d+)_tp(?P<tp>\d+)_layer(?P<layer>\d+)\.npy$")


@dataclass(frozen=True)
class ReplayRow:
    wave: int
    request_id: str
    layer: int
    target_ep: int
    logical_rows: int
    assignments: int
    active_experts: int
    expert_histogram: list[int]
    rank_histogram: list[int]
    source_destination_unique_rows: list[list[int]]
    remote_unique_rows: int
    dispatch_logical_bytes: int
    combine_logical_bytes: int
    remote_fanout_by_source: list[int]
    max_rank_assignments: int
    mean_rank_assignments: float
    source_mapping: str
    evidence: str


def _schedule(trace_root: Path) -> dict[int, dict]:
    path = trace_root / "schedule.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["wave"]): row for row in rows}


def _route_files(trace_root: Path) -> list[Path]:
    root = trace_root / "raw" / "routes"
    return sorted(path for path in root.glob("*.npy") if ROUTE_RE.match(path.name))


def replay_trace(trace_root: Path, target_ep: int, waves: Iterable[int] | None = None,
                 hidden_size: int = HIDDEN_SIZE) -> list[ReplayRow]:
    if target_ep not in (2, 4, 8):
        raise ValueError("target_ep must be 2, 4 or 8")
    schedule = _schedule(trace_root)
    wanted = None if waves is None else set(int(w) for w in waves)
    grouped: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for path in _route_files(trace_root):
        match = ROUTE_RE.match(path.name)
        assert match
        wave, dp, tp, layer = (int(match.group(name)) for name in ("wave", "dp", "tp", "layer"))
        if wanted is not None and wave not in wanted:
            continue
        ids = np.load(path)
        validate_topk(ids)
        grouped.setdefault((wave, layer), []).append(
            (source_rank_for_target(dp, tp, target_ep, wave), ids)
        )
    owners = expert_owner_map(NUM_EXPERTS, target_ep)
    rows: list[ReplayRow] = []
    for (wave, layer), shards in sorted(grouped.items()):
        if wave not in schedule:
            continue
        expert_hist = np.zeros(NUM_EXPERTS, dtype=np.int64)
        rank_hist = np.zeros(target_ep, dtype=np.int64)
        matrix = np.zeros((target_ep, target_ep), dtype=np.int64)
        fanout = np.zeros(target_ep, dtype=np.int64)
        logical_rows = 0
        combine_remote_assignments = 0
        for src, expert_ids in shards:
            logical_rows += int(expert_ids.shape[0])
            flat = expert_ids.reshape(-1)
            expert_hist += np.bincount(flat, minlength=NUM_EXPERTS)
            dest = owners[expert_ids]
            rank_hist += np.bincount(dest.reshape(-1), minlength=target_ep)
            destinations: set[int] = set()
            for row_dest in dest:
                for dst in np.unique(row_dest):
                    matrix[src, int(dst)] += 1
                    destinations.add(int(dst))
                combine_remote_assignments += int(np.count_nonzero(row_dest != src))
            fanout[src] = len(destinations - {src})
        remote_unique = int(matrix.sum() - np.trace(matrix))
        rows.append(ReplayRow(
            wave=wave,
            request_id=str(schedule[wave]["request_id"]),
            layer=layer,
            target_ep=target_ep,
            logical_rows=logical_rows,
            assignments=int(expert_hist.sum()),
            active_experts=int(np.count_nonzero(expert_hist)),
            expert_histogram=expert_hist.astype(int).tolist(),
            rank_histogram=rank_hist.astype(int).tolist(),
            source_destination_unique_rows=matrix.astype(int).tolist(),
            remote_unique_rows=remote_unique,
            dispatch_logical_bytes=remote_unique * hidden_size * DTYPE_BYTES,
            combine_logical_bytes=combine_remote_assignments * hidden_size * DTYPE_BYTES,
            remote_fanout_by_source=fanout.astype(int).tolist(),
            max_rank_assignments=int(rank_hist.max(initial=0)),
            mean_rank_assignments=float(rank_hist.mean()),
            source_mapping=(
                "TP2 source copies; EP2 maps each request to DP0, EP4 preserves measured DP0/1, "
                "EP8 assigns the unchanged arrival stream round-robin over DP0..3"
            ),
            evidence=("MEASURED_ROUTE_REPLAY" if target_ep == 4 else
                      "VIRTUAL_EP2_STRUCTURAL_REPLAY" if target_ep == 2 else
                      "UNVALIDATED_EP8_EXTRAPOLATION"),
        ))
    if not rows:
        raise RuntimeError(f"no valid route rows found under {trace_root}")
    return rows


def write_replay(output: Path, trace_root: Path, target_ep: int,
                 waves: Iterable[int] | None = None) -> dict:
    rows = replay_trace(trace_root, target_ep, waves)
    payload = {
        "schema_version": "virtual_ep_replay.v1",
        "trace_root": str(trace_root.resolve()),
        "target_ep": target_ep,
        "physical_gpu_count_required": 0,
        "latency_claim": "NONE_WORKLOAD_ONLY",
        "evidence": ("MEASURED_EP4_TOPOLOGY_REPLAY" if target_ep == 4 else
                     "VIRTUAL_EP2_STRUCTURAL_REPLAY" if target_ep == 2 else
                     "EP8_EXTRAPOLATION"),
        "rows": [asdict(row) for row in rows],
    }
    atomic_json(output, payload)
    return payload
