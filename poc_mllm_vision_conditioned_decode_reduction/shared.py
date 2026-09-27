"""Shared, latency-free structural utilities for the dual-track PoC.

The helpers in this module operate only on measured routing records.  They do
not call a simulator and intentionally refuse to manufacture remote-traffic
results when source-rank identity is absent.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

NUM_EXPERTS = 128
NUM_LAYERS = 48
TOP_K = 8
HIDDEN_SIZE = 2048
DTYPE_BYTES = 2
LOGICAL_ACTIVATION_BYTES = HIDDEN_SIZE * DTYPE_BYTES
SUPPORTED_EP = (2, 4, 8)
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
WINDOWS = ((1, 32), (33, 64), (65, 96), (97, 128), (1, 128))
TP2_DUMMY_IDS = tuple(range(TOP_K))
TP2_DUMMY_WEIGHT = 1.0 / TOP_K


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dense_sparse(value: Mapping[str, float]) -> np.ndarray:
    result = np.zeros(NUM_EXPERTS, dtype=np.float64)
    for key, item in value.items():
        expert = int(key)
        if not 0 <= expert < NUM_EXPERTS:
            raise ValueError(f"expert ID outside [0,{NUM_EXPERTS}): {expert}")
        result[expert] = float(item)
    return result


def owner_rank(expert_id: int, ep_size: int) -> int:
    if ep_size not in SUPPORTED_EP:
        raise ValueError(f"unsupported EP size: {ep_size}")
    expert = int(expert_id)
    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError(f"expert ID outside [0,{NUM_EXPERTS}): {expert}")
    return expert // (NUM_EXPERTS // ep_size)


def is_tp2_dummy_route(
    route_ids: Sequence[Sequence[int]], route_weights: Sequence[Sequence[float]]
) -> bool:
    """Recognize the measured TP2 decode-padding row exactly.

    The preserved long-reasoning trace has one real decode row and one padded
    TP shard.  The padded shard is top-8 IDs 0..7 at equal 1/8 weights.  This
    predicate is deliberately strict so a real row is never silently removed.
    """

    if len(route_ids) != 1 or len(route_weights) != 1:
        return False
    ids = tuple(int(value) for value in route_ids[0])
    weights = tuple(float(value) for value in route_weights[0])
    return ids == TP2_DUMMY_IDS and len(weights) == TOP_K and all(
        math.isclose(value, TP2_DUMMY_WEIGHT, rel_tol=0.0, abs_tol=1e-7)
        for value in weights
    )


def compact_decode_histogram(
    row: Mapping[str, Any], *, tp2_dummy_copies: int = 0
) -> np.ndarray:
    """Return physical assignment counts, optionally stripping proven padding.

    ``tp2_dummy_copies`` is explicit because a compact histogram alone cannot
    prove that IDs 0..7 are padding.  Callers may pass one only for the audited
    TP2-sharded long-reasoning decode trace.
    """

    counts = dense_sparse(row.get("expert_count", {}))
    if tp2_dummy_copies:
        if tp2_dummy_copies < 0:
            raise ValueError("tp2_dummy_copies must be non-negative")
        counts[:TOP_K] -= float(tp2_dummy_copies)
        if np.any(counts < 0):
            raise ValueError("compact row does not contain the declared TP2 dummy copy")
    return counts


@dataclass(frozen=True)
class ExpertPath:
    request_id: str
    step: int
    layer: int
    token_id: str
    original_expert_id: int
    executed_expert_id: int | None
    source_rank: int | None
    route_weight: float = 1.0


def _coerce_path(row: Mapping[str, Any], index: int) -> ExpertPath:
    original = int(row.get("original_expert_id", row.get("expert_id")))
    executed_value = row.get("executed_expert_id", original)
    executed = None if executed_value is None else int(executed_value)
    source_value = row.get("source_rank")
    return ExpertPath(
        request_id=str(row.get("request_id", "UNMAPPED")),
        step=int(row.get("step", -1)),
        layer=int(row.get("layer", -1)),
        token_id=str(row.get("token_id", row.get("path_id", index))),
        original_expert_id=original,
        executed_expert_id=executed,
        source_rank=None if source_value is None else int(source_value),
        route_weight=float(row.get("route_weight", 1.0)),
    )


def structural_ep_projection(
    rows: Iterable[Mapping[str, Any]], *, ep_size: int
) -> dict[str, Any]:
    """Project actual policy decisions to an EP owner map without latency.

    Multiple original paths from the same token that consolidate onto one
    target expert are counted once in the residual executed path set.  Remote
    traffic is returned only when every path has an explicit target-topology
    source rank.
    """

    if ep_size not in SUPPORTED_EP:
        raise ValueError(f"unsupported EP size: {ep_size}")
    paths = [_coerce_path(row, index) for index, row in enumerate(rows)]
    for path in paths:
        owner_rank(path.original_expert_id, ep_size)
        if path.executed_expert_id is not None:
            owner_rank(path.executed_expert_id, ep_size)
        if path.source_rank is not None and not 0 <= path.source_rank < ep_size:
            raise ValueError(f"source rank {path.source_rank} invalid for EP{ep_size}")

    events: dict[tuple[str, int, int], list[ExpertPath]] = defaultdict(list)
    for path in paths:
        events[(path.request_id, path.step, path.layer)].append(path)

    source_complete = all(path.source_rank is not None for path in paths)
    baseline_remote = residual_remote = 0
    changed_local = changed_remote = 0
    baseline_active_rank: list[list[int]] = []
    residual_active_rank: list[list[int]] = []
    baseline_rank_loads: list[list[int]] = []
    residual_rank_loads: list[list[int]] = []
    baseline_path_count = 0
    residual_path_count = 0

    for event_paths in events.values():
        baseline_unique = {
            (path.token_id, path.original_expert_id, path.source_rank): path
            for path in event_paths
        }
        residual_unique = {
            (path.token_id, path.executed_expert_id, path.source_rank): path
            for path in event_paths
            if path.executed_expert_id is not None
        }
        baseline_path_count += len(baseline_unique)
        residual_path_count += len(residual_unique)
        base_load = [0] * ep_size
        new_load = [0] * ep_size
        base_active = [set() for _ in range(ep_size)]
        new_active = [set() for _ in range(ep_size)]
        for path in baseline_unique.values():
            rank = owner_rank(path.original_expert_id, ep_size)
            base_load[rank] += 1
            base_active[rank].add(path.original_expert_id)
            if source_complete and rank != path.source_rank:
                baseline_remote += 1
        for path in residual_unique.values():
            assert path.executed_expert_id is not None
            rank = owner_rank(path.executed_expert_id, ep_size)
            new_load[rank] += 1
            new_active[rank].add(path.executed_expert_id)
            if source_complete and rank != path.source_rank:
                residual_remote += 1
        baseline_rank_loads.append(base_load)
        residual_rank_loads.append(new_load)
        baseline_active_rank.append([len(value) for value in base_active])
        residual_active_rank.append([len(value) for value in new_active])

    for path in paths:
        if path.executed_expert_id is None or path.executed_expert_id == path.original_expert_id:
            continue
        target = owner_rank(path.executed_expert_id, ep_size)
        if path.source_rank is None:
            continue
        if target == path.source_rank:
            changed_local += 1
        else:
            changed_remote += 1

    def mean_rank_ratio(loads: list[list[int]]) -> float | None:
        values = []
        for load in loads:
            mean = sum(load) / len(load)
            if mean:
                values.append(max(load) / mean)
        return float(np.mean(values)) if values else None

    def mean_per_rank(values: list[list[int]]) -> list[float]:
        if not values:
            return [0.0] * ep_size
        return np.asarray(values, dtype=np.float64).mean(axis=0).tolist()

    eliminated = baseline_path_count - residual_path_count
    output: dict[str, Any] = {
        "evidence_label": (
            "STRUCTURAL_EP8_PROJECTION_ONLY" if ep_size == 8 else "STRUCTURAL_REPLAY_ONLY"
        ),
        "ep_size": ep_size,
        "experts_per_rank": NUM_EXPERTS // ep_size,
        "events": len(events),
        "baseline_expert_paths": baseline_path_count,
        "residual_expert_paths": residual_path_count,
        "eliminated_expert_paths": eliminated,
        "eliminated_expert_path_fraction": eliminated / max(baseline_path_count, 1),
        "baseline_mean_active_experts_per_rank": mean_per_rank(baseline_active_rank),
        "residual_mean_active_experts_per_rank": mean_per_rank(residual_active_rank),
        "baseline_mean_assignments_per_rank": mean_per_rank(baseline_rank_loads),
        "residual_mean_assignments_per_rank": mean_per_rank(residual_rank_loads),
        "baseline_mean_rank_max_mean": mean_rank_ratio(baseline_rank_loads),
        "residual_mean_rank_max_mean": mean_rank_ratio(residual_rank_loads),
        "source_rank_status": "AVAILABLE" if source_complete else "UNAVAILABLE",
        "changed_target_local_paths": changed_local if source_complete else None,
        "changed_target_remote_paths": changed_remote if source_complete else None,
        "latency_ms": None,
        "tpot_gain": None,
        "e2e_gain": None,
    }
    if source_complete:
        output.update(
            {
                "baseline_remote_assignments": baseline_remote,
                "residual_remote_assignments": residual_remote,
                "avoided_remote_assignments": baseline_remote - residual_remote,
                "baseline_logical_dispatch_bytes": baseline_remote
                * LOGICAL_ACTIVATION_BYTES,
                "baseline_logical_combine_bytes": baseline_remote
                * LOGICAL_ACTIVATION_BYTES,
                "residual_logical_dispatch_bytes": residual_remote
                * LOGICAL_ACTIVATION_BYTES,
                "residual_logical_combine_bytes": residual_remote
                * LOGICAL_ACTIVATION_BYTES,
            }
        )
    else:
        output.update(
            {
                "baseline_remote_assignments": None,
                "residual_remote_assignments": None,
                "avoided_remote_assignments": None,
                "baseline_logical_dispatch_bytes": None,
                "baseline_logical_combine_bytes": None,
                "residual_logical_dispatch_bytes": None,
                "residual_logical_combine_bytes": None,
            }
        )
    return output


def batch_regime_sweep(
    rows: Sequence[Mapping[str, Any]],
    pair_ids: Sequence[str],
    *,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    tp2_dummy_copies: int = 1,
    expert_weight_bytes: int | None = None,
) -> dict[str, Any]:
    """Measure baseline expert working-set shape for deterministic frozen waves."""

    pair_ids = sorted(set(map(str, pair_ids)))
    decoded: dict[tuple[str, int, int], np.ndarray] = {}
    for row in rows:
        if row.get("record_type") != "decode" or row.get("pair_id") not in pair_ids:
            continue
        decoded[(str(row["pair_id"]), int(row["layer"]), int(row["step"]))] = (
            compact_decode_histogram(row, tp2_dummy_copies=tp2_dummy_copies)
        )
    common_steps = set.intersection(
        *[
            {step for request, _layer, step in decoded if request == pair}
            for pair in pair_ids
        ]
    )
    results: dict[str, Any] = {}
    for batch in batch_sizes:
        if batch > len(pair_ids):
            results[str(batch)] = {
                "status": "INSUFFICIENT_REQUEST_POOL",
                "pool_requests": len(pair_ids),
                "requested_batch": batch,
            }
            continue
        full = len(pair_ids) // batch
        used = full * batch
        waves = [pair_ids[offset : offset + batch] for offset in range(0, used, batch)]
        unions: list[int] = []
        paths_per_expert: list[float] = []
        concentration: list[float] = []
        active_per_rank: list[list[int]] = []
        for wave in waves:
            for step in sorted(common_steps):
                for layer in range(NUM_LAYERS):
                    count = sum((decoded[(pair, layer, step)] for pair in wave), np.zeros(NUM_EXPERTS))
                    active = int(np.count_nonzero(count))
                    total = float(count.sum())
                    unions.append(active)
                    paths_per_expert.append(total / max(active, 1))
                    probability = count / total if total else count
                    concentration.append(float(np.square(probability).sum()))
                    active_per_rank.append(
                        [
                            int(np.count_nonzero(chunk))
                            for chunk in np.split(count, 4)
                        ]
                    )
        mean_union = float(np.mean(unions))
        results[str(batch)] = {
            "status": "COMPLETE_FROZEN_WAVES",
            "waves": len(waves),
            "requests_used": used,
            "requests_omitted": pair_ids[used:],
            "sampled_decode_steps": sorted(common_steps),
            "events": len(unions),
            "mean_active_expert_union": mean_union,
            "p50_active_expert_union": float(np.median(unions)),
            "p90_active_expert_union": float(np.quantile(unions, 0.9)),
            "mean_expert_paths_per_active_expert": float(np.mean(paths_per_expert)),
            "mean_expert_frequency_hhi": float(np.mean(concentration)),
            "mean_active_experts_per_ep4_rank": np.asarray(active_per_rank).mean(axis=0).tolist(),
            "expert_weight_working_set_proxy_bytes": (
                None if expert_weight_bytes is None else mean_union * expert_weight_bytes
            ),
            "working_set_proxy_label": (
                "UNAVAILABLE"
                if expert_weight_bytes is None
                else "DISTINCT_EXPERT_WEIGHT_BYTES_NOT_HBM_TRAFFIC"
            ),
            "timing_status": "NOT_MEASURED_STRUCTURAL_ONLY",
        }
    return {
        "schema_version": 1,
        "evidence_label": "MEASURED_TRACE_STRUCTURAL_REPLAY_NO_LATENCY",
        "pool_requests": len(pair_ids),
        "tp2_dummy_copies_removed_per_request_layer_step": tp2_dummy_copies,
        "results": results,
    }


def _scale_layer_component(values: np.ndarray) -> np.ndarray:
    maximum = values.max(axis=1, keepdims=True)
    return np.divide(values, maximum, out=np.zeros_like(values), where=maximum > 0)


def build_discovery_vision_criticality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze a discovery-only layer/expert criticality score from image swaps."""

    prefill: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    decode: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("analysis_split") != "discovery" or row.get("repeat_id") is not None:
            continue
        condition = row.get("condition")
        if condition not in ("A", "B"):
            continue
        pair = str(row["pair_id"])
        layer = int(row["layer"])
        if row.get("record_type") == "prefill":
            prefill[(pair, condition, layer)] = row
        else:
            decode[(pair, condition, layer, int(row["step"]))] = row
    pairs = sorted(
        set(pair for pair, condition, _layer in prefill if condition == "A")
        & set(pair for pair, condition, _layer in prefill if condition == "B")
    )
    assignment_num = np.zeros((NUM_LAYERS, NUM_EXPERTS))
    assignment_den = np.zeros_like(assignment_num)
    mass_num = np.zeros_like(assignment_num)
    mass_den = np.zeros_like(assignment_num)
    vision_affinity = np.zeros_like(assignment_num)
    persistence_hits = np.zeros_like(assignment_num)
    persistence_total = np.zeros((NUM_LAYERS, 1))
    matched_events = 0
    for pair in pairs:
        for layer in range(NUM_LAYERS):
            vision_affinity[layer] += dense_sparse(prefill[(pair, "A", layer)]["vision_mass"])
            steps = sorted(
                set(step for req, cond, lyr, step in decode if req == pair and cond == "A" and lyr == layer)
                & set(step for req, cond, lyr, step in decode if req == pair and cond == "B" and lyr == layer)
            )
            for step in steps:
                left = compact_decode_histogram(
                    decode[(pair, "A", layer, step)], tp2_dummy_copies=1
                )
                right = compact_decode_histogram(
                    decode[(pair, "B", layer, step)], tp2_dummy_copies=1
                )
                left_mass = dense_sparse(decode[(pair, "A", layer, step)]["expert_mass"])
                right_mass = dense_sparse(decode[(pair, "B", layer, step)]["expert_mass"])
                # Compact mass includes one measured dummy row on both sides;
                # it cancels in the numerator and is removed from denominators.
                left_mass[:TOP_K] -= TP2_DUMMY_WEIGHT
                right_mass[:TOP_K] -= TP2_DUMMY_WEIGHT
                left_mass = np.maximum(left_mass, 0.0)
                right_mass = np.maximum(right_mass, 0.0)
                assignment_num[layer] += np.abs(left - right)
                assignment_den[layer] += left + right
                mass_num[layer] += np.abs(left_mass - right_mass)
                mass_den[layer] += left_mass + right_mass
                persistence_hits[layer] += left > 0
                persistence_total[layer, 0] += 1
                matched_events += 1
    assignment_change = np.divide(
        assignment_num, assignment_den, out=np.zeros_like(assignment_num), where=assignment_den > 0
    )
    mass_change = np.divide(mass_num, mass_den, out=np.zeros_like(mass_num), where=mass_den > 0)
    persistence = np.divide(
        persistence_hits,
        persistence_total,
        out=np.zeros_like(persistence_hits),
        where=persistence_total > 0,
    )
    components = {
        "assignment_change": _scale_layer_component(assignment_change),
        "routing_mass_change": _scale_layer_component(mass_change),
        "vision_prefill_affinity": _scale_layer_component(vision_affinity),
        "decode_persistence": _scale_layer_component(persistence),
    }
    score = (
        0.35 * components["assignment_change"]
        + 0.35 * components["routing_mass_change"]
        + 0.20 * components["vision_prefill_affinity"]
        + 0.10 * components["decode_persistence"]
    )
    records = []
    for layer in range(NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            records.append(
                {
                    "layer": layer,
                    "expert_id": expert,
                    "criticality": float(score[layer, expert]),
                    **{name: float(value[layer, expert]) for name, value in components.items()},
                }
            )
    return {
        "schema_version": 1,
        "status": "FROZEN_DISCOVERY_ONLY",
        "pair_count": len(pairs),
        "matched_layer_step_events": matched_events,
        "formula": "0.35 assignment_change + 0.35 routing_mass_change + 0.20 vision_prefill_affinity + 0.10 decode_persistence; each component max-normalized within layer",
        "heldout_used": False,
        "records": records,
    }
