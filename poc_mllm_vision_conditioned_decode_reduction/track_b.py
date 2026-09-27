"""Bounded Track-B temporal-reuse analysis.

The existing long-reasoning trace contains exact route IDs/weights at selected
decode checkpoints, but no expert input/output vectors.  This module therefore
keeps two evidence levels separate:

* route recurrence is measured from the raw (not compact) TRUE-EP4 trace;
* the low-error reuse oracle is evaluated only from an explicit vector-capture
  artifact.  Missing vectors produce ``CAPTURE_REQUIRED`` rather than an
  invented oracle value.

No function in this module mutates routing or claims latency.
"""
from __future__ import annotations

import json
import math
import re
import gzip
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

NUM_EXPERTS = 128
TOP_K = 8
LAYERS = 48
RECENCY_WINDOWS = (1, 2, 4, 8, 16)
ERROR_THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00)
REUSE_BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30)
HIDDEN_SIZE = 2048
BF16_BYTES = 2
LOGICAL_DISPATCH_COMBINE_BYTES_PER_PATH = 2 * HIDDEN_SIZE * BF16_BYTES


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def step_bucket(step: int) -> str:
    if step <= 32:
        return "1-32"
    if step <= 64:
        return "33-64"
    if step <= 96:
        return "65-96"
    return "97-128"


def layer_bucket(layer: int) -> str:
    if layer < 16:
        return "early"
    if layer < 32:
        return "middle"
    return "late"


@dataclass(frozen=True)
class RoutePoint:
    pair_id: str
    split: str
    step: int
    layer: int
    ids: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class RecurrenceEvent:
    pair_id: str
    split: str
    step: int
    step_bucket: str
    layer: int
    layer_bucket: str
    expert_id: int
    router_weight: float
    window: int
    recurred: bool
    distance: int | None


@dataclass(frozen=True)
class CapturedPath:
    expert_id: int
    weight: float
    output: np.ndarray


@dataclass(frozen=True)
class CapturedToken:
    pair_id: str
    split: str
    step: int
    layer: int
    fingerprint: str
    hidden: np.ndarray
    paths: tuple[CapturedPath, ...]


@dataclass(frozen=True)
class VectorOracleEvent:
    pair_id: str
    split: str
    step: int
    layer: int
    expert_id: int
    window: int
    recurred: bool
    distance: int | None
    hidden_cosine: float | None
    output_cosine: float | None
    output_relative_l2: float | None
    weighted_relative_l2: float | None
    current_reweighted_relative_l2: float | None


def load_pool(path: Path) -> dict[str, str]:
    """Load an image-disjoint frozen pool without consulting route outcomes."""
    value = read_json(path)
    output: dict[str, str] = {}
    for anchor in value["anchors"]:
        pair = str(anchor["pair_id"])
        split = str(
            anchor.get("reduction_split")
            or anchor.get("coscheduling_split")
            or anchor["analysis_split"]
        )
        if split not in {"discovery", "heldout"}:
            raise ValueError((pair, split))
        if pair in output:
            raise ValueError(f"duplicate pair in frozen pool: {pair}")
        output[pair] = split
    return output


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_active_tp0_routes(
    trace_dirs: Sequence[Path], pool: dict[str, str]
) -> list[RoutePoint]:
    """Read condition-A decode routes, removing idle-DP and TP-copy inflation.

    The earlier compact artifact sums both TP rows.  In the historical trace,
    the idle TP1 row is the deterministic dummy route ``0..7``.  Selecting the
    source request's TP0 record preserves one real logical decode row.  This is
    verified rather than inferred by filtering on route contents.
    """
    indexed: dict[tuple[str, int, int], RoutePoint] = {}
    for root in trace_dirs:
        for path in sorted(root.glob("route_summary.pid*.jsonl")):
            for row in _iter_jsonl(path):
                pair = str(row.get("pair_id") or "")
                if pair not in pool or row.get("condition") != "A":
                    continue
                step = row.get("decode_step")
                if step is None or not 1 <= int(step) <= 128:
                    continue
                if int(row.get("tp_rank", -1)) != 0:
                    continue
                ids_matrix = row.get("route_ids") or []
                weights_matrix = row.get("route_weights") or []
                if len(ids_matrix) != 1 or len(weights_matrix) != 1:
                    raise ValueError(f"expected one decode row: {path} {pair} step={step}")
                ids = tuple(int(value) for value in ids_matrix[0])
                weights = tuple(float(value) for value in weights_matrix[0])
                layer = int(row["layer"])
                if len(ids) != TOP_K or len(weights) != TOP_K:
                    raise ValueError((pair, step, layer, len(ids), len(weights)))
                if len(set(ids)) != TOP_K or min(ids) < 0 or max(ids) >= NUM_EXPERTS:
                    raise ValueError((pair, step, layer, ids))
                point = RoutePoint(pair, pool[pair], int(step), layer, ids, weights)
                key = (pair, layer, int(step))
                prior = indexed.get(key)
                if prior is not None and prior != point:
                    raise ValueError(f"non-identical duplicate route: {key}")
                indexed[key] = point
    points = sorted(indexed.values(), key=lambda row: (row.pair_id, row.layer, row.step))
    if not points:
        raise RuntimeError("no eligible condition-A TP0 decode routes found")
    for point in points:
        if not 0 <= point.layer < LAYERS:
            raise ValueError(point)
    return points


def load_compact_routes(path: Path, pool: dict[str, str]) -> list[RoutePoint]:
    """Load the committed exact condition-A/TP0 route export."""
    opener = gzip.open if path.suffix == ".gz" else open
    indexed: dict[tuple[str, int, int], RoutePoint] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = str(row["pair_id"])
            if pair not in pool:
                continue
            split = str(row.get("split") or pool[pair])
            if split != pool[pair]:
                raise ValueError(f"compact split mismatch for {pair}: {split} != {pool[pair]}")
            ids = tuple(int(value) for value in row["route_ids"])
            weights = tuple(float(value) for value in row["route_weights"])
            step, layer = int(row["step"]), int(row["layer"])
            if len(ids) != TOP_K or len(weights) != TOP_K or len(set(ids)) != TOP_K:
                raise ValueError((pair, step, layer, ids, weights))
            if min(ids) < 0 or max(ids) >= NUM_EXPERTS or not 1 <= step <= 128 or not 0 <= layer < LAYERS:
                raise ValueError((pair, step, layer, ids))
            point = RoutePoint(pair, split, step, layer, ids, weights)
            key = (pair, layer, step)
            prior = indexed.get(key)
            if prior is not None and prior != point:
                raise ValueError(f"conflicting compact route duplicate: {key}")
            indexed[key] = point
    if not indexed:
        raise RuntimeError(f"no eligible routes in compact artifact: {path}")
    return sorted(indexed.values(), key=lambda row: (row.pair_id, row.layer, row.step))


def recurrence_events(points: Sequence[RoutePoint]) -> list[RecurrenceEvent]:
    grouped: dict[tuple[str, int], list[RoutePoint]] = defaultdict(list)
    for point in points:
        grouped[(point.pair_id, point.layer)].append(point)
    output: list[RecurrenceEvent] = []
    for (_pair, _layer), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row.step)
        for window in RECENCY_WINDOWS:
            for index, current in enumerate(rows):
                available = [prior for prior in rows[:index] if current.step - prior.step <= window]
                if not available:
                    continue
                for expert, weight in zip(current.ids, current.weights):
                    matching = [prior for prior in available if expert in prior.ids]
                    distance = (
                        current.step - max(matching, key=lambda row: row.step).step
                        if matching else None
                    )
                    output.append(RecurrenceEvent(
                        current.pair_id, current.split, current.step, step_bucket(current.step),
                        current.layer, layer_bucket(current.layer), expert, weight, window,
                        bool(matching), distance,
                    ))
    return output


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _summarize_events(rows: Sequence[RecurrenceEvent]) -> dict[str, Any]:
    eligible = len(rows)
    hits = [row for row in rows if row.recurred]
    distances = [float(row.distance) for row in hits if row.distance is not None]
    result: dict[str, Any] = {
        "eligible_checkpoint_paths": eligible,
        "recurrent_paths": len(hits),
        "recurrence_rate": len(hits) / eligible if eligible else None,
        "distance_p50": _percentile(distances, 50),
        "distance_p90": _percentile(distances, 90),
    }
    for ep in (2, 4, 8):
        per_rank = NUM_EXPERTS // ep
        remote_hits = sum(row.expert_id // per_rank != 0 for row in hits)
        result[f"ep{ep}_reused_remote_paths"] = remote_hits
        result[f"ep{ep}_avoided_logical_dispatch_combine_bytes"] = (
            remote_hits * LOGICAL_DISPATCH_COMBINE_BYTES_PER_PATH
        )
    return result


def summarize_recurrence(
    events: Sequence[RecurrenceEvent], points: Sequence[RoutePoint], *, evidence: str
) -> dict[str, Any]:
    all_steps = sorted({point.step for point in points})
    result: dict[str, Any] = {
        "evidence": evidence,
        "boundary": (
            "Sparse selected checkpoints only. Rates are opportunity among paths whose current "
            "checkpoint has at least one recorded prior checkpoint inside W; unrecorded intermediate "
            "steps are not imputed. This is not an output-reuse safety oracle."
        ),
        "selected_steps": all_steps,
        "request_count": len({point.pair_id for point in points}),
        "point_count": len(points),
        "route_path_count": len(points) * TOP_K,
        "by_split_window": {},
        "by_split_step_bucket_window": {},
        "by_split_layer_bucket_window": {},
    }
    for split in ("discovery", "heldout", "all"):
        split_rows = [row for row in events if split == "all" or row.split == split]
        for window in RECENCY_WINDOWS:
            rows = [row for row in split_rows if row.window == window]
            result["by_split_window"][f"{split}:W{window}"] = _summarize_events(rows)
            for bucket in ("1-32", "33-64", "65-96", "97-128"):
                local = [row for row in rows if row.step_bucket == bucket]
                result["by_split_step_bucket_window"][f"{split}:{bucket}:W{window}"] = _summarize_events(local)
            for bucket in ("early", "middle", "late"):
                local = [row for row in rows if row.layer_bucket == bucket]
                result["by_split_layer_bucket_window"][f"{split}:{bucket}:W{window}"] = _summarize_events(local)
    return result


def route_recurrence_ep8_projection(events: Sequence[RecurrenceEvent], *, window: int) -> dict[str, Any]:
    """An unsafe maximum based on route recurrence, never a chosen reuse mask."""
    rows = [row for row in events if row.window == window]
    hits = [row for row in rows if row.recurred]
    residual = [0] * 8
    baseline = [0] * 8
    for row in rows:
        rank = row.expert_id // 16
        baseline[rank] += 1
        if not row.recurred:
            residual[rank] += 1
    avoided_remote = sum(row.expert_id // 16 != 0 for row in hits)
    return {
        "status": "ROUTE_RECURRENCE_MAXIMUM_NOT_ERROR_GATED",
        "evidence_label": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "window": window,
        "eligible_checkpoint_paths": len(rows),
        "recurrent_paths": len(hits),
        "maximum_path_elimination_fraction": len(hits) / len(rows) if rows else None,
        "baseline_assignments_per_rank": baseline,
        "residual_assignments_per_rank_if_every_recurrence_were_reused": residual,
        "avoided_remote_assignments": avoided_remote,
        "avoided_logical_dispatch_combine_bytes": avoided_remote * LOGICAL_DISPATCH_COMBINE_BYTES_PER_PATH,
        "latency": "FORBIDDEN_NOT_COMPUTED",
        "warning": (
            "This recurrence-only maximum ignores measured output error and is not a safe or deployable policy."
        ),
    }


def attach_safe_ep8_effect(unsafe: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    """Separate the frozen safe effect from route-recurrence upper bound."""
    gate = oracle.get("gate", {})
    measured = oracle.get("status") in {
        "MEASURED_DISCOVERY_ORACLE", "MEASURED_BOUNDED_VECTOR_ORACLE"
    }
    zero_safe_mask = measured and gate.get("applicable") and gate.get("best_reuse_rate") == 0.0
    if zero_safe_mask:
        safe = {
            "status": "FROZEN_DISCOVERY_GATE_MASK",
            "weighted_relative_l2_threshold": gate["registered_very_low_error"],
            "eliminated_expert_paths": 0,
            "eliminated_path_fraction": 0.0,
            "avoided_remote_assignments": 0,
            "avoided_logical_dispatch_combine_bytes": 0,
            "rank_load_change": [0] * 8,
            "interpretation": "SAFE_EFFECT_0",
        }
    else:
        safe = {
            "status": "NOT_AVAILABLE_NO_FROZEN_SAFE_REUSE_MASK",
            "eliminated_expert_paths": None,
            "avoided_remote_assignments": None,
            "avoided_logical_dispatch_combine_bytes": None,
        }
    return {
        "evidence_label": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "safe_frozen_effect": safe,
        "unsafe_route_recurrence_upper_bound": unsafe,
        "latency": "FORBIDDEN_NOT_COMPUTED",
    }


_ROUTER_NAME = re.compile(
    r"^router\.(?P<request>.+)\.rank\d+\.layer(?P<layer>\d+)\."
    r"step(?P<step>\d+)\.call\d+\.npz$"
)
_EXPERT_NAME = re.compile(
    r"^experts\.(?P<request>.+)\.rank\d+\.layer(?P<layer>\d+)\."
    r"step(?P<step>\d+)\.call\d+\.npz$"
)


def _capture_root(path: Path) -> tuple[Path, Path | None]:
    if (path / "raw").is_dir():
        return path / "raw", path / "manifest.json"
    manifest = path.parent / "manifest.json"
    return path, manifest if manifest.exists() else None


def load_shared_npz_capture(
    capture_dir: Path, pool: dict[str, str]
) -> tuple[list[CapturedToken], dict[str, Any]]:
    """Recombine source-router rows with receiver EP shards.

    External-DP DeepEP emits receiver padding/dummy rows.  A receiver row is
    accepted only when its fingerprint occurs in the source-side router shard
    for the same request/layer/step.  Receiver-only fingerprints are counted
    and discarded.
    """
    raw, manifest_path = _capture_root(capture_dir)
    manifest_splits: dict[str, str] = {}
    if manifest_path is not None and manifest_path.exists():
        manifest = read_json(manifest_path)
        for request in manifest.get("requests", []):
            pair = str(request.get("pair_id") or request.get("capture_id") or "")
            if pair:
                manifest_splits[pair] = str(request.get("reduction_split") or request.get("analysis_split") or pool.get(pair, "discovery"))

    router: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    source_by_fingerprint: dict[tuple[str, int, str], list[tuple[str, int, int, str]]] = defaultdict(list)
    malformed_router = duplicate_router = discarded_non_decode_router_rows = 0
    router_shards = sorted(raw.glob("router.*.npz"))
    for shard in router_shards:
        match = _ROUTER_NAME.match(shard.name)
        if match is None:
            malformed_router += 1
            continue
        request = match.group("request")
        layer, step = int(match.group("layer")), int(match.group("step"))
        with np.load(shard) as payload:
            required = {"fingerprints", "hidden_states", "topk_ids", "topk_weights"}
            if not required.issubset(payload.files):
                malformed_router += 1
                continue
            # A real autoregressive decode invocation has one logical source
            # row.  Multi-chunk prefill can inherit a decode-looking filename
            # step from the worker call counter, but has O(100) rows.  Never
            # admit it to the temporal cache population.
            if int(payload["fingerprints"].shape[0]) != 1:
                discarded_non_decode_router_rows += int(payload["fingerprints"].shape[0])
                continue
            for row, value in enumerate(payload["fingerprints"]):
                fingerprint = str(value)
                ids = payload["topk_ids"][row].astype(np.int64)
                weights = payload["topk_weights"][row].astype(np.float32)
                hidden = payload["hidden_states"][row].astype(np.float32)
                if ids.shape != (TOP_K,) or weights.shape != (TOP_K,) or hidden.shape != (HIDDEN_SIZE,):
                    malformed_router += 1
                    continue
                key = (request, layer, step, fingerprint)
                value_dict = {"ids": ids, "weights": weights, "hidden": hidden}
                if key in router:
                    prior = router[key]
                    if not (
                        np.array_equal(prior["ids"], ids)
                        and np.allclose(prior["weights"], weights, atol=1e-6)
                        and np.array_equal(prior["hidden"], hidden)
                    ):
                        raise ValueError(f"conflicting router duplicate: {key}")
                    duplicate_router += 1
                else:
                    router[key] = value_dict
                    source_by_fingerprint[(request, layer, fingerprint)].append(key)

    receiver_paths: dict[tuple[str, int, int, str], dict[int, CapturedPath]] = defaultdict(dict)
    malformed_expert = filtered_non_source_rows = route_mismatch = weight_mismatch = 0
    receiver_filename_step_mismatch_rows = 0
    expert_shards = sorted(raw.glob("experts.*.npz"))
    for shard in expert_shards:
        match = _EXPERT_NAME.match(shard.name)
        if match is None:
            malformed_expert += 1
            continue
        request = match.group("request")
        layer, step = int(match.group("layer")), int(match.group("step"))
        with np.load(shard) as payload:
            required = {"fingerprints", "expert_ids", "router_weights", "local_mask", "raw_outputs"}
            if not required.issubset(payload.files):
                malformed_expert += 1
                continue
            for row, value in enumerate(payload["fingerprints"]):
                fingerprint = str(value)
                # DeepEP receiver/idle-DP call counters can be offset from the
                # active source.  Fingerprint is the cross-rank row identity;
                # the canonical step always comes from the one-row router
                # shard, never the expert filename.
                keys = source_by_fingerprint.get((request, layer, fingerprint), [])
                if not keys:
                    filtered_non_source_rows += 1
                    continue
                if all(key[2] != step for key in keys):
                    receiver_filename_step_mismatch_rows += 1
                ids = payload["expert_ids"][row]
                weights = payload["router_weights"][row]
                mask = payload["local_mask"][row]
                outputs = payload["raw_outputs"][row]
                if ids.shape != (TOP_K,) or outputs.shape != (TOP_K, HIDDEN_SIZE):
                    malformed_expert += 1
                    continue
                for key in keys:
                    source = router[key]
                    for slot in np.flatnonzero(mask):
                        expert = int(ids[slot])
                        expected = int(source["ids"][slot])
                        if expert != expected:
                            route_mismatch += 1
                            continue
                        weight = float(weights[slot])
                        if not np.isclose(weight, float(source["weights"][slot]), atol=1e-5):
                            weight_mismatch += 1
                            continue
                        path = CapturedPath(expert, weight, outputs[slot].astype(np.float32))
                        prior = receiver_paths[key].get(expert)
                        if prior is not None and not (
                            np.isclose(prior.weight, path.weight, atol=1e-5)
                            and np.array_equal(prior.output, path.output)
                        ):
                            raise ValueError(f"conflicting receiver duplicate: {key} expert={expert}")
                        receiver_paths[key][expert] = path

    tokens: list[CapturedToken] = []
    incomplete = 0
    for key, source in sorted(router.items()):
        paths = receiver_paths.get(key, {})
        expected = [int(value) for value in source["ids"]]
        if len(paths) != TOP_K or set(paths) != set(expected):
            incomplete += 1
            continue
        request, layer, step, fingerprint = key
        split = manifest_splits.get(request) or pool.get(request) or "discovery"
        tokens.append(CapturedToken(
            pair_id=request,
            split=split,
            step=step,
            layer=layer,
            fingerprint=fingerprint,
            hidden=source["hidden"],
            paths=tuple(paths[expert] for expert in expected),
        ))
    integrity = {
        "capture_schema": "shared_decode_vector_capture_npz_v1",
        "router_shards": len(router_shards),
        "expert_shards": len(expert_shards),
        "source_fingerprints": len(router),
        "complete_top8_tokens": len(tokens),
        "incomplete_source_tokens": incomplete,
        "filtered_non_source_receiver_rows": filtered_non_source_rows,
        "discarded_non_decode_router_rows": discarded_non_decode_router_rows,
        "receiver_filename_step_mismatch_rows": receiver_filename_step_mismatch_rows,
        "malformed_router_shards_or_rows": malformed_router,
        "malformed_expert_shards_or_rows": malformed_expert,
        "route_mismatches": route_mismatch,
        "weight_mismatches": weight_mismatch,
        "duplicate_router_rows": duplicate_router,
        "request_count": len({token.pair_id for token in tokens}),
        "layers": sorted({token.layer for token in tokens}),
        "steps": sorted({token.step for token in tokens}),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
    }
    return tokens, integrity


def _cosine_vector(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right)) / max(float(np.linalg.norm(left)), 1e-12)


def vector_oracle_events(tokens: Sequence[CapturedToken]) -> list[VectorOracleEvent]:
    """Build a future-truth oracle selecting the lowest-error prior output in W."""
    grouped: dict[tuple[str, int], list[CapturedToken]] = defaultdict(list)
    for token in tokens:
        grouped[(token.pair_id, token.layer)].append(token)
    output: list[VectorOracleEvent] = []
    for (_request, _layer), local in sorted(grouped.items()):
        local = sorted(local, key=lambda item: item.step)
        for window in RECENCY_WINDOWS:
            for index, current in enumerate(local):
                prior_tokens = [item for item in local[:index] if 0 < current.step - item.step <= window]
                if not prior_tokens:
                    continue
                for current_path in current.paths:
                    candidates: list[tuple[float, CapturedToken, CapturedPath, dict[str, float]]] = []
                    for prior in prior_tokens:
                        match = next((path for path in prior.paths if path.expert_id == current_path.expert_id), None)
                        if match is None:
                            continue
                        raw_error = _relative_l2(current_path.output, match.output)
                        current_weighted = current_path.weight * current_path.output
                        cached_weighted = match.weight * match.output
                        weighted_error = _relative_l2(current_weighted, cached_weighted)
                        metrics = {
                            "hidden_cosine": _cosine_vector(current.hidden, prior.hidden),
                            "output_cosine": _cosine_vector(current_path.output, match.output),
                            "output_relative_l2": raw_error,
                            "weighted_relative_l2": weighted_error,
                            "current_reweighted_relative_l2": raw_error,
                        }
                        candidates.append((weighted_error, prior, match, metrics))
                    if not candidates:
                        output.append(VectorOracleEvent(
                            current.pair_id, current.split, current.step, current.layer,
                            current_path.expert_id, window, False, None, None, None, None, None, None,
                        ))
                        continue
                    _, prior, _path, metrics = min(candidates, key=lambda item: item[0])
                    output.append(VectorOracleEvent(
                        current.pair_id, current.split, current.step, current.layer,
                        current_path.expert_id, window, True, current.step - prior.step,
                        metrics["hidden_cosine"], metrics["output_cosine"],
                        metrics["output_relative_l2"], metrics["weighted_relative_l2"],
                        metrics["current_reweighted_relative_l2"],
                    ))
    return output


def analyze_npz_vector_capture(capture_dir: Path, pool: dict[str, str]) -> dict[str, Any]:
    tokens, integrity = load_shared_npz_capture(capture_dir, pool)
    if not tokens:
        return {
            "status": "CAPTURE_INCOMPLETE",
            "capture_integrity": integrity,
            "oracle_frontier": [],
            "gate": {"applicable": False, "reason": "no complete source-matched top8 token"},
        }
    events = vector_oracle_events(tokens)
    frontier: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []
    for split in ("discovery", "heldout", "all"):
        for window in RECENCY_WINDOWS:
            local = [
                event for event in events
                if event.window == window and (split == "all" or event.split == split)
            ]
            recurrent = [event for event in local if event.recurred]
            candidate_metrics.append({
                "split": split,
                "window": window,
                "eligible_paths": len(local),
                "recurrent_paths": len(recurrent),
                "recurrence_rate": len(recurrent) / len(local) if local else None,
                "distance_p50": _percentile(
                    [event.distance for event in recurrent if event.distance is not None], 50
                ),
                "distance_p90": _percentile(
                    [event.distance for event in recurrent if event.distance is not None], 90
                ),
                "hidden_cosine_p50": _percentile(
                    [event.hidden_cosine for event in recurrent if event.hidden_cosine is not None], 50
                ),
                "hidden_cosine_p90": _percentile(
                    [event.hidden_cosine for event in recurrent if event.hidden_cosine is not None], 90
                ),
                "output_cosine_p50": _percentile(
                    [event.output_cosine for event in recurrent if event.output_cosine is not None], 50
                ),
                "output_cosine_p90": _percentile(
                    [event.output_cosine for event in recurrent if event.output_cosine is not None], 90
                ),
                "output_relative_l2_min": min(
                    (event.output_relative_l2 for event in recurrent if event.output_relative_l2 is not None),
                    default=None,
                ),
                "output_relative_l2_p50": _percentile(
                    [event.output_relative_l2 for event in recurrent if event.output_relative_l2 is not None], 50
                ),
                "output_relative_l2_p90": _percentile(
                    [event.output_relative_l2 for event in recurrent if event.output_relative_l2 is not None], 90
                ),
                "weighted_relative_l2_min": min(
                    (event.weighted_relative_l2 for event in recurrent if event.weighted_relative_l2 is not None),
                    default=None,
                ),
                "weighted_relative_l2_p50": _percentile(
                    [event.weighted_relative_l2 for event in recurrent if event.weighted_relative_l2 is not None], 50
                ),
                "weighted_relative_l2_p90": _percentile(
                    [event.weighted_relative_l2 for event in recurrent if event.weighted_relative_l2 is not None], 90
                ),
            })
            for threshold in ERROR_THRESHOLDS:
                accepted = [
                    event for event in local
                    if event.recurred
                    and event.weighted_relative_l2 is not None
                    and event.weighted_relative_l2 <= threshold
                ]
                frontier.append({
                    "split": split,
                    "window": window,
                    "weighted_relative_l2_threshold": threshold,
                    "eligible_paths": len(local),
                    "recurrent_paths": len(recurrent),
                    "accepted_paths": len(accepted),
                    "recurrence_rate": len(recurrent) / len(local) if local else None,
                    "reuse_rate": len(accepted) / len(local) if local else None,
                    "hidden_cosine_p50": _percentile(
                        [event.hidden_cosine for event in accepted if event.hidden_cosine is not None], 50
                    ),
                    "output_cosine_p50": _percentile(
                        [event.output_cosine for event in accepted if event.output_cosine is not None], 50
                    ),
                    "output_relative_l2_p50": _percentile(
                        [event.output_relative_l2 for event in accepted if event.output_relative_l2 is not None], 50
                    ),
                    "weighted_relative_l2_p50": _percentile(
                        [event.weighted_relative_l2 for event in accepted if event.weighted_relative_l2 is not None], 50
                    ),
                    "weighted_relative_l2_p90": _percentile(
                        [event.weighted_relative_l2 for event in accepted if event.weighted_relative_l2 is not None], 90
                    ),
                })
    request_counts = {
        split: len({token.pair_id for token in tokens if token.split == split})
        for split in ("discovery", "heldout")
    }
    # Stage-A is a discovery gate.  A negative result stops before held-out;
    # held-out coverage is required only after a positive discovery frontier
    # freezes a deployable B2/B3/B4 policy.
    gate_split = "discovery"
    low_error = [
        row for row in frontier
        if row["split"] == gate_split and row["weighted_relative_l2_threshold"] == 0.01
        and row["reuse_rate"] is not None
    ]
    best = max(
        low_error,
        key=lambda row: (row["reuse_rate"], row["recurrence_rate"] or 0.0),
        default=None,
    )
    discovery_gate_applicable = request_counts["discovery"] >= 4
    heldout_available = request_counts["heldout"] > 0
    if discovery_gate_applicable and heldout_available:
        status = "MEASURED_BOUNDED_VECTOR_ORACLE"
    elif discovery_gate_applicable:
        status = "MEASURED_DISCOVERY_ORACLE"
    else:
        status = "MEASURED_SMOKE_ONLY"
    return {
        "status": status,
        "capture_integrity": integrity,
        "request_count": len({token.pair_id for token in tokens}),
        "request_counts": request_counts,
        "token_layer_steps": len(tokens),
        "candidate_event_count": len(events),
        "oracle_definition": (
            "non-deployable future-truth: among prior same-expert outputs inside W, choose the one "
            "with minimum cached weighted-contribution relative-L2"
        ),
        "candidate_metrics": candidate_metrics,
        "oracle_frontier": frontier,
        "gate": {
            "applicable": discovery_gate_applicable,
            "gate_split": gate_split,
            "registered_very_low_error": 0.01,
            "best_window": None if best is None else best["window"],
            "best_reuse_rate": None if best is None else best["reuse_rate"],
            "pass_15_percent": bool(discovery_gate_applicable and best is not None and best["reuse_rate"] >= 0.15),
            "pass_20_percent": bool(discovery_gate_applicable and best is not None and best["reuse_rate"] >= 0.20),
            "reason": None if discovery_gate_applicable else "fewer than four frozen discovery requests",
            "heldout_required_if_promoted": not heldout_available,
        },
        "vision_policy_status": "NOT_EVALUATED_BY_OUTPUT_ORACLE_CAPTURE",
    }


def capture_audit(vector_capture: Path | None) -> dict[str, Any]:
    if vector_capture is None or not vector_capture.exists():
        return {
            "status": "CAPTURE_REQUIRED",
            "hidden_vectors": "NOT_CAPTURED",
            "raw_expert_outputs": "NOT_CAPTURED",
            "weighted_contributions": "NOT_CAPTURED",
            "existing_support_audit": {
                "long_reasoning_hook": (
                    "poc_mllm_long_reasoning_decode_vision_signal/hooks/sitecustomize.py "
                    "stores route IDs/weights only and only at selected decode checkpoints"
                ),
                "expert_output_hook": (
                    "poc_flashvep/visual_expert_functional_redundancy/instrumentation.py "
                    "can materialize raw per-slot outputs, but intentionally rejects the <50-row "
                    "decode path and was used only for one-token prefill diagnostics"
                ),
                "reusability": (
                    "reuse its per-slot reconstruction/replay logic, not its historical prefill vectors"
                ),
            },
            "required_capture": {
                "cohort": "freeze discovery/heldout before vector inspection; start 4 discovery + 4 heldout",
                "decode": "natural reference or fixed128; contiguous steps 1-128, never beyond natural answer",
                "layers": [0, 9, 19, 28, 38, 47],
                "store_only": [
                    "request/split/step/layer", "BF16 router-input hidden[2048] on one validated TP copy",
                    "top8 IDs and current weights",
                ],
                "offline_replay": (
                    "use frozen BF16 checkpoint experts to compute current per-slot raw outputs; validate a "
                    "small subset against live stock fused-MoE output before oracle analysis"
                ),
                "estimated_raw_hidden_mib_for_8x128x6": 8 * 128 * 6 * HIDDEN_SIZE * BF16_BYTES / 2**20,
                "validation": [
                    "TP0/TP1 hidden parity/tolerance", "top8 uniqueness and route parity",
                    "sum current weighted slot outputs vs stock MoE output on sampled rows",
                    "idle-DP dummy exclusion",
                ],
            },
        }
    if vector_capture.is_dir():
        raw, manifest = _capture_root(vector_capture)
        router = sorted(raw.glob("router.*.npz"))
        experts = sorted(raw.glob("experts.*.npz"))
        return {
            "status": "AVAILABLE_NPZ",
            "path": str(vector_capture),
            "router_shards": len(router),
            "expert_shards": len(experts),
            "manifest": str(manifest) if manifest is not None else None,
        }
    value = read_json(vector_capture)
    required = {"manifest", "rows"}
    if not required.issubset(value):
        raise ValueError(f"vector capture missing keys {required - set(value)}")
    return {
        "status": "AVAILABLE",
        "path": str(vector_capture),
        "row_count": len(value["rows"]),
        "manifest": value["manifest"],
    }


def _bootstrap_mean_difference(
    left: dict[str, float], right: dict[str, float], *, seed: int = 3407, repeats: int = 4000
) -> dict[str, Any]:
    requests = sorted(set(left) & set(right))
    if not requests:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    values = np.asarray([left[key] - right[key] for key in requests], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return {
        "n": len(requests),
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }


def analyze_vector_capture(path: Path) -> dict[str, Any]:
    """Analyze a frozen vector capture without selecting thresholds on held-out."""
    payload = read_json(path)
    rows = payload["rows"]
    total_paths = int(payload["manifest"].get("all_path_denominator", 0))
    if total_paths <= 0:
        raise ValueError("capture manifest needs positive all_path_denominator")
    if not rows:
        raise ValueError("empty vector capture")
    mandatory = {
        "pair_id", "split", "step", "layer", "expert_id", "distance",
        "hidden_cosine", "output_cosine", "output_relative_l2", "weighted_relative_l2",
        "generic_risk", "full_prefill_risk", "separated_risk",
    }
    for row in rows:
        missing = mandatory - set(row)
        if missing:
            raise ValueError(f"capture row missing {sorted(missing)}")
    discovery = [row for row in rows if row["split"] == "discovery"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if not discovery or not heldout:
        raise ValueError("capture needs frozen discovery and heldout rows")

    frontier = []
    for threshold in ERROR_THRESHOLDS:
        accepted = [row for row in rows if float(row["weighted_relative_l2"]) <= threshold]
        frontier.append({
            "weighted_relative_l2_threshold": threshold,
            "accepted_paths": len(accepted),
            "reuse_fraction_of_all_paths": len(accepted) / max(total_paths, 1),
            "output_relative_l2_p50": _percentile([float(row["output_relative_l2"]) for row in accepted], 50),
            "output_relative_l2_p90": _percentile([float(row["output_relative_l2"]) for row in accepted], 90),
        })

    # Policy thresholds are selected once on discovery as the score cut that
    # most closely reaches each registered reuse budget, then applied unchanged.
    comparisons: dict[str, Any] = {}
    score_names = {
        "B2_generic": "generic_risk",
        "B3_full": "full_prefill_risk",
        "B4_separated": "separated_risk",
    }
    per_policy: dict[str, dict[str, Any]] = {}
    for policy, score_name in score_names.items():
        policy_rows = {}
        for budget in REUSE_BUDGETS:
            count = max(1, int(math.floor(len(discovery) * budget)))
            threshold = sorted(float(row[score_name]) for row in discovery)[count - 1]
            selected = [row for row in heldout if float(row[score_name]) <= threshold]
            by_request: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                by_request[str(row["pair_id"])].append(float(row["weighted_relative_l2"]))
            policy_rows[str(budget)] = {
                "frozen_discovery_threshold": threshold,
                "heldout_selected": len(selected),
                "heldout_reuse_fraction_of_candidates": len(selected) / len(heldout),
                "heldout_reuse_fraction_of_all_paths": len(selected) / total_paths,
                "heldout_weighted_relative_l2_mean": (
                    float(np.mean([float(row["weighted_relative_l2"]) for row in selected])) if selected else None
                ),
                "heldout_weighted_relative_l2_p90": _percentile(
                    [float(row["weighted_relative_l2"]) for row in selected], 90
                ),
                "request_error": {key: float(np.mean(value)) for key, value in by_request.items()},
            }
        per_policy[policy] = policy_rows
    for budget in REUSE_BUDGETS:
        key = str(budget)
        separated = per_policy["B4_separated"][key]["request_error"]
        for control in ("B2_generic", "B3_full"):
            comparisons[f"B4_minus_{control}:{budget}"] = _bootstrap_mean_difference(
                separated, per_policy[control][key]["request_error"]
            )

    low_error = next(row for row in frontier if row["weighted_relative_l2_threshold"] == 0.01)
    return {
        "status": "MEASURED_VECTOR_ORACLE",
        "manifest": payload["manifest"],
        "candidate_rows": len(rows),
        "all_path_denominator": total_paths,
        "oracle_frontier": frontier,
        "policies": per_policy,
        "paired_bootstrap": comparisons,
        "gate": {
            "registered_low_error_threshold": 0.01,
            "reuse_fraction": low_error["reuse_fraction_of_all_paths"],
            "pass_15_percent": low_error["reuse_fraction_of_all_paths"] >= 0.15,
            "pass_20_percent": low_error["reuse_fraction_of_all_paths"] >= 0.20,
        },
    }


def render_track_b_reports(root: Path, recurrence: dict[str, Any], oracle: dict[str, Any], ep8: dict[str, Any]) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    heldout = recurrence["by_split_window"]["heldout:W16"]
    all_w16 = recurrence["by_split_window"]["all:W16"]
    temporal = f"""# Track B — Temporal Locality

## Evidence boundary

This report uses **{recurrence['evidence']}**.  The source contains condition A,
TP0, one logical decode row.  It does **not** use the old histogram compact
trace, which sums a real TP row and an idle-DP dummy row.

The historical tracer recorded checkpoints {recurrence['selected_steps']}, not every
decode step.  Recurrence is therefore a sparse-checkpoint structural diagnostic,
not a full W=1/2/4/8/16 cache hit rate.

## Route recurrence

- requests: {recurrence['request_count']}
- route points: {recurrence['point_count']:,}
- W16 all-split recurrence among checkpoint-eligible paths: {all_w16['recurrence_rate']:.4%}
- W16 held-out recurrence: {heldout['recurrence_rate']:.4%}
- hidden vectors: **{oracle.get('hidden_vectors', 'AVAILABLE')}**
- expert outputs: **{oracle.get('raw_expert_outputs', 'AVAILABLE')}**

Route recurrence is necessary but insufficient for output reuse.  No local
error or quality claim is derived from this table.
"""
    (reports / "track_b_temporal_locality.md").write_text(temporal, encoding="utf-8")

    if oracle["status"] == "CAPTURE_REQUIRED":
        oracle_body = """# Track B — Oracle Reuse Frontier

**Status: `CAPTURE_REQUIRED`**

The frozen long-reasoning artifacts contain exact IDs/weights at selected
checkpoints, but do not contain current expert-input hidden vectors or raw
per-expert outputs.  Consequently the required `reuse rate vs true output
error` frontier is **not measured**.  Route recurrence is not substituted for
semantic safety.

Required bounded capture is stored machine-readably in
`artifacts/track_b_capture_audit.json`.  The smallest valid run captures one
validated TP copy at layers 0/9/19/28/38/47 over contiguous decode steps for a
frozen 4-discovery + 4-held-out request subset, then computes slot outputs by
offline BF16 checkpoint replay and validates reconstruction against a sampled
stock MoE output.

Until that capture exists:

- Track-B Stage-A gate: `UNRESOLVED_CAPTURE_REQUIRED`
- quality propagation: `NOT_RUN`
- physical EP2/EP4 timing: `NOT_RUN`
- Track-B verdict: `B_RUNTIME_BLOCKED`
"""
    else:
        gate = oracle["gate"]
        if oracle["status"] in {
            "MEASURED_BOUNDED_VECTOR_ORACLE", "MEASURED_DISCOVERY_ORACLE", "MEASURED_SMOKE_ONLY"
        }:
            reuse = gate.get("best_reuse_rate")
            reuse_text = "N/A" if reuse is None else f"{reuse:.4%}"
            best_window = gate.get("best_window")
            candidate = next(
                (
                    row for row in oracle["candidate_metrics"]
                    if row["split"] == gate.get("gate_split", "all") and row["window"] == best_window
                ),
                {},
            )
            weighted_p50 = candidate.get("weighted_relative_l2_p50")
            weighted_min = candidate.get("weighted_relative_l2_min")
            weighted_p50_text = "N/A" if weighted_p50 is None else f"{weighted_p50:.4f}"
            weighted_min_text = "N/A" if weighted_min is None else f"{weighted_min:.4f}"
            oracle_body = f"""# Track B — Oracle Reuse Frontier

**Status: `{oracle['status']}`**

- complete source-matched token/layer/steps: {oracle['token_layer_steps']:,}
- requests: {oracle['request_count']}
- best reuse at registered 1% cached-weighted-contribution error: {reuse_text}
- best W: {best_window}
- gate applicable: {gate['applicable']}
- 15% gate: {gate['pass_15_percent']}
- 20% strong gate: {gate['pass_20_percent']}
- recurrent-candidate weighted relative-L2 minimum: {weighted_min_text}
- recurrent-candidate weighted relative-L2 median: {weighted_p50_text}

Only receiver rows whose fingerprint appeared in the source-side router shard
were admitted.  The oracle sees true current/prior outputs and is explicitly
non-deployable.  Full per-W values are in `artifacts/track_b_oracle.json`.
"""
        else:
            oracle_body = f"""# Track B — Oracle Reuse Frontier

**Status: `{oracle['status']}`**

- candidate rows: {oracle['candidate_rows']:,}
- all-path denominator: {oracle['all_path_denominator']:,}
- reuse at registered 1% weighted-relative-L2: {gate['reuse_fraction']:.4%}
- 15% gate: {gate['pass_15_percent']}
- 20% strong gate: {gate['pass_20_percent']}

Thresholds for B2/B3/B4 are frozen on discovery and evaluated unchanged on
held-out.  Full values and request-cluster bootstrap intervals are in
`artifacts/track_b_oracle.json`.
"""
    (reports / "track_b_oracle_reuse_frontier.md").write_text(oracle_body, encoding="utf-8")

    if oracle.get("status") == "MEASURED_SMOKE_ONLY":
        quality_reason = "the vector capture is smoke-only"
    elif oracle.get("status") == "MEASURED_DISCOVERY_ORACLE" and not oracle["gate"]["pass_15_percent"]:
        quality_reason = "the frozen discovery oracle failed the registered 15% low-error reuse gate"
    else:
        quality_reason = "Track-B Stage A lacks the required promoted expert-output oracle evidence"
    quality = f"""# Track B — Quality Frontier

`NOT_RUN`: {quality_reason}.
No cached output was propagated through the model, and no local similarity was
misrepresented as long-reasoning task quality.
"""
    runtime = """# Track B — TRUE EP2 / EP4 Runtime

`NOT_RUN`: no Track-B candidate passed oracle plus full-model quality gates.
TPOT/E2E fields remain `N/A`; simulator values are not used.
"""
    safe = ep8["safe_frozen_effect"]
    unsafe = ep8["unsafe_route_recurrence_upper_bound"]
    safe_paths = safe.get("eliminated_expert_paths")
    safe_bytes = safe.get("avoided_logical_dispatch_combine_bytes")
    ep8_report = f"""# Track B — EP8 Structural Projection

Evidence label: **`STRUCTURAL_EP8_PROJECTION_ONLY`**

## Safe frozen discovery effect

- status: **{safe['status']}**
- eliminated expert paths: {safe_paths if safe_paths is not None else 'N/A'}
- avoided logical dispatch+combine bytes: {safe_bytes if safe_bytes is not None else 'N/A'}

The registered <=1% weighted-contribution-error oracle accepted zero paths, so
the only quality-gated structural EP8 effect is **SAFE_EFFECT_0**.

## Unsafe recurrence-only upper bound

This is the route-recurrence maximum for W={unsafe['window']}, not a safe B4 mask.

- eligible sparse-checkpoint paths: {unsafe['eligible_checkpoint_paths']:,}
- recurrent paths: {unsafe['recurrent_paths']:,}
- maximum structural path elimination: {unsafe['maximum_path_elimination_fraction']:.4%}
- avoided remote assignments under source-rank-0 replay: {unsafe['avoided_remote_assignments']:,}
- latency: **FORBIDDEN / NOT COMPUTED**

The measured outputs reject this upper bound as a safe method effect.
"""
    (reports / "track_b_quality_frontier.md").write_text(quality, encoding="utf-8")
    (reports / "track_b_ep2_ep4_runtime.md").write_text(runtime, encoding="utf-8")
    (reports / "track_b_ep8_structural.md").write_text(ep8_report, encoding="utf-8")


def run_track_b(
    *, trace_dirs: Sequence[Path], pool_path: Path, output_root: Path,
    vector_capture: Path | None = None, compact_routes: Path | None = None,
) -> dict[str, Any]:
    pool = load_pool(pool_path)
    if compact_routes is not None:
        points = load_compact_routes(compact_routes, pool)
        route_evidence = "COMMITTED_EXACT_CONDITION_A_TP0_ROUTE_EXPORT"
    else:
        points = load_active_tp0_routes(trace_dirs, pool)
        route_evidence = "MEASURED_FROM_RAW_TRUE_EP4_ROUTE_TRACE"
    events = recurrence_events(points)
    recurrence = summarize_recurrence(events, points, evidence=route_evidence)
    audit = capture_audit(vector_capture)
    if audit["status"] == "AVAILABLE_NPZ" and vector_capture is not None:
        oracle = analyze_npz_vector_capture(vector_capture, pool)
    elif audit["status"] == "AVAILABLE" and vector_capture is not None:
        oracle = analyze_vector_capture(vector_capture)
    else:
        oracle = audit
    unsafe_ep8 = route_recurrence_ep8_projection(events, window=16)
    ep8 = attach_safe_ep8_effect(unsafe_ep8, oracle)
    artifacts = output_root / "artifacts"
    write_json(artifacts / "track_b_temporal_locality.json", recurrence)
    write_json(artifacts / "track_b_capture_audit.json", audit)
    write_json(artifacts / "track_b_oracle.json", oracle)
    write_json(artifacts / "track_b_ep8_structural.json", ep8)
    discovery_oracle_failed = (
        oracle["status"] in {"MEASURED_DISCOVERY_ORACLE", "MEASURED_BOUNDED_VECTOR_ORACLE"}
        and oracle["gate"]["applicable"]
        and not oracle["gate"]["pass_15_percent"]
    )
    result = {
        "structural_status": "MEASURED_SPARSE_CHECKPOINT_RECURRENCE",
        "oracle_status": oracle["status"],
        "quality_status": (
            "NOT_RUN_STAGE_A_FAILED" if discovery_oracle_failed else "NOT_RUN_STAGE_A_UNRESOLVED"
        ),
        "ep2_tpot_gain": None,
        "ep4_tpot_gain": None,
        "ep8_status": (
            "STRUCTURAL_EP8_SAFE_EFFECT_0_WITH_UNSAFE_RECURRENCE_UPPER_BOUND"
            if ep8["safe_frozen_effect"].get("interpretation") == "SAFE_EFFECT_0"
            else "STRUCTURAL_EP8_PROJECTION_ONLY_NO_SAFE_MASK"
        ),
        "verdict": (
            "B_INSUFFICIENT_REUSE_HEADROOM" if discovery_oracle_failed
            else "B_RUNTIME_BLOCKED" if oracle["status"] in {"CAPTURE_REQUIRED", "CAPTURE_INCOMPLETE", "MEASURED_SMOKE_ONLY"}
            else "UNRESOLVED_PENDING_GATE_INTEGRATION"
        ),
    }
    write_json(artifacts / "track_b_result.json", result)
    render_track_b_reports(output_root, recurrence, oracle, ep8)
    return result
