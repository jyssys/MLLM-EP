"""Bounded Track-A consolidation-oracle analysis.

The existing long-reasoning traces contain routes and router weights, but not
per-path expert outputs.  This module deliberately refuses to infer
substitution quality from routing histograms and consumes the project's shared,
bounded decode-vector capture. Clean timing must use a separate run.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOP_K = 8
HIDDEN_SIZE = 2048
REDUCTION_BUDGETS = (0.10, 0.20, 0.30, 0.40)

# Frozen before vector collection.  These are conservative local-vector gates,
# not claims about end-task quality; passing them only promotes Track A to the
# required full-model quality stage.
ORACLE_GATE = {
    "minimum_actual_path_reduction": 0.20,
    "minimum_mean_output_cosine": 0.99,
    "maximum_p90_weighted_contribution_relative_l2": 0.025,
    "maximum_p90_mixture_relative_l2": 0.025,
}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_existing_evidence(root: Path) -> dict[str, Any]:
    """Audit whether Track-A vector evidence already exists.

    The audit examines schemas rather than filenames so a route histogram can
    never be accidentally accepted as an expert-output capture.
    """

    jsonl_files = sorted(root.glob("artifacts/**/*.jsonl"))
    route_rows = 0
    route_ids = False
    route_weights = False
    hidden_vectors = False
    expert_output_vectors = False
    inspected: list[str] = []
    vector_keys = {
        "expert_output",
        "expert_output_vector",
        "hidden_state",
        "hidden_vector",
        "path_output",
    }
    for path in jsonl_files:
        inspected.append(str(path))
        try:
            for index, row in enumerate(_read_jsonl(path)):
                keys = set(row)
                route_rows += int("route_ids" in keys or "expert_count" in keys)
                route_ids |= "route_ids" in keys
                route_weights |= "route_weights" in keys or "expert_mass" in keys
                hidden_vectors |= bool(keys & {"hidden_state", "hidden_vector"})
                expert_output_vectors |= bool(
                    keys & {"expert_output", "expert_output_vector", "path_output"}
                )
                if keys & vector_keys:
                    break
                if index >= 7:
                    break
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

    vector_shards = sorted(root.glob("artifacts/**/experts.*.npz"))
    if vector_shards:
        with np.load(vector_shards[0]) as payload:
            hidden_vectors |= "hidden_states" in payload
            expert_output_vectors |= "raw_outputs" in payload
    return {
        "status": (
            "READY_FOR_ORACLE_ANALYSIS"
            if hidden_vectors and expert_output_vectors
            else "BOUNDED_VECTOR_CAPTURE_REQUIRED"
        ),
        "route_rows_seen_in_sampled_files": route_rows,
        "route_ids_available": route_ids,
        "route_weights_available": route_weights,
        "hidden_vectors_available": hidden_vectors,
        "expert_output_vectors_available": expert_output_vectors,
        "candidate_capture_shards": [str(path) for path in vector_shards],
        "inspected_jsonl_count": len(inspected),
        "reason": (
            "The frozen traces store route IDs/weights and compact histograms, "
            "not per-path Expert_e(h) vectors required by the Track-A oracle."
        ),
        "evidence_label": "FILESYSTEM_SCHEMA_AUDIT",
    }


@dataclass(frozen=True)
class PathVector:
    expert_id: int
    weight: float
    output: np.ndarray


@dataclass(frozen=True)
class TokenGroup:
    request_id: str
    split: str
    layer: int
    step: int
    hidden_hash: str
    paths: tuple[PathVector, ...]


def load_capture_groups(capture_dir: Path) -> tuple[list[TokenGroup], dict[str, Any]]:
    """Load the shared bounded branch capture written by decode_vector_capture."""

    # The router shard is the source of truth for both row identity and decode
    # step.  Receiver/expert call counters can differ across external-DP ranks,
    # so the expert filename's step must not be part of the join key.
    source_steps: dict[tuple[str, int, str], int] = {}
    ambiguous_source_keys: set[tuple[str, int, str]] = set()
    router_shards = sorted(capture_dir.glob("router.*.npz"))
    malformed_router = filtered_non_decode_router_rows = 0
    for shard in router_shards:
        match = re.match(
            r"^router\.(?P<request>.+)\.rank\d+\.layer(?P<layer>\d+)\."
            r"step(?P<step>\d+)\.call\d+\.npz$",
            shard.name,
        )
        if match is None:
            malformed_router += 1
            continue
        with np.load(shard) as payload:
            if "fingerprints" not in payload.files:
                malformed_router += 1
                continue
            # A decode invocation has exactly one logical source row in this
            # bounded B1 capture. Some images require a second chunked-prefill
            # model call, which the shared call counter labels step=1. Those
            # files contain hundreds of rows and must not enter decode oracle
            # accounting even though their fingerprints are source-side.
            if len(payload["fingerprints"]) != 1:
                filtered_non_decode_router_rows += len(payload["fingerprints"])
                continue
            for fingerprint in payload["fingerprints"]:
                source_key = (
                    match.group("request"),
                    int(match.group("layer")),
                    str(fingerprint),
                )
                router_step = int(match.group("step"))
                previous_step = source_steps.get(source_key)
                if previous_step is not None and previous_step != router_step:
                    ambiguous_source_keys.add(source_key)
                else:
                    source_steps[source_key] = router_step

    for source_key in ambiguous_source_keys:
        source_steps.pop(source_key, None)

    paths: dict[tuple[str, int, int, str], dict[int, PathVector]] = defaultdict(dict)
    shards = sorted(capture_dir.glob("experts.*.npz"))
    malformed = 0
    filtered_non_source_rows: set[tuple[str, int, int, str]] = set()
    step_mismatch_shards = 0
    step_mismatch_rows = 0
    for shard in shards:
        match = re.match(
            r"^experts\.(?P<request>.+)\.rank\d+\.layer(?P<layer>\d+)\."
            r"step(?P<step>\d+)\.call\d+\.npz$",
            shard.name,
        )
        if match is None:
            malformed += 1
            continue
        request = match.group("request")
        layer, step = int(match.group("layer")), int(match.group("step"))
        with np.load(shard) as payload:
            required = {
                "fingerprints",
                "expert_ids",
                "router_weights",
                "local_mask",
                "raw_outputs",
            }
            if not required.issubset(payload.files):
                malformed += 1
                continue
            fingerprints = payload["fingerprints"]
            expert_ids = payload["expert_ids"]
            weights = payload["router_weights"]
            local_mask = payload["local_mask"]
            outputs = payload["raw_outputs"]
            if outputs.ndim != 3 or outputs.shape[:2] != expert_ids.shape:
                malformed += 1
                continue
            shard_has_step_mismatch = False
            for row in range(expert_ids.shape[0]):
                fingerprint = str(fingerprints[row])
                source_key = (request, layer, fingerprint)
                canonical_step = source_steps.get(source_key)
                if canonical_step is None:
                    filtered_non_source_rows.add((request, layer, step, fingerprint))
                    continue
                if step != canonical_step:
                    shard_has_step_mismatch = True
                    step_mismatch_rows += 1
                key = (request, layer, canonical_step, fingerprint)
                for slot in range(expert_ids.shape[1]):
                    if not bool(local_mask[row, slot]):
                        continue
                    path = PathVector(
                        expert_id=int(expert_ids[row, slot]),
                        weight=float(weights[row, slot]),
                        output=outputs[row, slot].astype(np.float32),
                    )
                    paths[key].setdefault(path.expert_id, path)
            step_mismatch_shards += int(shard_has_step_mismatch)

    groups: list[TokenGroup] = []
    incomplete = 0
    for key, experts in paths.items():
        if len(experts) != TOP_K:
            incomplete += 1
            continue
        groups.append(TokenGroup(
            request_id=key[0],
            split="discovery",
            layer=key[1],
            step=key[2],
            hidden_hash=key[3],
            paths=tuple(sorted(experts.values(), key=lambda item: item.weight, reverse=True)),
        ))
    return groups, {
        "shards": len(shards),
        "router_shards": len(router_shards),
        "source_fingerprints": len(source_steps),
        "ambiguous_source_fingerprints": len(ambiguous_source_keys),
        "complete_top8_groups": len(groups),
        "unmatched_receiver_rows": 0,
        "ambiguous_receiver_rows": 0,
        "incomplete_top8_groups": incomplete,
        "filtered_non_source_groups": len(filtered_non_source_rows),
        "step_mismatch_shards": step_mismatch_shards,
        "step_mismatch_rows": step_mismatch_rows,
        "malformed_shards": malformed,
        "malformed_router_shards": malformed_router,
        "filtered_non_decode_router_rows": filtered_non_decode_router_rows,
        "capture_schema": "shared_decode_vector_capture_npz_v1",
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 0.0


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def oracle_frontier(groups: list[TokenGroup]) -> dict[str, Any]:
    """Compute a same-token, already-active-target consolidation oracle."""

    if not groups:
        return {
            "status": "BOUNDED_VECTOR_CAPTURE_REQUIRED",
            "gate_pass": False,
            "frontier": [],
            "gate": ORACLE_GATE,
            "evidence_label": "NOT_MEASURED",
        }

    candidates: list[dict[str, Any]] = []
    baselines: dict[tuple[str, int, int, str], np.ndarray] = {}
    group_lookup = {
        (group.request_id, group.layer, group.step, group.hidden_hash): group
        for group in groups
    }
    for key, group in group_lookup.items():
        baseline = sum((path.weight * path.output for path in group.paths), np.zeros(HIDDEN_SIZE))
        baselines[key] = baseline
        denominator = max(float(np.linalg.norm(baseline)), 1e-12)
        # Highest two router-weight paths are frozen primary paths.  Each
        # secondary may consolidate only into an already-selected primary,
        # so the oracle does not invent an uncomputed target expert.
        targets = group.paths[:2]
        for source in group.paths[2:]:
            choices = []
            for target in targets:
                difference = source.output - target.output
                relative = float(np.linalg.norm(difference)) / max(
                    float(np.linalg.norm(source.output)), 1e-12
                )
                contribution = (
                    abs(source.weight) * float(np.linalg.norm(difference)) / denominator
                )
                choices.append((contribution, relative, target, _cosine(source.output, target.output)))
            contribution, relative, target, cosine = min(choices, key=lambda item: item[0])
            candidates.append({
                "group": key,
                "source": source,
                "target": target,
                "weighted_contribution_relative_l2": contribution,
                "expert_output_relative_l2": relative,
                "expert_output_cosine": cosine,
            })
    candidates.sort(key=lambda item: item["weighted_contribution_relative_l2"])
    total_paths = len(groups) * TOP_K
    frontier = []
    for budget in REDUCTION_BUDGETS:
        take = min(len(candidates), int(math.floor(total_paths * budget)))
        chosen = candidates[:take]
        by_group: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
        for item in chosen:
            by_group[item["group"]].append(item)
        before_union: dict[tuple[int, int], set[int]] = defaultdict(set)
        after_union: dict[tuple[int, int], set[int]] = defaultdict(set)
        chosen_sources = {
            (item["group"], item["source"].expert_id) for item in chosen
        }
        for key, group in group_lookup.items():
            event = (group.layer, group.step)
            for path in group.paths:
                before_union[event].add(path.expert_id)
                if (key, path.expert_id) not in chosen_sources:
                    after_union[event].add(path.expert_id)
            for item in by_group.get(key, []):
                after_union[event].add(item["target"].expert_id)
        active_before = sum(len(values) for values in before_union.values())
        active_after = sum(len(values) for values in after_union.values())
        mixture_errors = []
        mixture_cosines = []
        for key, items in by_group.items():
            modified = baselines[key].copy()
            for item in items:
                source, target = item["source"], item["target"]
                modified += source.weight * (target.output - source.output)
            baseline = baselines[key]
            mixture_errors.append(
                float(np.linalg.norm(modified - baseline))
                / max(float(np.linalg.norm(baseline)), 1e-12)
            )
            mixture_cosines.append(_cosine(modified, baseline))
        contribution_errors = [item["weighted_contribution_relative_l2"] for item in chosen]
        output_cosines = [item["expert_output_cosine"] for item in chosen]
        output_l2 = [item["expert_output_relative_l2"] for item in chosen]
        frontier.append({
            "requested_reduction": budget,
            "actual_path_reduction": take / total_paths,
            "consolidated_paths": take,
            "total_paths": total_paths,
            "active_expert_union_before": active_before,
            "active_expert_union_after": active_after,
            "active_expert_union_reduction": (
                (active_before - active_after) / active_before if active_before else 0.0
            ),
            "mean_expert_output_cosine": (
                None if not output_cosines else float(np.mean(output_cosines))
            ),
            "p10_expert_output_cosine": _percentile(output_cosines, 10),
            "median_expert_output_relative_l2": _percentile(output_l2, 50),
            "p90_weighted_contribution_relative_l2": _percentile(contribution_errors, 90),
            "median_mixture_relative_l2": _percentile(mixture_errors, 50),
            "p90_mixture_relative_l2": _percentile(mixture_errors, 90),
            "mean_mixture_cosine": (
                None if not mixture_cosines else float(np.mean(mixture_cosines))
            ),
        })

    point = next((item for item in frontier if item["requested_reduction"] == 0.20), None)
    gate_pass = bool(
        point
        and point["actual_path_reduction"] >= ORACLE_GATE["minimum_actual_path_reduction"]
        and point["mean_expert_output_cosine"] >= ORACLE_GATE["minimum_mean_output_cosine"]
        and point["p90_weighted_contribution_relative_l2"]
        <= ORACLE_GATE["maximum_p90_weighted_contribution_relative_l2"]
        and point["p90_mixture_relative_l2"]
        <= ORACLE_GATE["maximum_p90_mixture_relative_l2"]
    )
    return {
        "status": "MEASURED_BOUNDED_VECTOR_ORACLE",
        "groups": len(groups),
        "candidate_secondary_paths": len(candidates),
        "target_contract": "same-token already-selected top2 expert",
        "frontier": frontier,
        "gate": ORACLE_GATE,
        "gate_pass": gate_pass,
        "evidence_label": "DIAGNOSTIC_VECTOR_ORACLE_NOT_FULL_MODEL_QUALITY",
    }
