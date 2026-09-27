from __future__ import annotations

import functools
import gzip
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .contract import (
    EP_SIZE,
    EXPERTS_PER_RANK,
    GROUP_SIZE,
    NUM_EXPERTS,
    NUM_LAYERS,
    POLICIES,
    PREDICTED_SET_SIZE,
    RANDOM_SEED,
    WINDOWS,
    owner_rank,
    stable_hash,
)


def dense(value: dict[str, float]) -> np.ndarray:
    output = np.zeros(NUM_EXPERTS, dtype=np.float64)
    for key, item in value.items():
        output[int(key)] = float(item)
    return output


def normalize(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    return vector / total if total > 0 else np.zeros_like(vector)


def layer_bucket(layer: int) -> str:
    if layer < 16:
        return "early"
    if layer < 32:
        return "middle"
    return "late"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", compresslevel=9) as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def freeze_pool(cohort: dict[str, Any], *, heldout_count: int = 8) -> dict[str, Any]:
    eligible = [
        dict(row)
        for row in cohort["anchors"]
        if int(row["natural_output_token_count"]) >= 128
    ]
    eligible.sort(key=lambda row: row["pair_id"])
    if len(eligible) % GROUP_SIZE:
        raise RuntimeError(f"eligible pool {len(eligible)} is not divisible by {GROUP_SIZE}")
    family_counts = {
        family: sum(row["dataset"] == family for row in eligible)
        for family in sorted({row["dataset"] for row in eligible})
    }
    quotas = {
        family: max(1, int(math.floor(count * heldout_count / len(eligible))))
        for family, count in family_counts.items()
    }
    while sum(quotas.values()) < heldout_count:
        family = max(
            quotas,
            key=lambda name: (
                family_counts[name] * heldout_count / len(eligible) - quotas[name],
                family_counts[name] - quotas[name],
                name,
            ),
        )
        quotas[family] += 1
    while sum(quotas.values()) > heldout_count:
        family = min(
            (name for name in quotas if quotas[name] > 1),
            key=lambda name: (
                family_counts[name] * heldout_count / len(eligible) - quotas[name],
                name,
            ),
        )
        quotas[family] -= 1

    heldout_ids = set()
    for family, quota in quotas.items():
        members = [row for row in eligible if row["dataset"] == family]
        members.sort(key=lambda row: stable_hash(f"coscheduling-v1:{row['pair_id']}"))
        heldout_ids.update(row["pair_id"] for row in members[:quota])
    for row in eligible:
        row["coscheduling_split"] = (
            "heldout" if row["pair_id"] in heldout_ids else "discovery"
        )
    return {
        "schema_version": 1,
        "selection_before_stage_a_routing_analysis": True,
        "selection_fields": [
            "dataset",
            "image hash",
            "natural output length >=128",
            "visual token count",
        ],
        "routing_oracle_used_for_selection": False,
        "group_size": GROUP_SIZE,
        "eligible_count": len(eligible),
        "recommended_minimum": 24,
        "shortage": max(0, 24 - len(eligible)),
        "family_counts": family_counts,
        "heldout_quotas": quotas,
        "anchors": eligible,
    }


def filter_pool_trace(
    compact_rows: list[dict[str, Any]], pool_ids: set[str]
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in compact_rows
        if row.get("pair_id") in pool_ids
        and row.get("condition") == "A"
        and row.get("repeat_id") is None
    ]
    rows.sort(
        key=lambda row: (
            row["pair_id"],
            row["record_type"],
            int(row["layer"]),
            int(row.get("step", -1)),
        )
    )
    return rows


class TraceView:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.prefill: dict[tuple[str, int], dict[str, Any]] = {}
        self.decode: dict[tuple[str, int, int], dict[str, Any]] = {}
        for row in rows:
            pair = str(row["pair_id"])
            if row["record_type"] == "prefill":
                self.prefill[(pair, int(row["layer"]))] = row
            else:
                self.decode[(pair, int(row["layer"]), int(row["step"]))] = row

    def validate(self, pair_ids: list[str]) -> list[int]:
        expected_prefill = {(pair, layer) for pair in pair_ids for layer in range(NUM_LAYERS)}
        if not expected_prefill.issubset(self.prefill):
            missing = sorted(expected_prefill - set(self.prefill))
            raise RuntimeError(f"missing prefill rows: {missing[:4]}")
        step_sets = []
        for pair in pair_ids:
            steps = {step for request, _layer, step in self.decode if request == pair}
            step_sets.append(steps)
        common = set.intersection(*step_sets)
        if not common or max(common) < 128:
            raise RuntimeError(f"pool lacks complete fixed128 trace checkpoints: {sorted(common)}")
        for pair in pair_ids:
            missing = {
                (layer, step)
                for layer in range(NUM_LAYERS)
                for step in common
                if (pair, layer, step) not in self.decode
            }
            if missing:
                raise RuntimeError(f"incomplete decode trace for {pair}: {sorted(missing)[:4]}")
        return sorted(common)


def _top_set(vector: np.ndarray, count: int = PREDICTED_SET_SIZE) -> set[int]:
    return set(int(value) for value in np.argsort(vector)[-count:])


def predicted_sets(
    trace: TraceView,
    pair_ids: list[str],
    alphas: dict[str, float],
) -> dict[tuple[str, str, int], set[int]]:
    output = {}
    for pair in pair_ids:
        for layer in range(NUM_LAYERS):
            row = trace.prefill[(pair, layer)]
            vision = normalize(dense(row["vision_mass"]))
            text = normalize(dense(row["text_mass"]))
            full = normalize(vision + text)
            alpha = float(alphas[layer_bucket(layer)])
            separated = alpha * vision + (1.0 - alpha) * text
            output[("S2_TEXT", pair, layer)] = _top_set(text)
            output[("S3_FULL", pair, layer)] = _top_set(full)
            output[("S4_SEPARATED", pair, layer)] = _top_set(separated)
    return output


def exact_partition(
    pair_ids: list[str], group_cost: Callable[[tuple[str, ...]], float]
) -> tuple[float, tuple[tuple[str, ...], ...]]:
    if len(pair_ids) % GROUP_SIZE:
        raise ValueError("request count must be divisible by group size")
    pair_ids = sorted(pair_ids)

    @functools.lru_cache(None)
    def solve(remaining: tuple[str, ...]) -> tuple[float, tuple[tuple[str, ...], ...]]:
        if not remaining:
            return 0.0, ()
        first = remaining[0]
        best = None
        for rest in itertools.combinations(remaining[1:], GROUP_SIZE - 1):
            group = tuple(sorted((first, *rest)))
            group_members = set(group)
            leftover = tuple(value for value in remaining if value not in group_members)
            future_cost, future_groups = solve(leftover)
            candidate = (float(group_cost(group)) + future_cost, (group,) + future_groups)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        return best

    return solve(tuple(pair_ids))


def make_assignments(
    pool: dict[str, Any],
    trace: TraceView,
    steps: list[int],
    alphas: dict[str, float],
) -> dict[str, dict[str, list[list[str]]]]:
    metadata = {row["pair_id"]: row for row in pool["anchors"]}
    pairs = list(metadata)
    predicted = predicted_sets(trace, pairs, alphas)

    def actual_cost(group: tuple[str, ...]) -> float:
        return float(
            sum(
                len(
                    set().union(
                        *(
                            set(map(int, trace.decode[(pair, layer, step)]["expert_count"]))
                            for pair in group
                        )
                    )
                )
                for layer in range(NUM_LAYERS)
                for step in steps
            )
        )

    def predictor_cost(policy: str, group: tuple[str, ...]) -> float:
        return float(
            sum(
                len(set().union(*(predicted[(policy, pair, layer)] for pair in group)))
                for layer in range(NUM_LAYERS)
            )
        )

    def length_cost(group: tuple[str, ...]) -> float:
        values = np.asarray(
            [
                [
                    math.log1p(float(metadata[pair]["vision_token_count"])),
                    math.log1p(float(metadata[pair]["prompt_token_count"])),
                ]
                for pair in group
            ]
        )
        return float(np.square(values - values.mean(axis=0)).sum())

    result: dict[str, dict[str, list[list[str]]]] = {policy: {} for policy in POLICIES}
    for split in ("discovery", "heldout"):
        split_pairs = sorted(
            row["pair_id"]
            for row in pool["anchors"]
            if row["coscheduling_split"] == split
        )
        shuffled = list(split_pairs)
        random.Random(RANDOM_SEED + (0 if split == "discovery" else 1)).shuffle(shuffled)
        result["S0_FIFO_RANDOM"][split] = [
            sorted(shuffled[offset : offset + GROUP_SIZE])
            for offset in range(0, len(shuffled), GROUP_SIZE)
        ]
        costs: dict[str, Callable[[tuple[str, ...]], float]] = {
            "S1_LENGTH": length_cost,
            "S2_TEXT": lambda group: predictor_cost("S2_TEXT", group),
            "S3_FULL": lambda group: predictor_cost("S3_FULL", group),
            "S4_SEPARATED": lambda group: predictor_cost("S4_SEPARATED", group),
            "S5_ORACLE": actual_cost,
        }
        for policy, cost in costs.items():
            _value, groups = exact_partition(split_pairs, cost)
            result[policy][split] = [list(group) for group in groups]
    return result


def _pair_jaccard(sets: list[set[int]]) -> float:
    values = []
    for left, right in itertools.combinations(sets, 2):
        values.append(len(left & right) / max(len(left | right), 1))
    return float(np.mean(values)) if values else 1.0


def event_metrics(
    assignments: dict[str, dict[str, list[list[str]]]],
    trace: TraceView,
    steps: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for policy, splits in assignments.items():
        for split, waves in splits.items():
            for window, (low, high) in WINDOWS.items():
                window_steps = [step for step in steps if low <= step <= high]
                for layer in range(NUM_LAYERS):
                    for step in window_steps:
                        for wave_id, wave in enumerate(waves):
                            expert_sets = [
                                set(map(int, trace.decode[(pair, layer, step)]["expert_count"]))
                                for pair in wave
                            ]
                            union = set().union(*expert_sets)
                            per_rank_union = [
                                sum(owner_rank(expert) == rank for expert in union)
                                for rank in range(EP_SIZE)
                            ]
                            rank_counts = np.zeros(EP_SIZE, dtype=np.float64)
                            remote = []
                            for pair in wave:
                                decode = trace.decode[(pair, layer, step)]
                                histogram = dense(decode["expert_count"])
                                rank_counts += histogram.reshape(EP_SIZE, EXPERTS_PER_RANK).sum(axis=1)
                                remote.append(
                                    float(decode["source_normalized_remote_route_fraction"])
                                )
                            rows.append(
                                {
                                    "policy": policy,
                                    "split": split,
                                    "window": window,
                                    "layer": layer,
                                    "step": step,
                                    "wave": wave_id,
                                    "active_expert_union": len(union),
                                    "per_rank_active_expert_union": per_rank_union,
                                    "expert_set_jaccard": _pair_jaccard(expert_sets),
                                    "rank_assignment_counts": rank_counts.tolist(),
                                    "rank_max_mean": float(
                                        rank_counts.max() / max(rank_counts.mean(), 1e-12)
                                    ),
                                    "remote_destination_fraction": float(np.mean(remote)),
                                }
                            )
    return rows


def summarize_metrics(
    events: list[dict[str, Any]], expert_weight_bytes: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("discovery", "heldout"):
        output[split] = {}
        for window in WINDOWS:
            output[split][window] = {}
            for policy in POLICIES:
                group = [
                    row
                    for row in events
                    if row["split"] == split
                    and row["window"] == window
                    and row["policy"] == policy
                ]
                if not group:
                    continue
                unions = np.asarray([row["active_expert_union"] for row in group])
                rank_union = np.asarray(
                    [row["per_rank_active_expert_union"] for row in group]
                )
                rank_counts = np.asarray([row["rank_assignment_counts"] for row in group])
                output[split][window][policy] = {
                    "events": len(group),
                    "mean_active_expert_union": float(unions.mean()),
                    "p50_active_expert_union": float(np.median(unions)),
                    "p90_active_expert_union": float(np.quantile(unions, 0.9)),
                    "per_rank_active_expert_union_mean": rank_union.mean(axis=0).tolist(),
                    "active_expert_count_per_rank_mean": rank_union.mean(axis=0).tolist(),
                    "expert_set_jaccard_mean": float(
                        np.mean([row["expert_set_jaccard"] for row in group])
                    ),
                    "rank_assignment_distribution": (
                        rank_counts.sum(axis=0) / rank_counts.sum()
                    ).tolist(),
                    "rank_max_mean": float(
                        np.mean([row["rank_max_mean"] for row in group])
                    ),
                    "remote_destination_fraction": float(
                        np.mean([row["remote_destination_fraction"] for row in group])
                    ),
                    "expert_weight_working_set_proxy_bytes": float(
                        unions.mean() * expert_weight_bytes
                    ),
                    "expert_weight_working_set_proxy_label": "ESTIMATED_DISTINCT_WEIGHT_BYTES_NOT_HBM_TRAFFIC",
                }
            policies = output[split][window]
            fifo = policies["S0_FIFO_RANDOM"]["mean_active_expert_union"]
            length = policies["S1_LENGTH"]["mean_active_expert_union"]
            oracle = policies["S5_ORACLE"]["mean_active_expert_union"]
            oracle_reduction = (fifo - oracle) / fifo
            for policy, row in policies.items():
                row["union_reduction_vs_fifo"] = (fifo - row["mean_active_expert_union"]) / fifo
                row["union_reduction_vs_length"] = (length - row["mean_active_expert_union"]) / length
                row["recovered_oracle_headroom"] = (
                    (fifo - row["mean_active_expert_union"]) / (fifo - oracle)
                    if fifo != oracle
                    else None
                )
            output[split][window]["oracle_reduction_vs_fifo"] = oracle_reduction
            output[split][window]["oracle_reduction_vs_length"] = (
                (length - oracle) / length
            )
    return output


def paired_layer_bootstrap(
    events: list[dict[str, Any]], left: str, right: str, *, samples: int = 4000
) -> dict[str, Any]:
    selected = [
        row
        for row in events
        if row["split"] == "heldout"
        and row["window"] == "1-128"
        and row["policy"] in (left, right)
    ]
    layer_values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in selected:
        layer_values[int(row["layer"])][row["policy"]].append(
            float(row["active_expert_union"])
        )
    deltas = np.asarray(
        [
            np.mean(layer_values[layer][right]) - np.mean(layer_values[layer][left])
            for layer in sorted(layer_values)
        ]
    )
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap = np.asarray(
        [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(samples)]
    )
    return {
        "beneficial_union_delta_experts": float(deltas.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "clusters": len(deltas),
        "cluster_unit": "MoE layer",
    }
