from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Iterable

import numpy as np

from .trace_schema import TraceRecord


def linear_expert_ownership(num_experts: int, ep_size: int) -> np.ndarray:
    if num_experts % ep_size:
        raise ValueError("linear ownership requires experts divisible by EP size")
    return np.repeat(np.arange(ep_size, dtype=np.int64), num_experts // ep_size)


def rank_loads(expert_counts: np.ndarray, ownership: np.ndarray, ep_size: int) -> np.ndarray:
    expert_counts = np.asarray(expert_counts, dtype=np.int64)
    ownership = np.asarray(ownership, dtype=np.int64)
    if expert_counts.shape[-1] != ownership.size:
        raise ValueError("expert vector and ownership map differ")
    return np.stack(
        [expert_counts[..., ownership == rank].sum(axis=-1) for rank in range(ep_size)],
        axis=-1,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 1.0


def rank_temporal_summary(records: Iterable[TraceRecord], horizons=(1, 2, 4)) -> dict:
    groups: dict[tuple, list[TraceRecord]] = defaultdict(list)
    for record in records:
        groups[(record.request_id, record.block_id, record.layer_id)].append(record)
    result = {}
    for horizon in horizons:
        cosines, argmax_same = [], []
        for group in groups.values():
            ordered = sorted(group, key=lambda item: item.iteration_id)
            by_iteration = {item.iteration_id: item for item in ordered}
            for current in ordered:
                future = by_iteration.get(current.iteration_id + horizon)
                if future is None:
                    continue
                left = np.asarray(current.rank_assignment_counts, dtype=float)
                right = np.asarray(future.rank_assignment_counts, dtype=float)
                cosines.append(cosine(left, right))
                argmax_same.append(int(np.argmax(left) == np.argmax(right)))
        result[horizon] = {
            "pairs": len(cosines),
            "rank_cosine_mean": float(np.mean(cosines)) if cosines else float("nan"),
            "argmax_persistence": float(np.mean(argmax_same)) if argmax_same else float("nan"),
        }
    return result


def hot_expert_persistence(
    records: Iterable[TraceRecord], horizons=(1, 2, 4), hot_factor: float = 1.5
) -> dict:
    groups: dict[tuple, list[TraceRecord]] = defaultdict(list)
    for record in records:
        groups[(record.request_id, record.block_id, record.layer_id)].append(record)
    out = {}
    for horizon in horizons:
        numerator = denominator = 0
        weighted_numerator = weighted_denominator = 0.0
        for group in groups.values():
            ordered = sorted(group, key=lambda item: item.iteration_id)
            by_iteration = {item.iteration_id: item for item in ordered}
            for current in ordered:
                future = by_iteration.get(current.iteration_id + horizon)
                if future is None:
                    continue
                current_load = np.asarray(current.expert_assignment_counts, dtype=float)
                future_load = np.asarray(future.expert_assignment_counts, dtype=float)
                hot = current_load >= hot_factor * max(float(current_load.mean()), 1e-12)
                denominator += int(hot.sum())
                numerator += int(np.logical_and(hot, future_load >= hot_factor * max(float(future_load.mean()), 1e-12)).sum())
                weighted_denominator += float(current_load[hot].sum())
                weighted_numerator += float(current_load[np.logical_and(hot, future_load > 0)].sum())
        out[horizon] = {
            "hot_instances": denominator,
            "conditional_persistence": numerator / denominator if denominator else float("nan"),
            "load_weighted_reuse": weighted_numerator / weighted_denominator if weighted_denominator else float("nan"),
        }
    return out


def vector_stats(values: np.ndarray) -> tuple[int, float, float, float]:
    values = np.asarray(values, dtype=float)
    maximum = int(values.max(initial=0))
    mean = float(values.mean()) if values.size else 0.0
    ratio = maximum / mean if mean else 0.0
    cv = float(values.std() / mean) if mean else 0.0
    return maximum, mean, ratio, cv
