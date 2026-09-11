from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


def batch_cost(vectors: np.ndarray) -> float:
    return float(np.asarray(vectors, dtype=float).sum(axis=0).max(initial=0.0))


def greedy_complementary(vectors: np.ndarray, batch_size: int, age=None, age_weight: float = 0.0) -> list[list[int]]:
    vectors = np.asarray(vectors, dtype=float)
    remaining = list(range(len(vectors)))
    age = np.zeros(len(vectors), dtype=float) if age is None else np.asarray(age, dtype=float)
    batches = []
    while remaining:
        # Seed with the oldest request.  Complementarity chooses companions;
        # it must not obtain an unreported queue-delay advantage by skipping
        # the head request.
        seed = remaining.pop(0)
        batch: list[int] = [seed]
        aggregate = vectors[seed].copy()
        while remaining and len(batch) < batch_size:
            candidate = min(
                remaining,
                key=lambda idx: float(np.max(aggregate + vectors[idx])) - age_weight * age[idx],
            )
            batch.append(candidate)
            aggregate += vectors[candidate]
            remaining.remove(candidate)
        batches.append(batch)
    return batches


def exact_first_batch(vectors: np.ndarray, batch_size: int) -> tuple[int, ...]:
    vectors = np.asarray(vectors, dtype=float)
    if batch_size > len(vectors):
        raise ValueError("batch exceeds request pool")
    return min(combinations(range(len(vectors)), batch_size), key=lambda ids: batch_cost(vectors[list(ids)]))


def policy_total_cost(vectors: np.ndarray, batches: Iterable[Iterable[int]]) -> float:
    return float(sum(batch_cost(vectors[list(batch)]) for batch in batches))


def predictor_recovery(baseline: float, oracle: float, practical: float) -> float:
    available = baseline - oracle
    return (baseline - practical) / available if available > 0 else 0.0
