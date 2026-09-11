from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReplicaDecision:
    expert: int
    home_rank: int
    replica_rank: int
    lifetime: int
    gross_load_benefit: float
    gross_latency_ms: float
    visible_copy_ms: float
    net_latency_ms: float


def equalizing_split(home_load: float, replica_load: float, expert_load: float) -> float:
    """Return tokens assigned to replica to minimize the two affected ranks' max."""
    return float(np.clip((home_load + expert_load - replica_load) / 2.0, 0.0, expert_load))


def apply_replica(rank_vector: np.ndarray, expert_load: float, home: int, replica: int) -> np.ndarray:
    result = np.asarray(rank_vector, dtype=float).copy()
    moved = equalizing_split(result[home] - expert_load, result[replica], expert_load)
    result[home] -= moved
    result[replica] += moved
    return result


def perfect_replica_oracle(
    future_expert_loads: np.ndarray,
    ownership: Sequence[int],
    visible_copy_ms: float,
    latency_per_max_assignment_ms: float,
    lifetime: int,
) -> ReplicaDecision | None:
    """Choose one exact replica using only future loads (an explicitly offline oracle).

    `future_expert_loads` is [future_iteration, expert]. The candidate replica is
    installed now and reused for up to `lifetime` future iterations.
    """
    loads = np.asarray(future_expert_loads, dtype=float)[:lifetime]
    ownership = np.asarray(ownership, dtype=int)
    if loads.ndim != 2 or loads.shape[1] != ownership.size or loads.size == 0:
        return None
    ep_size = int(ownership.max()) + 1
    base_rank = np.stack([loads[:, ownership == rank].sum(axis=1) for rank in range(ep_size)], axis=1)
    base_max = base_rank.max(axis=1)
    best = None
    for expert in range(loads.shape[1]):
        home = int(ownership[expert])
        for replica in range(ep_size):
            if replica == home:
                continue
            candidate = base_rank.copy()
            for step in range(candidate.shape[0]):
                candidate[step] = apply_replica(candidate[step], loads[step, expert], home, replica)
            gross_load = float(np.maximum(0.0, base_max - candidate.max(axis=1)).sum())
            gross_ms = gross_load * latency_per_max_assignment_ms
            decision = ReplicaDecision(
                expert=expert,
                home_rank=home,
                replica_rank=replica,
                lifetime=min(lifetime, loads.shape[0]),
                gross_load_benefit=gross_load,
                gross_latency_ms=gross_ms,
                visible_copy_ms=visible_copy_ms,
                net_latency_ms=gross_ms - visible_copy_ms,
            )
            if best is None or decision.net_latency_ms > best.net_latency_ms:
                best = decision
    return best
