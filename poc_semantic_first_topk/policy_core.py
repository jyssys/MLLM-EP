"""Allocation primitives for semantic-first then EP-critical refinement.

All allocations obey real prefix Top-K semantics: choosing k retains slots
``[0, k)`` from Qwen's descending Top-8 router output.  Text tokens stay at
k=8.  The same-budget EP refinement is a swap, not arbitrary branch deletion:
one vision token is decremented while another is restored.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np


FULL_K = (1, 2, 3, 4, 5, 6, 7, 8)
COARSE_K = (1, 2, 4, 6, 8)


def spatial_schedule_k(modality: np.ndarray, coords: np.ndarray,
                       group_k: list[int]) -> np.ndarray:
    """Map a square-grid K schedule onto an arbitrary real vision grid."""
    k = np.full(len(modality), 8, dtype=np.int16)
    vision = modality == "vision"
    if not np.any(vision):
        return k
    height = int(coords[vision, 0].max()) + 1
    width = int(coords[vision, 1].max()) + 1
    side = int(round(len(group_k) ** .5))
    if side * side != len(group_k):
        raise ValueError("schedule must describe a square spatial grid")
    groups = ((coords[vision, 0] * side // height) * side +
              (coords[vision, 1] * side // width)).astype(np.int64)
    k[vision] = np.asarray(group_k, dtype=np.int16)[groups]
    return k


def keep_from_k(k: np.ndarray, width: int = 8) -> np.ndarray:
    return np.arange(width)[None, :] < k[:, None]


def rank_load(ids: np.ndarray, k: np.ndarray, num_ranks: int = 4,
              experts_per_rank: int = 32) -> np.ndarray:
    keep = keep_from_k(k, ids.shape[1])
    ranks = ids // experts_per_rank
    return np.asarray([np.count_nonzero(keep & (ranks == rank))
                       for rank in range(num_ranks)], dtype=np.int64)


def branch_risk(weights: np.ndarray, outputs: np.ndarray, kind: str) -> np.ndarray:
    """Return a nonnegative removal-risk proxy for each routed branch."""
    if kind == "router":
        return weights.astype(np.float64)
    weighted = weights[..., None] * outputs
    contribution = np.linalg.norm(weighted.astype(np.float64), axis=-1)
    combined = weighted.sum(axis=1)
    denominator = np.maximum(np.linalg.norm(combined.astype(np.float64), axis=-1), 1e-12)
    if kind == "contribution":
        return contribution / denominator[:, None]
    if kind == "contribution_squared":
        return np.square(contribution / denominator[:, None])
    raise ValueError(kind)


def previous_allowed(k: int, allowed: tuple[int, ...]) -> int | None:
    choices = [value for value in allowed if value < k]
    return max(choices) if choices else None


def next_allowed(k: int, allowed: tuple[int, ...]) -> int | None:
    choices = [value for value in allowed if value > k]
    return min(choices) if choices else None


@dataclass
class Allocation:
    k: np.ndarray
    risk: float
    dropped: int
    target_dropped: int
    operations: int


def semantic_allocation(risk: np.ndarray, modality: np.ndarray, target_dropped: int,
                        allowed: tuple[int, ...] = FULL_K) -> Allocation:
    """Greedy minimum-risk allocation under a branch-removal target.

    A transition may remove multiple branches on the coarse grid; its score is
    average risk per removed assignment.  We stop at the closest attainable
    target, preferring not to overshoot when the coarse grid cannot represent
    it exactly.
    """
    k = np.full(len(modality), 8, dtype=np.int16)
    vision = np.flatnonzero(modality == "vision")
    total_risk = 0.0
    dropped = 0
    operations = 0
    choices: list[tuple[float, float, int, int, int]] = []
    for token in vision:
        new_k = previous_allowed(8, allowed)
        if new_k is not None:
            width = 8 - new_k
            delta = float(risk[token, new_k:8].sum())
            heapq.heappush(choices, (delta / width, delta, int(token), new_k, width))
    while dropped < target_dropped:
        if not choices:
            break
        _, delta, token, new_k, width = heapq.heappop(choices)
        if dropped and dropped + width > target_dropped:
            break
        k[token] = new_k
        total_risk += delta
        dropped += width
        operations += 1
        following = previous_allowed(new_k, allowed)
        if following is not None:
            next_width = new_k - following
            next_delta = float(risk[token, following:new_k].sum())
            heapq.heappush(choices, (next_delta / next_width, next_delta,
                                    token, following, next_width))
    return Allocation(k=k, risk=total_risk, dropped=dropped,
                      target_dropped=target_dropped, operations=operations)


@dataclass
class Refinement:
    k: np.ndarray
    initial_risk: float
    final_risk: float
    swaps: int
    initial_load: np.ndarray
    final_load: np.ndarray


def ep_same_budget_refinement(ids: np.ndarray, risk: np.ndarray, modality: np.ndarray,
                              initial_k: np.ndarray, risk_slack: float = 0.05,
                              max_swaps: int = 10000) -> Refinement:
    """Reduce critical-rank load with exact assignment-count-preserving swaps.

    A decrement removes the last retained (lowest-router-order) branch.  A
    restore adds the first omitted branch of another token.  Each accepted
    swap must strictly reduce the maximum rank load and keep cumulative risk
    within ``risk_slack`` of the semantic-first allocation.
    """
    k = initial_k.astype(np.int16).copy()
    vision = np.flatnonzero(modality == "vision")
    initial_risk = float(sum(risk[t, int(k[t]):].sum() for t in vision))
    risk_limit = initial_risk * (1.0 + risk_slack) + 1e-12
    current_risk = initial_risk
    initial_load = rank_load(ids, k)
    swaps = 0
    for _ in range(max_swaps):
        loads = rank_load(ids, k)
        old_max = int(loads.max())
        old_profile = tuple(sorted(loads.tolist(), reverse=True))
        critical = set(np.flatnonzero(loads == old_max).tolist())
        remove: list[tuple[float, int, int]] = []
        restore: list[tuple[float, int, int]] = []
        for token in vision:
            kt = int(k[token])
            if kt > 1:
                slot = kt - 1
                rank = int(ids[token, slot] // 32)
                if rank in critical:
                    remove.append((float(risk[token, slot]), int(token), rank))
            if kt < 8:
                slot = kt
                rank = int(ids[token, slot] // 32)
                if rank not in critical:
                    restore.append((float(risk[token, slot]), int(token), rank))
        if not remove or not restore:
            break
        candidates: list[tuple[float, int, int, int, int]] = []
        # Only the safest options per rank are needed to find a good local swap.
        remove = sorted(remove)[:64]
        restore = sorted(restore, reverse=True)[:64]
        for remove_risk, a, rank_a in remove:
            for restore_risk, b, rank_b in restore:
                if a == b:
                    continue
                trial = loads.copy(); trial[rank_a] -= 1; trial[rank_b] += 1
                trial_profile = tuple(sorted(trial.tolist(), reverse=True))
                if trial_profile >= old_profile:
                    continue
                delta = remove_risk - restore_risk
                if current_risk + delta <= risk_limit:
                    candidates.append((delta, a, b, rank_a, rank_b))
        if not candidates:
            break
        delta, a, b, _, _ = min(candidates)
        k[a] -= 1
        k[b] += 1
        current_risk += delta
        swaps += 1
    return Refinement(k=k, initial_risk=initial_risk, final_risk=current_risk,
                      swaps=swaps, initial_load=initial_load,
                      final_load=rank_load(ids, k))


def output_metrics(weights: np.ndarray, outputs: np.ndarray, k: np.ndarray,
                   modality: np.ndarray, renormalize: bool = False) -> dict[str, float]:
    keep = keep_from_k(k, weights.shape[1])
    retained = weights * keep
    if renormalize:
        retained = retained / np.maximum(retained.sum(axis=1, keepdims=True), 1e-12)
    reference = np.einsum("mk,mkh->mh", weights, outputs, optimize=True)
    candidate = np.einsum("mk,mkh->mh", retained, outputs, optimize=True)
    delta = candidate - reference
    vision = modality == "vision"
    return {
        "combined_rel_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)),
        "vision_rel_l2": float(np.linalg.norm(delta[vision]) /
                               max(np.linalg.norm(reference[vision]), 1e-12)),
        "combined_cosine": float(np.dot(candidate.ravel(), reference.ravel()) /
                                 max(np.linalg.norm(candidate) * np.linalg.norm(reference), 1e-12)),
    }
