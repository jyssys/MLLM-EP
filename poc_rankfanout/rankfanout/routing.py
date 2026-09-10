"""Exact route controls and destination-rank fanout measurements.

The synthetic construction is deliberately stronger than a rank-load-only
control.  For every supported M, F1/F2/F3/F4 have the same *per-expert*
assignment histogram: every one of 128 experts receives M/16 assignments.
Only token-to-expert incidence (and therefore ranks touched per token) changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EPRouteSummary:
    num_tokens: int
    top_k: int
    ep_size: int
    num_experts: int
    mean_fanout: float
    median_fanout: float
    p90_fanout: float
    p99_fanout: float
    frac_f1: float
    frac_f2: float
    frac_f3: float
    frac_f4: float
    frac_full_fanout: float
    fanout_density: float
    active_src_dst_pairs: int
    rank_load_mean: float
    rank_load_max: int
    rank_load_max_over_mean: float
    rank_load_cv: float
    expert_load_mean: float
    expert_load_max: int
    expert_load_max_over_mean: float
    active_experts: int
    expert_hhi: float
    expert_entropy: float
    placement_hash: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def linear_expert_to_rank(num_experts: int = 128, ep_size: int = 4) -> np.ndarray:
    if num_experts <= 0 or ep_size <= 0 or num_experts % ep_size:
        raise ValueError("num_experts must be positive and divisible by ep_size")
    return np.repeat(np.arange(ep_size, dtype=np.int64), num_experts // ep_size)


def placement_digest(expert_to_rank: np.ndarray) -> str:
    mapping = np.asarray(expert_to_rank, dtype=np.int64)
    return sha256(mapping.tobytes()).hexdigest()


def _rank_counts_for_token(token: int, fanout: int) -> tuple[tuple[int, int], ...]:
    if fanout == 1:
        return ((token % 4, 8),)
    if fanout == 2:
        # Each four-token cycle touches every rank twice, with four branches
        # per selected rank: exactly eight assignments/rank/cycle.
        pairs = ((0, 1), (2, 3), (0, 2), (1, 3))
        return tuple((rank, 4) for rank in pairs[token % 4])
    if fanout == 3:
        # In each four-token cycle every rank is omitted once, receives the
        # two-branch share once, and receives a three-branch share twice.
        omitted = token % 4
        low = (omitted + 1) % 4
        return tuple((rank, 2 if rank == low else 3) for rank in range(4) if rank != omitted)
    if fanout == 4:
        return tuple((rank, 2) for rank in range(4))
    raise ValueError("EP4 fanout must be one of 1, 2, 3, 4")


def balanced_fanout_route(
    num_tokens: int,
    fanout: int,
    *,
    num_experts: int = 128,
    ep_size: int = 4,
    top_k: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Construct an exact-histogram EP4/top-8 fanout control.

    ``num_tokens`` must be divisible by 16 so that every expert receives the
    same integer load.  The requested experiment grid (128..8192) satisfies
    this condition.
    """

    if (ep_size, top_k) != (4, 8):
        raise ValueError("the exact constructive control currently targets EP4/top-k=8")
    if num_experts != 128:
        raise ValueError("the Qwen control requires 128 experts")
    if num_tokens <= 0 or num_tokens % 16:
        raise ValueError("num_tokens must be a positive multiple of 16")

    experts_per_rank = num_experts // ep_size
    cursor = np.zeros(ep_size, dtype=np.int64)
    route = np.empty((num_tokens, top_k), dtype=np.int64)
    rng = np.random.default_rng(seed)

    for token in range(num_tokens):
        row: list[int] = []
        for rank, count in _rank_counts_for_token(token, fanout):
            start = int(cursor[rank])
            row.extend(
                rank * experts_per_rank + ((start + j) % experts_per_rank)
                for j in range(count)
            )
            cursor[rank] += count
        if len(row) != top_k or len(set(row)) != top_k:
            raise AssertionError("construction produced invalid top-k row")
        # Slot order must not become a backend-specific confound.  The router
        # weights are uniform in the operator benchmark, so a deterministic
        # per-token permutation preserves all controlled statistics.
        route[token] = np.asarray(row, dtype=np.int64)[rng.permutation(top_k)]

    validate_route(route, linear_expert_to_rank(num_experts, ep_size), fanout)
    return route


def token_fanout(route: np.ndarray, expert_to_rank: np.ndarray) -> np.ndarray:
    route = np.asarray(route, dtype=np.int64)
    mapping = np.asarray(expert_to_rank, dtype=np.int64)
    if route.ndim != 2:
        raise ValueError("route must have shape [tokens, top_k]")
    if route.size and (route.min() < 0 or route.max() >= mapping.size):
        raise ValueError("route contains an expert outside the placement map")
    destination = mapping[route]
    return np.fromiter(
        (np.unique(row).size for row in destination),
        dtype=np.int64,
        count=route.shape[0],
    )


def validate_route(
    route: np.ndarray,
    expert_to_rank: np.ndarray,
    requested_fanout: int | None = None,
) -> None:
    route = np.asarray(route, dtype=np.int64)
    if route.ndim != 2 or route.shape[1] != 8:
        raise ValueError("route must have shape [M, 8]")
    if any(np.unique(row).size != route.shape[1] for row in route):
        raise ValueError("each token must select eight distinct experts")
    fanout = token_fanout(route, expert_to_rank)
    if requested_fanout is not None and not np.all(fanout == requested_fanout):
        raise ValueError("route does not realize the requested fanout")


def summarize_route(
    route: np.ndarray,
    expert_to_rank: np.ndarray,
    *,
    source_rank: int | np.ndarray | None = None,
) -> EPRouteSummary:
    route = np.asarray(route, dtype=np.int64)
    mapping = np.asarray(expert_to_rank, dtype=np.int64)
    validate_route(route, mapping)
    fanout = token_fanout(route, mapping)
    dest = mapping[route]
    expert_load = np.bincount(route.ravel(), minlength=mapping.size)
    ep_size = int(mapping.max()) + 1
    rank_load = np.bincount(dest.ravel(), minlength=ep_size)
    active = expert_load[expert_load > 0].astype(np.float64)
    probabilities = active / active.sum()

    if source_rank is None:
        # A single local route tensor is one sender.  This is sufficient for
        # the invocation-level active pair count used in real route capture.
        active_pairs = int(np.unique(dest).size)
    else:
        src = np.asarray(source_rank)
        if src.ndim == 0:
            src = np.full(route.shape[0], int(src), dtype=np.int64)
        if src.shape != (route.shape[0],):
            raise ValueError("source_rank must be scalar or length M")
        active_pairs = len({(int(src[i]), int(rank)) for i, row in enumerate(dest) for rank in np.unique(row)})

    mean_rank = float(rank_load.mean())
    mean_expert = float(active.mean())
    fractions = {f: float(np.mean(fanout == f)) for f in range(1, 5)}
    return EPRouteSummary(
        num_tokens=int(route.shape[0]),
        top_k=int(route.shape[1]),
        ep_size=ep_size,
        num_experts=int(mapping.size),
        mean_fanout=float(fanout.mean()),
        median_fanout=float(np.median(fanout)),
        p90_fanout=float(np.quantile(fanout, 0.90)),
        p99_fanout=float(np.quantile(fanout, 0.99)),
        frac_f1=fractions[1],
        frac_f2=fractions[2],
        frac_f3=fractions[3],
        frac_f4=fractions[4],
        frac_full_fanout=float(np.mean(fanout == ep_size)),
        fanout_density=float(fanout.mean() / ep_size),
        active_src_dst_pairs=int(active_pairs),
        rank_load_mean=mean_rank,
        rank_load_max=int(rank_load.max()),
        rank_load_max_over_mean=float(rank_load.max() / mean_rank),
        rank_load_cv=float(rank_load.std() / mean_rank),
        expert_load_mean=mean_expert,
        expert_load_max=int(active.max()),
        expert_load_max_over_mean=float(active.max() / mean_expert),
        active_experts=int(active.size),
        expert_hhi=float(np.square(probabilities).sum()),
        expert_entropy=float(-(probabilities * np.log(probabilities)).sum()),
        placement_hash=placement_digest(mapping),
    )


def validate_control_family(routes: Iterable[np.ndarray]) -> dict[str, object]:
    routes = [np.asarray(route, dtype=np.int64) for route in routes]
    if not routes:
        raise ValueError("at least one route is required")
    shapes = {route.shape for route in routes}
    if len(shapes) != 1:
        raise ValueError("all control routes must have identical shapes")
    expert_hists = [np.bincount(route.ravel(), minlength=128) for route in routes]
    if any(not np.array_equal(expert_hists[0], hist) for hist in expert_hists[1:]):
        raise ValueError("per-expert histograms differ across fanout controls")
    mapping = linear_expert_to_rank()
    rank_hists = [np.bincount(mapping[route].ravel(), minlength=4) for route in routes]
    if any(not np.array_equal(rank_hists[0], hist) for hist in rank_hists[1:]):
        raise ValueError("per-rank loads differ across fanout controls")
    return {
        "shape": list(routes[0].shape),
        "expert_histogram_equal": True,
        "rank_load_equal": True,
        "expert_load": expert_hists[0].tolist(),
        "rank_load": rank_hists[0].tolist(),
    }
