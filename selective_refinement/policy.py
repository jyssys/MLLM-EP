"""Interpretable online active-set policies.

All policy inputs are observations from the previous refinement.  The helpers
are deliberately independent of the model so selector behavior can be tested
without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


POLICIES = ("full", "random", "confidence_high", "confidence_low",
            "age_confidence", "ep_load", "confidence_ep")


@dataclass(frozen=True)
class PolicyConfig:
    name: str = "full"
    active_ratio: float = 1.0
    max_freeze_age: int = 1
    periodic_full_refresh: int = 0
    target_ep: int = 8
    lambda_max: float = 0.5
    lambda_cv: float = 0.25
    age_weight: float = 0.1

    def __post_init__(self):
        if self.name not in POLICIES:
            raise ValueError(f"unknown policy {self.name!r}")
        if not 0 < self.active_ratio <= 1:
            raise ValueError("active_ratio must be in (0, 1]")
        if self.max_freeze_age < 0:
            raise ValueError("max_freeze_age must be nonnegative")
        if self.periodic_full_refresh < 0:
            raise ValueError("periodic_full_refresh must be nonnegative")
        if self.target_ep not in (4, 8):
            raise ValueError("target_ep must be 4 or 8")


def _normalized(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    selected = np.asarray(values, dtype=np.float64)[valid]
    if selected.size == 0:
        return result
    lo, hi = float(selected.min()), float(selected.max())
    if hi > lo:
        result[valid] = (selected - lo) / (hi - lo)
    else:
        result[valid] = 0.5
    return result


def _greedy_load_selection(
    candidates: np.ndarray,
    count: int,
    token_rank_load: np.ndarray,
    base_rank_load: np.ndarray,
    semantic_utility: np.ndarray | None,
    lambda_max: float,
    lambda_cv: float,
) -> np.ndarray:
    """Greedily select a subset using previous-step rank signatures.

    ``token_rank_load`` is aggregated across routed layers.  The base vector is
    the previous-step prompt/prior/decoded contribution.  This is an online
    heuristic, not a future-aware oracle.
    """

    selected: list[int] = []
    remaining = list(map(int, candidates))
    loads = np.asarray(base_rank_load, dtype=np.float64).copy()
    utilities = (np.zeros(len(token_rank_load), dtype=np.float64)
                 if semantic_utility is None else np.asarray(semantic_utility, dtype=np.float64))
    while remaining and len(selected) < count:
        best_key = None
        best_position = None
        for position in remaining:
            trial = loads + token_rank_load[position]
            mean = max(float(trial.mean()), 1e-12)
            max_mean = float(trial.max() / mean)
            cv = float(trial.std() / mean)
            score = utilities[position] - lambda_max * (max_mean - 1.0) - lambda_cv * cv
            # Stable position tie-break keeps the policy deterministic.
            key = (score, -position)
            if best_key is None or key > best_key:
                best_key, best_position = key, position
        selected.append(int(best_position))
        loads += token_rank_load[best_position]
        remaining.remove(best_position)
    return np.asarray(selected, dtype=np.int64)


def choose_active_set(
    masked: np.ndarray,
    previous_confidence: np.ndarray | None,
    freeze_age: np.ndarray,
    config: PolicyConfig,
    *,
    block_iteration: int,
    token_rank_load: np.ndarray | None = None,
    base_rank_load: np.ndarray | None = None,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(active_mask, forced_mask)`` for the next refinement.

    Refinement zero is always a full-active warm-up.  Tokens whose age has
    reached ``max_freeze_age`` are mandatory even when this exceeds the nominal
    budget.  No future state is accepted by this API.
    """

    masked = np.asarray(masked, dtype=bool)
    age = np.asarray(freeze_age, dtype=np.int16)
    if masked.ndim != 1 or age.shape != masked.shape:
        raise ValueError("masked/freeze_age must be equal-length vectors")
    active = np.zeros_like(masked)
    forced = np.zeros_like(masked)
    positions = np.flatnonzero(masked)
    if positions.size == 0:
        return active, forced
    full_refresh = (
        block_iteration == 0
        or config.name == "full"
        or config.active_ratio == 1.0
        or (config.periodic_full_refresh > 0
            and block_iteration % config.periodic_full_refresh == 0)
    )
    if full_refresh:
        active[positions] = True
        forced[positions] = block_iteration > 0 and config.name != "full"
        return active, forced

    if config.max_freeze_age == 0:
        mandatory = positions
    else:
        mandatory = positions[age[positions] >= config.max_freeze_age]
    forced[mandatory] = True
    active[mandatory] = True
    nominal = max(1, int(math.ceil(config.active_ratio * positions.size)))
    target = max(nominal, len(mandatory))
    if active.sum() >= target:
        return active, forced

    candidates = positions[~active[positions]]
    need = target - int(active.sum())
    confidence = (np.zeros_like(age, dtype=np.float64)
                  if previous_confidence is None
                  else np.asarray(previous_confidence, dtype=np.float64))
    if confidence.shape != masked.shape:
        raise ValueError("previous_confidence has wrong shape")
    confidence_norm = _normalized(confidence, masked)
    age_norm = age.astype(np.float64) / max(1, int(age[masked].max(initial=1)))

    if config.name == "random":
        rng = np.random.default_rng(random_seed)
        chosen = rng.choice(candidates, size=need, replace=False)
    elif config.name == "confidence_low":
        chosen = candidates[np.lexsort((candidates, confidence[candidates]))[:need]]
    elif config.name in ("confidence_high", "age_confidence"):
        utility = confidence_norm + (
            config.age_weight * age_norm if config.name == "age_confidence" else 0
        )
        chosen = candidates[np.lexsort((candidates, -utility[candidates]))[:need]]
    elif config.name in ("ep_load", "confidence_ep"):
        if token_rank_load is None or base_rank_load is None:
            raise ValueError("EP policies require previous token/base rank loads")
        token_rank_load = np.asarray(token_rank_load, dtype=np.float64)
        base_rank_load = np.asarray(base_rank_load, dtype=np.float64)
        if token_rank_load.shape != (len(masked), config.target_ep):
            raise ValueError("token_rank_load has wrong shape")
        if base_rank_load.shape != (config.target_ep,):
            raise ValueError("base_rank_load has wrong shape")
        utility = None
        if config.name == "confidence_ep":
            utility = confidence_norm + config.age_weight * age_norm
        chosen = _greedy_load_selection(
            candidates, need, token_rank_load, base_rank_load, utility,
            config.lambda_max, config.lambda_cv,
        )
    else:
        # ``full`` is handled above.
        raise AssertionError(config.name)
    active[np.asarray(chosen, dtype=np.int64)] = True
    return active, forced
