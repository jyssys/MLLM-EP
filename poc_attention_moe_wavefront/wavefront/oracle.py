"""Pure, testable wavefront-oracle calculations.

The latency curves passed to this module are empirical inputs.  This module
does not silently substitute token-proportional scaling for missing data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


DEFAULT_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80, 0.90)


def candidate_splits(tokens: int, modality_boundaries: Iterable[int] = ()) -> list[int]:
    """Return normalized and boundary-adjacent cuts, clamped to (0, M)."""
    if tokens < 2:
        return []
    values = {round(tokens * fraction) for fraction in DEFAULT_FRACTIONS}
    for boundary in modality_boundaries:
        values.update(boundary + delta for delta in (-256, -128, 0, 128, 256))
    return sorted({max(1, min(tokens - 1, int(value))) for value in values})


def ideal_wave_ms(attention_head_ms: float, attention_tail_ms: float,
                  moe_head_ms: float, moe_tail_ms: float) -> float:
    """Zero-split-cost one-cut pipeline latency from the preregistered spec."""
    values = (attention_head_ms, attention_tail_ms, moe_head_ms, moe_tail_ms)
    if any(value < 0 for value in values):
        raise ValueError(f"latencies must be non-negative: {values}")
    return attention_head_ms + max(attention_tail_ms, moe_head_ms) + moe_tail_ms


def amdahl_adjusted_ttft(ttft_ms: float, affected_base_ms: float,
                         affected_oracle_ms: float) -> tuple[float, float]:
    """Return counterfactual TTFT and reduction percentage without double counting."""
    if ttft_ms <= 0 or affected_base_ms < 0 or affected_oracle_ms < 0:
        raise ValueError("invalid latency")
    if affected_oracle_ms > affected_base_ms + 1e-9:
        raise ValueError("oracle cannot be slower in an improvement oracle")
    # A noisy stage sum may exceed TTFT.  Only the observed TTFT can be removed.
    removable = min(ttft_ms, affected_base_ms - affected_oracle_ms)
    oracle_ttft = ttft_ms - removable
    return oracle_ttft, 100.0 * removable / ttft_ms


@dataclass(frozen=True)
class SplitPoint:
    request_id: str
    layer: int
    tokens: int
    split: int
    attention_head_ms: float
    attention_tail_ms: float
    moe_head_ms: float
    moe_tail_ms: float

    @property
    def wave_ms(self) -> float:
        return ideal_wave_ms(
            self.attention_head_ms,
            self.attention_tail_ms,
            self.moe_head_ms,
            self.moe_tail_ms,
        )


def compute_oracles(
    points: Iterable[SplitPoint],
    base_by_request_layer: Mapping[tuple[str, int], float],
    modality_boundary_by_request: Mapping[str, int | None],
) -> dict[str, object]:
    """Compute O1/O2/O3/O4 over a complete empirical split table.

    O2 and O4 use a normalized split fraction because request lengths differ.
    The nearest measured cut is used, with deterministic low-index tie breaking.
    """
    rows = list(points)
    if not rows:
        raise ValueError("empty split table")
    by_rl: dict[tuple[str, int], list[SplitPoint]] = {}
    for point in rows:
        key = (point.request_id, point.layer)
        by_rl.setdefault(key, []).append(point)
    if set(by_rl) != set(base_by_request_layer):
        raise ValueError("split/base request-layer keys do not align")

    def best(candidates: Iterable[SplitPoint]) -> SplitPoint:
        return min(candidates, key=lambda item: (item.wave_ms, item.split))

    o1_choice = {key: best(local) for key, local in by_rl.items()}

    fractions = sorted({round(point.split / point.tokens, 6) for point in rows})

    def nearest(local: list[SplitPoint], fraction: float) -> SplitPoint:
        return min(local, key=lambda item: (abs(item.split / item.tokens - fraction), item.split))

    requests = sorted({request for request, _ in by_rl})
    o2_choice: dict[tuple[str, int], SplitPoint] = {}
    o2_fraction: dict[str, float] = {}
    for request in requests:
        local_keys = [key for key in by_rl if key[0] == request]
        fraction = min(
            fractions,
            key=lambda f: sum(nearest(by_rl[key], f).wave_ms for key in local_keys),
        )
        o2_fraction[request] = fraction
        o2_choice.update({key: nearest(by_rl[key], fraction) for key in local_keys})

    o3_choice: dict[tuple[str, int], SplitPoint] = {}
    for key, local in by_rl.items():
        boundary = modality_boundary_by_request.get(key[0])
        if boundary is None:
            # O3 is not semantically defined for text; use O4 later rather than
            # inventing a modality boundary.
            continue
        o3_choice[key] = min(local, key=lambda item: (abs(item.split - boundary), item.split))

    global_fraction = min(
        fractions,
        key=lambda f: sum(nearest(local, f).wave_ms for local in by_rl.values()),
    )
    o4_choice = {key: nearest(local, global_fraction) for key, local in by_rl.items()}

    def aggregate(choice: Mapping[tuple[str, int], SplitPoint]) -> dict[str, float]:
        return {
            request: sum(point.wave_ms for key, point in choice.items() if key[0] == request)
            for request in requests
            if any(key[0] == request for key in choice)
        }

    return {
        "O1": aggregate(o1_choice),
        "O2": aggregate(o2_choice),
        "O3": aggregate(o3_choice),
        "O4": aggregate(o4_choice),
        "O1_choice": o1_choice,
        "O2_choice": o2_choice,
        "O3_choice": o3_choice,
        "O4_choice": o4_choice,
        "O2_fraction": o2_fraction,
        "O4_fraction": global_fraction,
    }
