"""Small dependency-aware MoE critical-path model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventPolicy:
    dispatch_expert_overlap: float = 0.0
    expert_combine_overlap: float = 0.0

    def __post_init__(self):
        for name, value in (
            ("dispatch_expert_overlap", self.dispatch_expert_overlap),
            ("expert_combine_overlap", self.expert_combine_overlap),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def critical_path_ms(
    dispatch_ms: float,
    rank_expert_ms: list[float],
    combine_ms: float,
    policy: EventPolicy,
) -> float:
    """Return the measured-policy critical path, not a theoretical overlap."""

    expert_ms = max(rank_expert_ms, default=0.0)
    dispatch_visible = dispatch_ms * (1.0 - policy.dispatch_expert_overlap)
    combine_visible = combine_ms * (1.0 - policy.expert_combine_overlap)
    return dispatch_visible + expert_ms + combine_visible
