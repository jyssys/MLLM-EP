"""Dependency model and interval accounting for expert-centered wavefront."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledStage:
    stage: str
    microbatch: int


def collective_launch_order(microbatches: int) -> list[ScheduledStage]:
    """Return the common-rank NCCL order when D and C share one resource."""
    if microbatches < 1:
        raise ValueError("microbatches must be positive")
    order = [ScheduledStage("dispatch", 0)]
    for index in range(microbatches - 1):
        order.append(ScheduledStage("dispatch", index + 1))
        order.append(ScheduledStage("combine", index))
    order.append(ScheduledStage("combine", microbatches - 1))
    return order


def interval_overlap_ms(
    left: tuple[float, float], right: tuple[float, float]
) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def overlap_summary(
    dispatch: list[tuple[float, float]],
    expert: list[tuple[float, float]],
    combine: list[tuple[float, float]],
) -> dict[str, float]:
    de = 0.0
    ec = 0.0
    for expert_interval in expert:
        de += sum(interval_overlap_ms(expert_interval, item) for item in dispatch)
        ec += sum(interval_overlap_ms(expert_interval, item) for item in combine)
    communication = sum(end - start for start, end in dispatch + combine)
    expert_time = sum(end - start for start, end in expert)
    denominator = min(communication, expert_time)
    return {
        "dispatch_expert_overlap_ms": de,
        "expert_combine_overlap_ms": ec,
        "actual_overlap_fraction": (de + ec) / denominator if denominator > 0 else 0.0,
    }
