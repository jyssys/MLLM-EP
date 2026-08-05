from __future__ import annotations

from poc_flashvep.offline_wavefront.expert_centered_pipeline import (
    ScheduledStage,
    collective_launch_order,
    interval_overlap_ms,
    overlap_summary,
)


def test_common_rank_collective_order_k4() -> None:
    assert collective_launch_order(4) == [
        ScheduledStage("dispatch", 0),
        ScheduledStage("dispatch", 1),
        ScheduledStage("combine", 0),
        ScheduledStage("dispatch", 2),
        ScheduledStage("combine", 1),
        ScheduledStage("dispatch", 3),
        ScheduledStage("combine", 2),
        ScheduledStage("combine", 3),
    ]


def test_overlap_is_interval_intersection_not_enqueue_time() -> None:
    assert interval_overlap_ms((1.0, 4.0), (3.0, 5.0)) == 1.0
    result = overlap_summary(
        dispatch=[(1.0, 2.0)],
        expert=[(1.5, 4.5)],
        combine=[(4.0, 5.0)],
    )
    assert result["dispatch_expert_overlap_ms"] == 0.5
    assert result["expert_combine_overlap_ms"] == 0.5
    assert result["actual_overlap_fraction"] == 0.5
