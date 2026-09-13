from __future__ import annotations

import math

import pytest


BLOCKS = (8, 16, 32, 64, 128)


def legacy_total(prompt_length: int, generation: int, block: int) -> int:
    return block * ((prompt_length + generation) // block)


def scheduled_blocks(start: int, total: int, schedule: tuple[int, ...]) -> list[int]:
    """Reference rule used by the bounded dynamic-schedule implementation.

    Schedules repeat at block boundaries.  A final block is shortened to the
    exact remaining multiple of the common block granularity; an open block is
    never resized.
    """
    granularity = math.gcd(*schedule)
    assert total >= start and (total - start) % granularity == 0
    result = []
    position = start
    step = 0
    while position < total:
        width = min(schedule[step % len(schedule)], total - position)
        assert width % granularity == 0
        result.append(width)
        position += width
        step += 1
    return result


def visible(mask_row: list[bool], prefix_end: int, block_start: int, block_end: int) -> bool:
    assert all(mask_row[:prefix_end])
    assert all(mask_row[block_start:block_end])
    assert not any(mask_row[block_end:])
    return True


def test_legacy_generation_extent_depends_on_block() -> None:
    totals = {b: legacy_total(197, 128, b) for b in BLOCKS}
    assert len(set(totals.values())) > 1
    assert totals[8] == 320
    assert totals[128] == 256


@pytest.mark.parametrize("schedule", [(16, 64, 32, 16), (64, 16), (128,), (8,)])
def test_dynamic_schedule_reaches_exact_extent(schedule: tuple[int, ...]) -> None:
    start, total = 128, 384
    blocks = scheduled_blocks(start, total, schedule)
    assert sum(blocks) == total - start
    assert all(width > 0 for width in blocks)


def test_completed_prefix_and_open_block_are_visible_but_future_is_not() -> None:
    total, prefix_end, start, end = 128, 48, 48, 80
    row = [index < end for index in range(total)]
    assert visible(row, prefix_end, start, end)
