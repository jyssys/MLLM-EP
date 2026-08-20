from __future__ import annotations

from poc_flashvep.tile_slack_mechanism.operator_replay import (
    _equal_groups,
    _spatial_groups,
)


def test_spatial_groups_partition_tokens_and_preserve_tiles() -> None:
    metadata = {
        "processor_prompt_tokens": 20,
        "images": [
            {
                "token_span": [2, 18],
                "post_merge_grid_hw": [4, 4],
            }
        ],
    }
    groups = _spatial_groups(metadata, 2)
    assert sorted(token for group in groups for token in group) == list(range(20))
    assert [set(range(2, 18)).intersection(group) for group in groups] == [
        {2, 3, 6, 7},
        {4, 5, 8, 9},
        {10, 11, 14, 15},
        {12, 13, 16, 17},
    ]


def test_equal_groups_preserve_spatial_size_multiset() -> None:
    groups = _equal_groups(10, [3, 2, 4, 1], list(range(9, -1, -1)))
    assert [len(group) for group in groups] == [3, 2, 4, 1]
    assert sorted(token for group in groups for token in group) == list(range(10))
