from __future__ import annotations

import unittest

import torch

from poc_mllm_ep_static4_comm_only.instrumentation import _partition_rows


class Static4ContractTest(unittest.TestCase):
    def test_partition_keeps_each_token_in_exactly_one_slice(self) -> None:
        ids = torch.tensor(
            [
                [0, 1, 32, 33, 64, 65, 96, 97],
                [2, 3, 4, 5, 6, 7, 8, 9],
                [34, 35, 36, 37, 38, 39, 40, 41],
                [66, 67, 98, 99, 4, 5, 36, 37],
                [10, 11, 42, 43, 74, 75, 106, 107],
            ],
            dtype=torch.int64,
        )
        slices, stats = _partition_rows(ids)
        flattened = [row for group in slices for row in group]
        self.assertEqual(sorted(flattened), list(range(ids.shape[0])))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(len(stats), 4)

    def test_partition_balances_equal_cost_rows(self) -> None:
        ids = torch.tensor(
            [[rank * 32 for rank in range(4)] * 2 for _ in range(20)],
            dtype=torch.int64,
        )
        _, stats = _partition_rows(ids)
        loads = [row["logical_dispatch_bytes"] for row in stats]
        self.assertEqual(len(set(loads)), 1)
        self.assertTrue(all(row["source_token_rows"] == 5 for row in stats))

    def test_empty_source_still_produces_four_slices(self) -> None:
        slices, stats = _partition_rows(torch.empty((0, 8), dtype=torch.int64))
        self.assertEqual(slices, [[], [], [], []])
        self.assertEqual(
            [row["logical_dispatch_bytes"] for row in stats], [0, 0, 0, 0]
        )


if __name__ == "__main__":
    unittest.main()
