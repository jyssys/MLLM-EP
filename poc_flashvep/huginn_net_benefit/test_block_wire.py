"""Whole-block transfer may include only pages owned by the exact frontier."""
import unittest
from .block_wire import owned_frontier_blocks


class BlockWireTests(unittest.TestCase):
    def test_only_completed_depths_are_selected(self):
        self.assertEqual(owned_frontier_blocks([[7, 8], [9, 10]], [(0, 2, 17)], 16), [7, 8])

    def test_decode_tail_reservation_is_not_copied(self):
        self.assertEqual(owned_frontier_blocks([[7, 8, 9]], [(0, 2, 32)], 16), [7, 8])

    def test_multiple_valid_layers_share_their_owned_blocks(self):
        self.assertEqual(owned_frontier_blocks([[7, 8]], [(0, 0, 32), (0, 1, 31)], 16), [7, 8])

    def test_aliasing_depth_tables_is_rejected(self):
        with self.assertRaises(ValueError):
            owned_frontier_blocks([[7], [7]], [(0, 0, 16)], 16)

    def test_frontier_cannot_exceed_owned_pages(self):
        with self.assertRaises(ValueError):
            owned_frontier_blocks([[7]], [(0, 0, 17)], 16)

    def test_invalid_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            owned_frontier_blocks([[7]], [(1, 0, 16)], 16)


if __name__ == '__main__':
    unittest.main()
