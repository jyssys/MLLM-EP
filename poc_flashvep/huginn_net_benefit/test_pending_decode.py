"""Keep asynchronous grouped Decode work inside the execution reserve."""
import unittest
from .pending_decode import additional_pending_loops


class PendingDecodeTests(unittest.TestCase):
    def test_single_loop_matches_the_qualified_reserve(self):
        self.assertEqual(additional_pending_loops([('a', 1)]), 0)

    def test_logical_coda_does_not_erase_pending_group(self):
        self.assertEqual(additional_pending_loops([('a', 32)]), 31)

    def test_batches_are_not_summed_across_different_requests(self):
        self.assertEqual(additional_pending_loops([('a', 8), ('b', 16)]), 15)

    def test_sequential_pending_groups_of_one_request_are_retained(self):
        self.assertEqual(additional_pending_loops([('a', 8), ('a', 8)]), 15)

    def test_no_pending_work_adds_no_reserve(self):
        self.assertEqual(additional_pending_loops([]), 0)

    def test_sync_fast_path_has_no_inherited_pending_loop_reserve(self):
        self.assertEqual(additional_pending_loops([('a', 32)], already_reserved=0), 32)

    def test_invalid_group_is_rejected(self):
        with self.assertRaises(ValueError):
            additional_pending_loops([('a', 0)])


if __name__ == '__main__':
    unittest.main()
