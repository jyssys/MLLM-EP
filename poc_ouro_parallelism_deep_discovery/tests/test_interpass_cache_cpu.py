import unittest
from poc_ouro_parallelism_deep_discovery.scripts.interpass_cache_screen import bounds, minimum_fetches, exhaustive_check


class CacheTests(unittest.TestCase):
    def test_exhaustive_optimistic_cache(self):
        self.assertEqual(exhaustive_check()['status'], 'PASS')

    def test_cold_and_warm_are_distinct(self):
        value = bounds(6, 2, 4)
        self.assertEqual(value['cold_first_pass_minimum_fetch_bytes'], 18)
        self.assertEqual(value['warm_initial_cache_minimum_fetch_bytes'], 16)

    def test_bypass_can_preserve_pinned_items(self):
        self.assertEqual(minimum_fetches(3, 4, 2, False), 6)
        self.assertEqual(minimum_fetches(3, 4, 2, True), 4)

    def test_full_capacity_and_zero_capacity(self):
        self.assertEqual(bounds(6, 12, 4)['cold_first_pass_minimum_fetch_bytes'], 6)
        self.assertEqual(bounds(6, 12, 4)['warm_initial_cache_minimum_fetch_bytes'], 0)
        self.assertEqual(bounds(6, 0, 4)['cold_first_pass_minimum_fetch_bytes'], 24)

    def test_invalid_arguments(self):
        for values in [(0, 1, 4), (3, -1, 4), (3, 1, 0)]:
            with self.assertRaises(ValueError):
                bounds(*values)


if __name__ == '__main__':
    unittest.main()
