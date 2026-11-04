import unittest
from .activity_figures import merged_intervals


class IntervalTests(unittest.TestCase):
    def test_overlap_union_preserves_real_idle(self):
        rows = [dict(start_monotonic=a, end_monotonic=b) for a, b in
                [(-1., .2), (.1, .5), (.8, 1.2), (1.3, 1.4)]]
        self.assertEqual(merged_intervals(rows, 0., 1.), [(0., .5), (.8, 1.)])

    def test_empty_capture_is_not_a_fabricated_busy_band(self):
        self.assertEqual(merged_intervals([], 0., 1.), [])


if __name__ == '__main__':
    unittest.main()
