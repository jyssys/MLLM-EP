import unittest
from .confirmation import arrivals


class ArrivalTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(arrivals(51, 128, 1.971), arrivals(51, 128, 1.971))

    def test_positive_open_loop_intervals(self):
        values = arrivals(52, 128, 1.971)
        self.assertEqual(values[0], 0.)
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))

    def test_distinct_trace(self):
        self.assertNotEqual(arrivals(51, 128, 1.971), arrivals(52, 128, 1.971))
