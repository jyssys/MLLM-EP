import unittest
from .wire_burst import bounded_progress


class BurstTests(unittest.TestCase):
    def test_first_native_result_is_preserved(self):
        pending = [3]
        def step():
            pending[0] -= 1
            return pending[0]
        result, calls = bounded_progress(step, lambda: pending[0], 8, 1.)
        self.assertEqual((result, calls, pending[0]), (2, 3, 0))

    def test_no_progress_stops_after_one_native_call(self):
        self.assertEqual(bounded_progress(lambda: 'blocked', lambda: 4, 32, 1.), ('blocked', 1))

    def test_limit_is_bounded(self):
        pending = [100]
        def step():
            pending[0] -= 1
        self.assertEqual(bounded_progress(step, lambda: pending[0], 2, 1.)[1], 2)

    def test_time_limit_is_checked_between_calls(self):
        values = iter([0., .002])
        pending = [10]
        def step():
            pending[0] -= 1
        self.assertEqual(bounded_progress(step, lambda: pending[0], 32, .001, lambda: next(values))[1], 1)

    def test_invalid_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            bounded_progress(lambda: None, lambda: 1, 0, 1.)


if __name__ == '__main__':
    unittest.main()
