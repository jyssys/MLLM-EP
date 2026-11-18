"""A revised scope must neither hide missing starts nor complete old grids."""
import unittest
from .scoped_closeout import coverage


class ScopeCoverageTest(unittest.TestCase):
    def test_extra_historical_starts_do_not_replace_missing_revised_work(self):
        with self.assertRaises(AssertionError):
            coverage(['new1', 'new2'], ['new1', 'old1', 'old2'])

    def test_duplicate_planned_start_is_not_an_independent_start(self):
        with self.assertRaises(AssertionError):
            coverage(['new1', 'new1'], ['new1'])

    def test_historical_starts_can_remain_separate(self):
        self.assertEqual(coverage(['new1', 'new2'], ['new1', 'new2', 'old1']), 2)


if __name__ == '__main__':
    unittest.main()
