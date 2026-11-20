import unittest
from .statistics import interval


class PairedStarts(unittest.TestCase):
    def test_scale(self):
        result=interval([2,4,6],[1,2,3])
        self.assertEqual(result['ratio'],2.)
        self.assertEqual(result['CI95'],[2.,2.])
        self.assertEqual(result['resamples'],27)

    def test_requires_three_starts(self):
        with self.assertRaises(AssertionError):interval([1,2],[1,2])


if __name__=='__main__':unittest.main()
