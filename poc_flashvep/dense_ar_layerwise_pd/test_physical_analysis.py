import unittest
from .physical_analysis import merge,gaps


class UnionTests(unittest.TestCase):
    def test_overlapping_streams_not_summed(self):
        self.assertEqual(merge([(0,2),(1,3),(4,5)]),[[0,3],[4,5]])
        self.assertEqual(gaps([(0,2),(1,3),(4,5)],0,6),[[3,4],[5,6]])

    def test_clip_boundary(self):
        self.assertEqual(gaps([(-2,1),(4,10)],0,6),[[1,4]])


if __name__=='__main__':unittest.main()
