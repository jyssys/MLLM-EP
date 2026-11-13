import unittest
from .physical_analysis import union,complement


class IntervalTests(unittest.TestCase):
    def test_multi_stream_union(self):
        self.assertEqual(union([(2,4),(1,3),(4,5),(8,9)],0,10),[[1,5],[8,9]])
    def test_window_clip(self):
        self.assertEqual(union([(-1,2),(4,9)],1,6),[[1,2],[4,6]])
    def test_physical_gaps(self):
        self.assertEqual(complement([[1,5],[8,9]],0,10),[[0,1],[5,8],[9,10]])
    def test_fully_inactive(self):
        self.assertEqual(complement([],2,3),[[2,3]])
    def test_fully_covered(self):
        self.assertEqual(complement([[2,3]],2,3),[])


if __name__=='__main__':unittest.main()
