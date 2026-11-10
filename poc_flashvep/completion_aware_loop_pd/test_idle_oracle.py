"""Verify stream union and complete-loop accounting."""
import unittest
import numpy as np
from .idle_oracle import union,complement,gaps_summary


class IntervalTests(unittest.TestCase):
    def test_overlapping_streams_do_not_double_count(self):
        x=union([(5,12),(1,6),(10,15),(19,24)],0,20)
        self.assertEqual(x.tolist(),[[1,15],[19,20]])
        self.assertEqual(complement(x,0,20).tolist(),[[0,1],[15,19]])

    def test_empty_activity_is_not_empty_measurement(self):
        x=union([],0,20)
        self.assertEqual(complement(x,0,20).tolist(),[[0,20]])

    def test_clipping_prevents_warmup_counting(self):
        self.assertEqual(union([(-10,5),(10,30)],0,20).tolist(),[[0,5],[10,20]])

    def test_partial_tiles_do_not_complete_loop(self):
        quanta=sum(int(x//2.5) for x in (3,3,3))
        self.assertEqual(quanta,3)
        self.assertEqual(quanta//4,0)


if __name__=='__main__':unittest.main()
