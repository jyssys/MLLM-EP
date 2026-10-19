import math
import unittest
from poc_ouro_parallelism_deep_discovery.scripts.baseline_depth_profile import distribution,jsd,quarters


class Tests(unittest.TestCase):
    def test_distribution(self):self.assertEqual(distribution([2,3,3,4]),[.25,.5,.25])
    def test_jsd_zero(self):self.assertAlmostEqual(jsd([.5,.5,0],[.5,.5,0]),0)
    def test_jsd_disjoint(self):self.assertAlmostEqual(jsd([1,0,0],[0,1,0]),math.log(2))
    def test_quarters_retain_order_and_all_positions(self):
        values=[2,3,4,2,3,4,2,3,4]
        self.assertEqual(sum(quarters(values),[]),values)
    def test_reject_invalid_depth(self):
        with self.assertRaises(ValueError):distribution([1,2,3])
    def test_reject_short_trace(self):
        with self.assertRaises(ValueError):quarters([2,3,4])


if __name__=='__main__':unittest.main()
