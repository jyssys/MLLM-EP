"""Keep burst timing reproducible and preserve the finite arrival horizon."""
import unittest
from .workloads import modulated_arrivals


class ArrivalTests(unittest.TestCase):
    def test_reproducible_order_and_horizon(self):
        a=modulated_arrivals(3.,750.,123)
        self.assertEqual(a,modulated_arrivals(3.,750.,123))
        self.assertTrue(all(0.<=t<750. for t in a))
        self.assertEqual(a,sorted(a))
        self.assertNotEqual(a,modulated_arrivals(3.,750.,124))
    def test_high_and_low_intensity_are_distinct(self):
        a=modulated_arrivals(3.,4000.,123)
        high=sum(t%40.<8. for t in a)/800.
        low=sum(t%40.>=8. for t in a)/3200.
        self.assertGreater(high,5*low)
        self.assertLess(high,7*low)


if __name__=='__main__':main=unittest.main()
