import unittest
from .confirmation import paired_interval


class ConfirmationTests(unittest.TestCase):
    def test_one_start_has_no_invented_confidence_interval(self):
        result = paired_interval([2.])
        self.assertEqual(result['mean_difference'], 2.)
        self.assertEqual(result['t_interval_95'], 'NOT_ESTIMATED')

    def test_paired_three_start_interval(self):
        result = paired_interval([-1., -2., -3.])
        self.assertEqual(result['starts'], 3)
        self.assertEqual(result['mean_difference'], -2.)
        self.assertLess(result['t_interval_95'][0], -2.)
        self.assertGreater(result['t_interval_95'][1], -2.)


if __name__ == '__main__':
    unittest.main()
