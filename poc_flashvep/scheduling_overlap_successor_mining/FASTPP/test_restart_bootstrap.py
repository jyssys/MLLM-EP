import unittest
from analyze_restart_screen import bootstrap_median_interval


class RestartBootstrapTest(unittest.TestCase):
    def test_too_few_restarts(self):
        self.assertEqual(bootstrap_median_interval([10, 20]), (None, None))

    def test_constant_effect(self):
        self.assertEqual(bootstrap_median_interval([12]*5), (12, 12))

    def test_wide_three_restart_evidence_stays_wide(self):
        self.assertEqual(bootstrap_median_interval([0, 50, 100]), (0, 100))


if __name__ == "__main__":
    unittest.main()
