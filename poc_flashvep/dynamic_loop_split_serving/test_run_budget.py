import unittest
from .common import LONG_START_CUTOFF, GPU_DEADLINE
from .run import fits_launch_budget


class BudgetTests(unittest.TestCase):
    def test_waiting_past_cutoff_does_not_authorize_a_late_launch(self):
        self.assertFalse(fits_launch_budget(dict(timeout=1, startup_timeout=1), LONG_START_CUTOFF + 1))

    def test_full_allowance_must_fit(self):
        cfg = dict(timeout=2000, startup_timeout=240)
        self.assertFalse(fits_launch_budget(cfg, GPU_DEADLINE - 2100))
        self.assertTrue(fits_launch_budget(cfg, LONG_START_CUTOFF - 3600))


if __name__ == '__main__':
    unittest.main()
