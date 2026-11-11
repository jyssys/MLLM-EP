"""Regression tests for the actual selective checkpoint decision seam."""
import unittest
from .planner import choose_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(length=2048, minimum_P_pressure=2, minimum_quantum_ms=1.,
            completion_deadline_ms=1200., handoff_upper_ms=10.,
            execution_candidates=[dict(tokens=2048, loops=1, operation='recurrent', terminal=t,
                cost_ms=20. if t else 15.) for t in (False, True)])
        self.ad = dict(observed=10., available=True, budget_ms=100.,
            future_quantum_budget_ms=100., decode_reserve_ms=20.)

    def select(self, depth=0, now=10.01, pressure=2, ad=None):
        return choose_checkpoint(depth, 32, [16, 24, 28, 31],
            self.ad if ad is None else ad, now, self.cfg, pressure)

    def test_unavailable_uses_atomic_native(self):
        self.assertEqual(self.select(ad=dict(self.ad, available=False)).target, 32)

    def test_stale_cannot_arm(self):
        self.assertEqual(self.select(now=10.1).target, 32)

    def test_no_pressure_cannot_arm(self):
        self.assertEqual(self.select(pressure=1).target, 32)

    def test_legal_future_boundary_only(self):
        self.assertEqual(self.select(depth=28).target, 31)

    def test_reconsideration_reprices_remaining_work(self):
        self.assertLess(self.select(depth=16).target, 32)
        self.assertEqual(self.select(depth=31).target, 32)

    def test_budget_too_small_cannot_arm(self):
        self.assertEqual(self.select(ad=dict(self.ad, future_quantum_budget_ms=.5)).target, 32)

    def test_observed_budget_changes_the_selected_depth(self):
        self.cfg['checkpoint_selection_deadline_ms'] = 350.
        small = self.select(ad=dict(self.ad, budget_ms=20., future_quantum_budget_ms=20.))
        large = self.select(ad=dict(self.ad, budget_ms=200., future_quantum_budget_ms=200.))
        self.assertEqual((small.target, large.target), (24, 16))

    def test_recovery_ceiling_only_allows_later_checkpoint(self):
        self.assertEqual(self.select(ad=dict(self.ad, recovery_remaining_ceiling=2)).target, 31)
        self.assertEqual(self.select(ad=dict(self.ad, recovery_remaining_ceiling=0)).target, 32)


if __name__ == '__main__':
    unittest.main()
