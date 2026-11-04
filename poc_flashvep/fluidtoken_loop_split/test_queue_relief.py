"""Synthetic CPU cases. These numbers are not GPU measurements."""
import unittest
from .decision_model import SplitCost, queue_relief_choice


class QueueRelief(unittest.TestCase):
    def setUp(self):
        self.full = SplitCost(32, .6, 0., .01, 0)
        self.small = SplitCost(31, .58, .03, .012, 0)
        self.large = SplitCost(28, .51, .07, .02, 0)
        self.costs = [self.full, self.small, self.large]

    def select(self, deficit, age=0., pending=False):
        return queue_relief_choice(self.costs, self.full, deficit,
            .1, age, .2, .18, 100, 2., pending)

    def test_relief_can_help_queue_even_when_current_total_increases(self):
        self.assertGreater(self.small.p_service_s + self.small.d_continuation_s, self.full.p_service_s)
        self.assertEqual(self.select(.01)[0], self.small)

    def test_bounded_budget_relieves_an_unclosable_deficit(self):
        self.assertEqual(self.select(1.)[0], self.large)

    def test_no_later_request_pressure_does_not_force_offload(self):
        self.assertEqual(self.select(0.)[0], self.full)

    def test_slack_and_pending_continuation_protect_decode(self):
        self.assertEqual(self.select(1., age=.08)[0], self.full)
        self.assertEqual(self.select(1., pending=True)[0], self.full)


if __name__ == '__main__':
    unittest.main()
