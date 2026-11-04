import unittest
from .online_adapter import select, next_output_mean_budget
from ..fluidtoken_loop_split.decision_model import SplitCost


class OnlineSelectionTests(unittest.TestCase):
    def setUp(self):
        self.full = SplitCost(32, 1., 0., .01, 0)
        self.partial = SplitCost(24, .7, .2, .01, 0)
        self.costs = [self.full, self.partial]
        self.table = dict(ITL_limit_s=.19, TPOT_margin_s=.167)
        self.state = dict(DPI=dict(load=.1, free_blocks=100), current_reserved_blocks=100,
            active_decode=1, later_queue_deficit_s=.3, observed_token_interval_s=.1,
            max_decode_age_s=.01, PPI=2., pending_continuations=0)

    def test_active_decode_is_protected(self):
        result, _ = select(self.costs, self.full, self.state, self.table)
        self.assertEqual(result, self.full)

    def test_empty_decode_does_not_invent_an_existing_deadline(self):
        self.state['active_decode'] = 0
        result, _ = select(self.costs, self.full, self.state, self.table)
        self.assertEqual(result, self.partial)
        self.assertEqual(self.table, dict(ITL_limit_s=.19, TPOT_margin_s=.167))

    def test_pending_and_credit_still_block_empty_decode(self):
        self.state.update(active_decode=0, pending_continuations=1)
        result, _ = select(self.costs, self.full, self.state, self.table)
        self.assertEqual(result, self.full)
        self.state.update(pending_continuations=0, current_reserved_blocks=0)
        result, _ = select(self.costs, self.full, self.state, self.table)
        self.assertEqual(result, self.full)

    def test_request_mean_budget_uses_no_future_length(self):
        budget = next_output_mean_budget(10., 10.36, 10, .04, .048)
        self.assertAlmostEqual(budget, .08)
        self.assertLess((.36 + .04 + .02) / 10, .048)
        self.assertAlmostEqual(next_output_mean_budget(None, 10., 0, .04, .048), .008)

    def test_mean_budget_does_not_relax_the_token_itl_limit(self):
        full = SplitCost(4, .2, 0., .01, 0)
        partial = SplitCost(3, .15, .02, .01, 0)
        table = dict(ITL_limit_s=.065, TPOT_margin_s=.048)
        state = dict(self.state, active_decode=1, observed_token_interval_s=.04,
            max_decode_age_s=0., request_mean_TPOT_extra_budget_s=.08)
        result, _ = select([full, partial], full, state, table)
        self.assertEqual(result, partial)
        state['max_decode_age_s'] = .03
        result, _ = select([full, partial], full, state, table)
        self.assertEqual(result, full)


if __name__ == '__main__':
    unittest.main()
