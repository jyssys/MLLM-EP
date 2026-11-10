"""Verify cumulative TPOT accounting without future output knowledge."""
import unittest
from .completion import mean_deadline_remaining,empty_epoch_recoverable,suffix_execution_cost


class MeanDeadlineTests(unittest.TestCase):
    def test_first_interval(self):
        self.assertAlmostEqual(mean_deadline_remaining(0.,1,.1,167.),67.)

    def test_past_fast_tokens_can_create_credit(self):
        self.assertAlmostEqual(mean_deadline_remaining(0.,3,.25,167.),251.)

    def test_overdue_mean_has_no_budget(self):
        self.assertEqual(mean_deadline_remaining(0.,1,.2,167.),0.)

    def test_no_first_output_is_not_a_forecast(self):
        with self.assertRaises(ValueError):mean_deadline_remaining(0.,0,.2,167.)

    def test_empty_epoch_can_recover_a_past_tail(self):
        self.assertTrue(empty_epoch_recoverable('DECODE_TAIL',0,0))

    def test_active_decode_keeps_the_latch(self):
        self.assertFalse(empty_epoch_recoverable('DECODE_TAIL',1,0))

    def test_owned_work_keeps_the_latch(self):
        self.assertFalse(empty_epoch_recoverable('DECODE_TAIL',0,1))

    def test_failed_completion_model_does_not_reset(self):
        self.assertFalse(empty_epoch_recoverable('COMPLETION_RESERVATION_MISSED',0,0))

    def test_suffix_cost_uses_legal_terminal_geometry(self):
        candidates=[dict(tokens=64,loops=1,operation='recurrent',terminal=False,cost_ms=5.),
            dict(tokens=64,loops=1,operation='recurrent',terminal=True,cost_ms=6.),
            dict(tokens=128,loops=1,operation='recurrent',terminal=False,cost_ms=9.)]
        self.assertEqual(suffix_execution_cost(candidates,2,128,6.,0.),(22.,6.))

    def test_no_fitting_terminal_quantum_is_not_a_partial_loop(self):
        candidates=[dict(tokens=64,loops=1,operation='recurrent',terminal=False,cost_ms=5.)]
        self.assertEqual(suffix_execution_cost(candidates,2,128,6.,0.),(float('inf'),float('inf')))


if __name__=='__main__':unittest.main()
