import unittest
from ..budget_controller import BudgetController

class ControllerTests(unittest.TestCase):
    def test_fixed_budget_does_not_use_future_information(self):
        c=BudgetController(initial_ms=80.)
        for i in range(100):
            self.assertEqual(c.observe_output('r',i*.2,True,4),80.)

    def test_low_sample_tail_does_not_change_budget(self):
        c=BudgetController(mode='FEEDBACK',minimum_dwell=1)
        for i in range(10):
            self.assertEqual(c.observe_output('r',i,True,4),80.)

    def test_feedback_changes_only_from_completed_past_gaps(self):
        c=BudgetController(mode='FEEDBACK',target_ITL_ms=100.,minimum_dwell=16)
        for i in range(40):
            c.observe_output('r',i*.2,True,4)
        self.assertEqual(c.value,60.)
        self.assertEqual(len(c.updates),1)

if __name__=='__main__':
    unittest.main()
