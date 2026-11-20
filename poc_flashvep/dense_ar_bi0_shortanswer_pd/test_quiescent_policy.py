import unittest
from .quiescent_policy import rejection_reasons


class QuiescentSafety(unittest.TestCase):
    def test_every_known_arrival_path_rejects(self):
        self.assertIn('D_ACTIVE',rejection_reasons('q0',0,0,1))
        self.assertIn('D_READY_PENDING',rejection_reasons(None,1,0,1))
        self.assertIn('OTHER_SOURCE_WORK_CAN_BECOME_READY',rejection_reasons(None,0,1,1))
    def test_only_backlogged_and_empty_state_is_eligible(self):
        self.assertEqual(rejection_reasons(None,0,0,1),[])
        self.assertTrue(rejection_reasons(None,0,0,0))
    def test_multiple_reasons_are_not_request_counts(self):
        self.assertEqual(len(rejection_reasons('q0',1,1,0)),4)


if __name__=='__main__':unittest.main()
