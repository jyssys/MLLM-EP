import unittest
from .completion import plan_completion,can_reconsider,execution_margin
from .protocol import Boundary


class CompletionTests(unittest.TestCase):
    def test_serial_cost_alone_does_not_admit_starvation(self):
        p=plan_completion(50,8,8,8,125,15,0,250)
        self.assertFalse(p.feasible)
        self.assertGreater(p.predicted_ms,800)

    def test_actual_opportunity_can_accept(self):
        p=plan_completion(27,7,30,30,90,10,0,150)
        self.assertTrue(p.feasible)
        self.assertEqual(p.predicted_ms,37)

    def test_future_floor_is_required(self):
        self.assertFalse(plan_completion(27,7,30,3,90,10,0,150).feasible)

    def test_queue_is_not_a_free_second_slot(self):
        first=plan_completion(35,7,10,20,90,10,0,250)
        second=plan_completion(35,7,10,20,90,10,first.predicted_ms,250)
        self.assertTrue(first.feasible)
        self.assertFalse(second.feasible)

    def test_transient_initial_rejection_can_reconsider(self):
        self.assertTrue(can_reconsider(False,2,4,True,True))
        self.assertFalse(can_reconsider(False,4,4,True,True))

    def test_execution_of_owned_work_still_obeys_deadline(self):
        self.assertEqual(execution_margin(128,10,90,3),25)
        self.assertEqual(execution_margin(128,50,90,3),0)
        self.assertEqual(execution_margin(128,0,90,3,True),0)

    def test_prepare_can_expire_without_duplicate_work(self):
        b=Boundary();epoch=b.offer(2,True,0,.01)
        self.assertFalse(b.prepared(epoch,.02));self.assertTrue(b.reject(epoch))
        newer=b.offer(3,True,.03,.01)
        self.assertFalse(b.prepared(epoch,.031));self.assertTrue(b.prepared(newer,.031))
        self.assertTrue(b.committed(newer));self.assertFalse(b.committed(newer))


if __name__=='__main__':unittest.main()
