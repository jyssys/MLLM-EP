import unittest
from types import SimpleNamespace
from ..fairness import CompletionFairness

class FairnessTests(unittest.TestCase):
    def test_submitted_loop_cannot_clear_debt(self):
        request=SimpleNamespace(request_id='a',generated_token_ids=[1],loops_done=0)
        scheduler=SimpleNamespace(requests={'a':request})
        fairness=CompletionFairness(None);fairness.targets=[(request,1)]
        request.loops_done=4
        self.assertTrue(fairness.pending(scheduler))
        request.generated_token_ids.append(2)
        self.assertFalse(fairness.pending(scheduler))
        self.assertEqual(fairness.events[-1]['kind'],'NATIVE_CODA_HOST_DELIVERY')

    def test_reused_id_does_not_keep_old_debt(self):
        old=SimpleNamespace(request_id='a',generated_token_ids=[1])
        new=SimpleNamespace(request_id='a',generated_token_ids=[7])
        fairness=CompletionFairness(None);fairness.targets=[(old,1)]
        self.assertFalse(fairness.pending(SimpleNamespace(requests={'a':new})))
        self.assertEqual(fairness.events[-1]['kind'],'DEBT_REMOVED_NO_LIVE_TARGET')

    def test_one_actual_completion_suffices_not_all_requests(self):
        a=SimpleNamespace(request_id='a',generated_token_ids=[1,2])
        b=SimpleNamespace(request_id='b',generated_token_ids=[1])
        fairness=CompletionFairness(None);fairness.targets=[(a,1),(b,1)]
        self.assertFalse(fairness.pending(SimpleNamespace(requests={'a':a,'b':b})))

if __name__=='__main__':
    unittest.main()
