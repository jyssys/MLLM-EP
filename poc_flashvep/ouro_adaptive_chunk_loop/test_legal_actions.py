"""Synthetic scheduling-definition tests. No GPU speedup evidence."""
import unittest
from .legal_actions import Frontier,enumerate_actions,DecodeNow,complete_next_token_schedule,candidate_feasible

class LegalTests(unittest.TestCase):
    def test_C_cannot_change_in_staged_chunk(self):
        actions=enumerate_actions(Frontier('a',1000,2,512,1024,1000,True))
        self.assertEqual({a.C for a in actions},{512})
        self.assertEqual({a.G for a in actions},{1,2})
        self.assertTrue(all(a.depth+a.G<=4 for a in actions))

    def test_invalid_KV_excludes_work(self):
        self.assertEqual(enumerate_actions(Frontier('a',1000,0,0,2048,2048,False)),())

    def test_completed_frontier_not_partial_work(self):
        actions=enumerate_actions(Frontier('a',512,0,0,1024,1024,True))
        for action in actions:
            self.assertEqual(action.completed_frontier_progress,action.C if action.G==4 else 0)

    def test_remaining_loops_and_coda_are_required(self):
        states=[DecodeNow('a','recurrent',1,10.)]
        completed,timeline=complete_next_token_schedule(states,0.,lambda kind,rs:1.)
        self.assertEqual(completed['a'],4.)
        self.assertEqual([r['phase'] for r in timeline],['recurrent','recurrent','recurrent','coda'])
        self.assertFalse(candidate_feasible(states,0.,7.,lambda kind,rs:1.)[0])
        self.assertTrue(candidate_feasible(states,0.,6.,lambda kind,rs:1.)[0])

    def test_finished_requests_have_no_future_deadline(self):
        completed,timeline=complete_next_token_schedule([DecodeNow('done','finished',4,-100.)],0.,lambda k,s:1.)
        self.assertEqual(completed,{})
        self.assertEqual(timeline,[])

if __name__=='__main__':
    unittest.main()
