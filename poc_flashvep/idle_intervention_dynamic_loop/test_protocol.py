import unittest
from .protocol import Boundary,legal_loops,choose,prefix_budget,matches_quantum,initial_token_prefix,should_capture_state,token_reservation_path,distinct_P_pressure,take_ready_batch,forecast_quantum,offer_completed_boundary


class ProtocolTests(unittest.TestCase):
    def test_completion_and_epoch(self):
        b=Boundary()
        with self.assertRaises(ValueError):b.offer(1,False,0.,1.)
        e=b.offer(1,True,0.,1.)
        self.assertFalse(b.prepared(e+1,.1));self.assertFalse(b.prepared(e,2.))
        self.assertTrue(b.reject(e));e=b.offer(2,True,2.,1.)
        self.assertFalse(b.reject(e-1));self.assertTrue(b.prepared(e,2.1))
        b.committed(e)
        self.assertFalse(b.committed(e))
        self.assertFalse(b.reject(e))
        with self.assertRaises(ValueError):b.offer(3,True,2.2,1.)

    def test_causal_producer(self):
        self.assertEqual(legal_loops([[128,128],[0,128],[0,0],[0,0]],0,16,1),3)
        self.assertEqual(legal_loops([[128,128],[0,128],[16,16],[0,0]],0,16,1),1)
        self.assertEqual(legal_loops([[0,8],[0,0]],0,16,0),0)
        self.assertEqual(legal_loops([[16,128],[0,16]],16,32,0),1)

    def test_budget_monotone(self):
        c=[dict(loops=n,cost_ms=n*3,saved_ms=n*4) for n in (1,2,4)]
        self.assertIsNone(choose(c,2));self.assertEqual(choose(c,6)['loops'],2)
        self.assertEqual(choose(c,20)['loops'],4)

    def test_rejected_token_prefix_prelude(self):
        self.assertEqual(prefix_budget(3584,3840,8192),256)
        self.assertEqual(prefix_budget(3840,3968,8192),128)
        with self.assertRaises(ValueError):prefix_budget(3840,3840,8192)

    def test_terminal_cost_is_not_nonterminal_cost(self):
        c=dict(operation='recurrent',tokens=2048,loops=1,terminal=False)
        self.assertTrue(matches_quantum(c,'recurrent',2048,30,2,32))
        self.assertFalse(matches_quantum(c,'recurrent',2048,31,1,32))
        self.assertFalse(matches_quantum(c,'recurrent',128,30,2,32))
        self.assertFalse(matches_quantum(c,'recurrent',2048,30,0,32))
        self.assertFalse(matches_quantum(c,'prelude',2048,0,0,32))

    def test_prelude_has_no_recurrent_work(self):
        c=dict(operation='prelude',tokens=128,loops=0,terminal=False)
        self.assertTrue(matches_quantum(c,'prelude',128,0,0,32))
        self.assertFalse(matches_quantum(c,'recurrent',128,0,32,32))

    def test_short_startup_uses_actual_prompt_length(self):
        self.assertEqual(initial_token_prefix(512,[3584,3840],False),512)
        self.assertEqual(initial_token_prefix(4096,[1024,2048,3072],True),1024)
        with self.assertRaises(ValueError):initial_token_prefix(512,[3584],True)

    def test_observer_does_not_hash_startup_as_measurement(self):
        cfg=dict(seed=100,count=32,state_probe_measured_only=True)
        self.assertFalse(should_capture_state(cfg,100,False))
        self.assertFalse(should_capture_state(cfg,99,True))
        self.assertTrue(should_capture_state(cfg,100,True))
        self.assertTrue(should_capture_state(cfg,131,True))
        self.assertFalse(should_capture_state(cfg,132,True))
        self.assertTrue(should_capture_state(dict(cfg,state_probe_measured_only=False),99,False))

    def test_zero_suffix_uses_native_reservation_not_a_partial_frontier(self):
        self.assertEqual(token_reservation_path(4096,4096),'native')
        self.assertEqual(token_reservation_path(3072,4096),'partial')
        self.assertEqual(token_reservation_path(512,512),'native')
        for prefix in (0,-1,4097):
            with self.assertRaises(ValueError):token_reservation_path(prefix,4096)

    def test_pending_and_ready_request_is_counted_once(self):
        self.assertEqual(distinct_P_pressure({'a'},{'a'}),1)
        self.assertEqual(distinct_P_pressure({'a','b'},{'b','c'},2),5)
        self.assertEqual(distinct_P_pressure(set(),set()),0)

    def test_ready_observation_can_expire_before_take(self):
        class Owner:
            def has_ready(self,phase):return True
            def _take(self,phase):return 'runnable' if phase=='second' else None
        self.assertEqual(take_ready_batch(Owner(),('first','second')),'runnable')
        self.assertIsNone(take_ready_batch(Owner(),('first',)))

    def test_empty_decode_does_not_invent_an_active_batch_reserve(self):
        self.assertEqual(forecast_quantum(0,45.,38.,1.,200.,200.),200.)
        self.assertEqual(forecast_quantum(1,45.,38.,1.,200.,200.),6.)
        self.assertEqual(forecast_quantum(0,45.,38.,1.,200.),6.)
        self.assertEqual(forecast_quantum(2,45.,50.,1.,200.,200.),0.)

    def test_mixed_depth_batch_can_end_before_one_requests_target(self):
        checkpoints=(24,28,30,31)
        self.assertFalse(offer_completed_boundary(4,24,checkpoints))
        self.assertFalse(offer_completed_boundary(28,31,checkpoints))
        self.assertTrue(offer_completed_boundary(28,28,checkpoints))
        with self.assertRaises(ValueError):offer_completed_boundary(29,28,checkpoints)


if __name__=='__main__':unittest.main()
