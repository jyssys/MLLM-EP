import unittest
from .replay import Cost,simulate


class ReplayTests(unittest.TestCase):
    def test_pipeline_updates_queue(self):
        rows=simulate([0.,0.,0.],[Cost(1.,2.,0.)]*3)
        self.assertEqual([r['P_ready'] for r in rows],[1.,2.,3.])
        self.assertEqual([r['D_start'] for r in rows],[1.,3.,5.])
        self.assertEqual([r['finish'] for r in rows],[3.,5.,7.])

    def test_source_window_moves_wait_to_P(self):
        costs=[Cost(1.,2.,0.)]*5
        a=simulate([0.]*5,costs,window=32)
        b=simulate([0.]*5,costs,window=1)
        self.assertEqual(a[-1]['finish'],11.)
        self.assertEqual(b[-1]['finish'],15.)
        self.assertEqual(b[-1]['Ready_D_wait'],0.)
        self.assertGreater(b[-1]['P_wait'],a[-1]['P_wait'])

    def test_batch_formation_and_first_token_barrier(self):
        rows=simulate([0.,3.,4.,5.],[Cost(1.,.5,2.)]*4,batch=2)
        self.assertEqual(rows[0]['D_start'],4.)
        self.assertEqual(rows[0]['first'],rows[1]['first'])
        self.assertEqual(rows[0]['finish'],7.)
        self.assertEqual(rows[2]['D_start'],7.)

    def test_action_changes_later_queue(self):
        rows=simulate([0.,0.],[Cost(1.,4.,0.),Cost(1.,1.,0.)])
        self.assertEqual(rows[1]['D_start'],5.)
        self.assertEqual(rows[1]['finish'],6.)

    def test_illegal_geometry_rejected(self):
        with self.assertRaises(AssertionError):
            simulate([0.]*2,[Cost(1.,1.,1.)]*2,batch=2,window=1)

    def test_KV_reservation_updates_and_retires(self):
        rows=simulate([0.]*3,[Cost(1.,2.,0.)]*3,kv_slots=1)
        self.assertEqual([r['finish'] for r in rows],[3.,6.,9.])
        self.assertTrue(all(r['reserved_KV_slots_at_P_start']==1 for r in rows))

    def test_rule_sees_no_future_arrivals(self):
        observations=[]
        def choose(index,now,state):
            observations.append((index,now,state))
            return Cost(1.,1.,0.)
        simulate([0.,10.],[Cost(1.,1.,0.)]*2,choose=choose)
        self.assertEqual(observations[0][2]['pending_source_indices'],())
        self.assertEqual(observations[1][1],10.)


if __name__=='__main__':
    unittest.main()
