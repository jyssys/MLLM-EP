import unittest
from .control_replay import simulate
from .replay import Cost


class ControlReplayTests(unittest.TestCase):
    def test_idle_source_still_requires_a_reservation_round_trip(self):
        r=simulate([0.],[Cost(.2,.1,.3)],output=4,idle_RTT=.002)[0]
        self.assertAlmostEqual(r['P_start'],.002)
        self.assertAlmostEqual(r['reservation_wait'],.002)

    def test_active_preparation_can_delay_the_next_source(self):
        rows=simulate([0.,0.],[Cost(.2,.1,.3)]*2,output=4,idle_RTT=.002)
        self.assertGreater(rows[1]['reservation_wait'],.09)
        self.assertGreaterEqual(rows[1]['P_start'],rows[0]['first'])

    def test_full_cohort_decode_and_single_source_are_preserved(self):
        rows=simulate([0.]*8,[Cost(.2,.08,.3)]*8,batch=4,output=16,idle_RTT=.001)
        self.assertEqual(rows[0]['cohort'],(0,1,2,3))
        self.assertEqual(rows[4]['cohort'],(4,5,6,7))
        self.assertGreaterEqual(rows[4]['D_start'],rows[0]['finish'])
        for a,b in zip(rows,rows[1:]):self.assertGreaterEqual(b['P_start'],a['P_ready'])

    def test_choice_sees_no_future_arrival_or_measured_finish(self):
        seen=[]
        def choose(index,now,state):
            seen.append((index,now,state));return Cost(.2,.1,.3)
        simulate([0.,5.],[Cost(.2,.1,.3)]*2,choose=choose,output=4,idle_RTT=.001)
        self.assertEqual(seen[0][2]['pending_source_indices'],())
        self.assertEqual(seen[0][2]['active_indices'],())

    def test_uniform_credit_slots_are_retired(self):
        rows=simulate([0.]*6,[Cost(.2,.1,.3)]*6,batch=2,window=2,kv_slots=2,
            output=4,idle_RTT=.001)
        self.assertEqual(len(rows),6)


if __name__=='__main__':
    unittest.main()
