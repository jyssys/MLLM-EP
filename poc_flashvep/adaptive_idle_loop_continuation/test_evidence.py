import unittest
from .analyze import latency,coverage_check,nonempty_core_upper_envelopes
from ..native_loop_2d_oracle.dependency import Frontier


class EvidenceTests(unittest.TestCase):
    def test_nonempty_core_upper_bound_keeps_unknown_internal_idle(self):
        chain,envelopes=nonempty_core_upper_envelopes([(0,2),(3,5),(4,9),(7,10)])
        self.assertEqual(chain,[(0,2),(3,5),(7,10)])
        # The longest possible gap between selected unknown kernel positions
        # spans enclosure interiors. It is not merely the external two-unit gap.
        self.assertEqual(max(b-a for a,b in envelopes),7)

    def test_missing_core_enclosures_do_not_create_a_finite_idle_bound(self):
        self.assertEqual(nonempty_core_upper_envelopes([]),([],[]))

    def test_empty_trace_never_means_idle(self):
        capture=dict(native_graph_owners=[dict(replays=10)],activation=dict(model_device='cuda:1',current_device=1),
            status=0,dropped=0,invalid=0)
        self.assertFalse(coverage_check(capture,[])['passed'])

    def test_missing_native_graph_or_unreturned_buffer_fails_coverage(self):
        capture=dict(native_graph_owners=[dict(replays=2)],activation=dict(model_device='cuda:1',current_device=1),
            status=0,dropped=0,invalid=0,collector_counters=dict(requests=2,completions=1))
        records=[dict(kind=51)]*2
        self.assertFalse(coverage_check(capture,records)['passed'])
        capture['collector_counters']['completions']=2
        self.assertTrue(coverage_check(capture,records)['passed'])
        self.assertFalse(coverage_check(capture,records[:1])['passed'])

    def test_empty_delivery_population_is_not_zero_latency(self):
        result=latency([],0,0,8)
        self.assertIsNone(result['ITL_p99_ms'])
        self.assertIsNone(result['max_ITL_ms'])
        self.assertEqual(result['ITL_samples'],0)

    def test_window_selects_overlapping_intervals_without_creating_tokens(self):
        row=dict(first_token=1.,finish=3.,tpot_ms=1000.,token_events=[
            dict(first=True,new_tokens=1,previous_time=1.,time=1.),
            dict(first=False,new_tokens=1,previous_time=1.,time=2.),
            dict(first=False,new_tokens=2,previous_time=2.,time=3.)])
        value=latency([row],0,1.5,2.5)
        self.assertEqual(value['ITL_samples'],1)
        self.assertEqual(value['ITL_p99_ms'],1000.)

    def test_one_way_suffix_structural_accounting_does_not_skip_work(self):
        for model,t,r in (('ouro',4096,4),('huginn',2048,32)):
            for c in range(r+1):
                frontier=Frontier(t,r,c,0)
                accounting=frontier.accounting(model)
                self.assertIsInstance(accounting,dict)
                # The constructor validates a legal monotone frontier.
                self.assertEqual(accounting['D_recurrent_token_rows'],(r-c)*t)
                self.assertTrue(accounting['DAG_closed_P_partition'])


if __name__=='__main__':unittest.main()
