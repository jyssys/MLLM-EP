"""Exercise frozen gate boundaries,co-delivery,overlap and token weighting."""
import unittest
from pathlib import Path
import tempfile
from .reproduction import intervals,frozen_gates
from .failure_analysis import failure_mask,classify
from .request_trace import credit_observation,external
from .profile_analysis import partition,clock_offset,mapped_host_ns,starts_in_enclosure
from .causal_analysis import minimum_fraction,descriptive_correlation
from .decode_path_analysis import token_times
from .policy_trace import weighted_states
from .gate_overlap_analysis import read_holds,state_spans
from .heldout_confirmation import budget_selection,CORE_LABELS,ALL_LABELS


class MetricContractTests(unittest.TestCase):
    def test_confirmation_budget_does_not_use_effect_signs(self):
        self.assertEqual(budget_selection(0,70,1,1)['selected_labels'],ALL_LABELS)
        self.assertEqual(budget_selection(0,50,1,1)['selected_labels'],CORE_LABELS)
        self.assertEqual(budget_selection(0,40,1,1)['selected_labels'],())

    def test_single_output(self):
        self.assertEqual(intervals([dict(time=1.,count=1)],1),[])

    def test_codelivery(self):
        self.assertEqual(intervals([dict(time=1.,count=2),dict(time=1.1,count=4)],4),[0.,100.00000000000009,0.])

    def test_all_eight_combinations(self):
        rows=[]
        for mask in range(8):
            rows.append(dict(TTFT_PASS=not mask&1,ITL_PASS=not mask&2,TPOT_PASS=not mask&4,output_tokens=mask+1))
        self.assertEqual([failure_mask(row) for row in rows],list(range(8)))
        categories,oracles,shapley=classify(rows,2)
        self.assertEqual([r['request_count'] for r in categories],[1]*8)
        self.assertEqual(oracles[0]['SLO_goodput'],.5)
        self.assertEqual(oracles[7]['SLO_goodput'],18.)
        self.assertEqual(sum(r['attributed_failed_requests'] for r in shapley),7.)
        self.assertAlmostEqual(sum(r['attributed_lost_output_tokens'] for r in shapley),35.)

    def test_gate_overlap_is_not_additive(self):
        rows=[dict(TTFT_PASS=False,ITL_PASS=False,TPOT_PASS=True,output_tokens=100)]
        categories,oracles,shapley=classify(rows,1)
        self.assertEqual(oracles[1]['SLO_goodput'],0)
        self.assertEqual(oracles[2]['SLO_goodput'],0)
        self.assertEqual(oracles[3]['SLO_goodput'],100)
        self.assertEqual([r['attributed_lost_output_tokens'] for r in shapley],[50.,50.,0.])

    def test_absent_credit_is_not_zero_blocking(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(credit_observation(Path(folder),{'continuous_source_credit_limit':0}),
                ('UNBOUNDED_COORDINATOR_WINDOW',[]))
            self.assertEqual(credit_observation(Path(folder),{'continuous_source_credit_limit':16}),
                ('NOT_INSTRUMENTED',[]))

    def test_agent_wide_transfer_event_has_no_request(self):
        self.assertIsNone(external(None))
        self.assertIsNone(external('warm0-aabb'))
        self.assertEqual(external('q12-aabb'),'q12')

    def test_frozen_inclusive_boundaries(self):
        self.assertEqual(frozen_gates(5000.,33.,[34.]),(True,True,True))
        self.assertEqual(frozen_gates(5000.+1e-6,33.,[34.]),(False,True,True))
        self.assertEqual(frozen_gates(5000.,33.,[34.+1e-6]),(True,False,True))
        self.assertEqual(frozen_gates(5000.,33.+1e-6,[34.]),(True,True,False))
        self.assertEqual(frozen_gates(1.,None,[]),(True,True,True))

    def test_exclusive_kernel_partition(self):
        self.assertEqual(partition({'A':[[0,5]],'B':[[3,7]]},0,10),
            {'A':3.,'CROSS_CATEGORY_OVERLAP':2.,'B':2.,'NO_OBSERVED_KERNEL':3.})
        self.assertEqual(partition({'A':[[0,5]],'B':[[3,7]]},4,6),
            {'NO_OBSERVED_KERNEL':0.,'CROSS_CATEGORY_OVERLAP':1.,'B':1.})

    def test_large_epoch_kernel_partition(self):
        left=1.7954549262959565e18;right=1.7954549263885486e18
        origin=int(left)
        value=partition({'A':[[origin+100,origin+1000]],'B':[[origin+1100,origin+1200]]},left,right)
        self.assertEqual(sum(value.values()),int(right)-int(left))
        offset=clock_offset(dict(cupti_ns=10**18,host_lower_ns=10**9,host_upper_ns=10**9+2))
        self.assertEqual(mapped_host_ns(1.,offset),10**18-1)

    def test_kernel_record_half_open_enclosure(self):
        self.assertEqual(starts_in_enclosure([2,2,4,6],2,6),3)
        self.assertEqual(starts_in_enclosure([2,2,4,6],6,7),1)
        self.assertEqual(starts_in_enclosure([2,2,4,6],3,3),0)

    def test_frozen_target_is_not_speedup(self):
        value=minimum_fraction([6000.,6000.],[2000.,2000.],5000.)
        self.assertTrue(value['reachable'])
        self.assertAlmostEqual(value['fraction'],.5)
        self.assertFalse(minimum_fraction([6000.],[500.],5000.)['reachable'])

    def test_descriptive_tied_ranks_and_missing_variance(self):
        value=descriptive_correlation([1,1,2,3],[8,8,4,2])
        self.assertAlmostEqual(value['Spearman'],-1.)
        self.assertIsNone(descriptive_correlation([1,1,1],[2,3,4])['Spearman'])
        self.assertEqual(descriptive_correlation([],[])['n'],0)

    def test_worker_client_gap_identity(self):
        rows=[dict(time=1.,worker_time=.99,count=1),dict(time=1.05,worker_time=1.01,count=3)]
        import numpy as np
        client=token_times(rows,'time',3);worker=token_times(rows,'worker_time',3)
        self.assertTrue(np.allclose(np.diff(client),np.diff(worker)+np.diff(client-worker)))
        self.assertTrue(np.allclose(1000*np.diff(client),[50.,0.]))
        self.assertTrue(np.allclose(1000*np.diff(worker),[20.,0.]))

    def test_missing_queue_state_is_not_empty(self):
        result=weighted_states([dict(time=0.,free_KV_blocks=4),dict(time=1.,free_KV_blocks=0)],0.,2.,
            ('free_KV_blocks','decode_ready_waiting'))
        self.assertEqual(result['minima']['free_KV_blocks'],0)
        self.assertEqual(result['time_weighted_means']['free_KV_blocks'],2.)
        self.assertEqual(result['zero_fractions']['free_KV_blocks'],.5)
        self.assertIsNone(result['time_weighted_means']['decode_ready_waiting'])
        self.assertEqual(result['field_status']['decode_ready_waiting'],'PARTIAL_OR_NOT_INSTRUMENTED')
        partial=weighted_states([dict(time=1.,free_KV_blocks=4)],0.,2.,('free_KV_blocks',))
        self.assertIsNone(partial['time_weighted_means']['free_KV_blocks'])
        self.assertEqual(partial['field_coverage_fraction']['free_KV_blocks'],.5)
        self.assertEqual(partial['observed_duration_time_weighted_means']['free_KV_blocks'],4.)

    def test_read_policy_hold_is_a_queue_not_compute(self):
        spans,opened=read_holds([
            dict(event='LOAD_METADATA',time=3.,pending=[],submitted=['q0-a']),
            dict(event='LOAD_METADATA',time=1.,pending=['q0-a'],submitted=[]),
            dict(event='LOAD_METADATA',time=2.,pending=['q0-a'],submitted=[]),
            dict(event='DEFER_READ_SETUP',time=4.,request_id='q1-a')])
        self.assertEqual(spans,{'q0':[[1.,3.]]})
        self.assertEqual(opened,{'q1':4.})
        self.assertEqual(state_spans([dict(time=1.,running_nonchunk=1),dict(time=3.,running_nonchunk=0)],0.,4.,
            'running_nonchunk'),([[1.,3.]],[[1.,3.],[3.,4.]]))


if __name__=='__main__':unittest.main()
