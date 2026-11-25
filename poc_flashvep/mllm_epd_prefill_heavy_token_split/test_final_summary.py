"""Keep physical idle, scoped decode protection and serving success separate."""
import unittest

from .final_summary import decisions,idle_comparison,static_decode_audit,prose_paragraphs


class FinalSummaryTest(unittest.TestCase):
    def test_paragraph_breaks_do_not_modify_tables_or_technical_sentences(self):
        rows=['A.','B.','C.','| Value |','| 1.05 |','D.']
        self.assertEqual(prose_paragraphs(rows),['A.','B.','','C.','| Value |','| 1.05 |','D.'])

    def test_decode_protection_does_not_promote_serving_or_future_headroom(self):
        value=decisions([dict(scoped_decode_protection=True) for _ in range(4)])
        self.assertEqual(value['tested_static10_decode_protection'],'PASS_IN_ALL_FOUR_C37_CELLS')
        self.assertEqual(value['token_split_runtime'],'NO_GO_ON_TESTED_GRID')
        self.assertEqual(value['actual_future_decode_safe_budget'],'NOT_VALIDATED')
        self.assertEqual(value['full_task_quality'],'HOLD')

    def test_separate_pass_rates_cannot_replace_actual_request_decode_AND(self):
        starts=[];cells=[];results=[]
        for subset in ['Main100','Heldout50']:
            for mode in ['natural','replay']:
                cells.append(dict(method='Static10',subset=subset,mode=mode,decode_relative_gate=True,
                    runtime_decision='NO_GO_ON_TESTED_RUNTIME'))
                for block in range(3):
                    run=f'{subset}_{mode}_{block}'
                    starts.append(dict(method='Static10',subset=subset,mode=mode,block=block,run=run))
                    results.append(dict(run=run,absolute_joint_decode_95_gate=False))
        rows=static_decode_audit(dict(starts=starts,runtime_decisions=cells),dict(results=results))
        self.assertTrue(all(not row['scoped_decode_protection'] for row in rows))

    def test_full_and_central_have_separate_denominators(self):
        def roi(seconds,busy):
            return dict(duration_seconds=seconds,roles={'encoder':dict(kernel_busy_fraction=busy,kernel_idle_fraction=1-busy)},
                E_D_simultaneous_kernel_idle_fraction=.5,E_D_simultaneous_idle_fraction_of_P_busy=.2)
        value=dict(observer_gate_passed=True,ROIs={'full_cohort':roi(100,.1),'central_arrival_window':roi(80,.2)})
        rows=idle_comparison(value)
        self.assertEqual([(r['seconds'],r['busy_percent']) for r in rows],[(100,10),(80,20)])
        value['observer_gate_passed']=False
        with self.assertRaises(AssertionError):idle_comparison(value)


if __name__=='__main__':unittest.main()
