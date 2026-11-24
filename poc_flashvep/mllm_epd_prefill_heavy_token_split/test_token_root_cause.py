"""Keep observed costs, metric attribution and hypothetical improvements separate."""
import unittest

from .token_confirmation_root_cause import failure_accounting,zero_gate_oracle


class FailureAccountingTest(unittest.TestCase):
    def test_overlapping_failures_are_not_counted_twice(self):
        rows=[dict(TTFT_PASS=True,ITL_PASS=True,TPOT_PASS=True,output_tokens=5),
            dict(TTFT_PASS=False,ITL_PASS=True,TPOT_PASS=True,output_tokens=30),
            dict(TTFT_PASS=False,ITL_PASS=False,TPOT_PASS=True,output_tokens=10)]
        counterfactual,attribution=failure_accounting(rows,5.)
        actual=next(r for r in counterfactual if r['removed_gates']=='NONE')
        ttft=next(r for r in counterfactual if r['removed_gates']=='TTFT_PASS')
        self.assertEqual(actual['qualified_tokens'],5)
        self.assertEqual(ttft['qualified_tokens'],35)
        self.assertEqual(len(counterfactual),8)
        self.assertAlmostEqual(sum(r['lost_output_tokens'] for r in attribution),40.)
        self.assertAlmostEqual(sum(r['lost_requests'] for r in attribution),2.)
        self.assertIn('Not runtime causality',attribution[0]['scope'])

    def test_wait_removal_does_not_repair_decode_failure(self):
        rows=[dict(TTFT_ms=6000,D_coordinator_gate_wait=2000,ITL_PASS=False,TPOT_PASS=True,output_tokens=100),
            dict(TTFT_ms=6000,D_coordinator_gate_wait=2000,ITL_PASS=True,TPOT_PASS=True,output_tokens=10)]
        result=zero_gate_oracle(rows,10.)
        self.assertEqual(result['evidence'],'OFFLINE_ORACLE')
        self.assertEqual(result['observed_SLO_goodput'],0.)
        self.assertEqual(result['zero_D_coordinator_wait_SLO_goodput'],1.)
        self.assertEqual(result['zero_D_coordinator_wait_TTFT_p95_ms'],4000.)
        self.assertIn('Not achievable speedup',result['assumptions'])

    def test_impossible_wait_is_rejected(self):
        with self.assertRaises(AssertionError):
            zero_gate_oracle([dict(TTFT_ms=1,D_coordinator_gate_wait=2,ITL_PASS=True,TPOT_PASS=True,output_tokens=1)],1.)


if __name__=='__main__':unittest.main()
