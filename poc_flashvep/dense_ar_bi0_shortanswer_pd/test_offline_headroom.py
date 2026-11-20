import unittest
from .offline_headroom import fixed_trace,extend,KV_contract_accounting


class OfflineHeadroomTests(unittest.TestCase):
    def test_incremental_transition_matches_full_credit_replay(self):
        arrivals=[0.,.1,.2,.3,.4,.5];costs=[{'x':(.7,.3,.4)} for _ in arrivals]
        full=fixed_trace(arrivals,costs,['x']*len(arrivals),credit=2);incremental=[]
        for i,a in enumerate(arrivals):incremental.append(extend(incremental,i,a,costs[i]['x'],credit=2))
        for a,b in zip(full,incremental):
            for key in ('P_start','P_finish','D_start','D_first','D_finish'):self.assertAlmostEqual(a[key],b[key])

    def test_KV_ownership_restores_under_fixed_credits(self):
        arrivals=[0.,.1,.2,.3,.4,.5];costs=[{'x':(.7,.3,.4)} for _ in arrivals]
        states=fixed_trace(arrivals,costs,['x']*len(arrivals),credit=4)
        audit=KV_contract_accounting(states,[dict(input_tokens=38912)]*len(arrivals),[152]*len(arrivals))
        self.assertTrue(audit['credits_restored'])
        self.assertLessEqual(audit['maximum_blocks']['source'],4*305)
        self.assertEqual(audit['maximum_blocks']['destination'],306)

    def test_KV_overcommit_is_not_accepted(self):
        state=dict(P_start=0.,D_start=1.,D_first=2.,D_finish=3.)
        with self.assertRaises(AssertionError):
            KV_contract_accounting([state],[dict(input_tokens=1000000)],[16])
