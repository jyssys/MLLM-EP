"""Keep point selection separate from quality and future confirmation gates."""
import copy
import unittest

from .token_discovery_control import choose


class TokenDiscoveryTest(unittest.TestCase):
    def test_lower_p95_with_decode_or_p50_regression_does_not_replace_safe_candidate(self):
        baseline=dict(TTFT_p50_ms=100,TTFT_p95_ms=200,ITL_p99_ms=10,TPOT_p95_ms=9,SLO_goodput=50,
            gate_attainment=dict(ITL_PASS=1.,TPOT_PASS=1.))
        rows=[dict(method='Native',fraction=0.,mode='replay',metrics=baseline)]
        for fraction,p95,goodput,changes in [(.1,180,60,{}),(.2,170,70,dict(ITL_p99_ms=10.6)),
            (.25,160,90,dict(TTFT_p50_ms=110)),(.3,175,65,{}),(.5,190,80,{})]:
            value=copy.deepcopy(baseline);value.update(changes,TTFT_p95_ms=p95,SLO_goodput=goodput)
            rows.append(dict(method='Static Token',fraction=fraction,mode='replay',metrics=value))
        selected=choose(rows)
        self.assertEqual(selected['fraction'],.3)
        self.assertEqual(selected['best_goodput_fraction'],.5)
        self.assertEqual(selected['quality'],'PENDING')
        self.assertEqual(selected['confidence_interval'],'NOT_AVAILABLE')


if __name__=='__main__':unittest.main()
