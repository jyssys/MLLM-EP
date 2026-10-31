"""CPU tests for the measurement definitions."""
import unittest
from .summary_tables import percentile,wilson
from .analyze_serving import token_intervals
from .prepare import GRID
from .common import START_EPOCH,remaining

class ContractTests(unittest.TestCase):
    def test_common_context(self):
        self.assertTrue(all(length+cap<=4096 for length,cap in GRID.values()))
    def test_percentile(self):
        self.assertEqual(percentile([1,2,3,4],.5),2.5)
        self.assertIsNone(percentile([],.99))
    def test_request_delivery_intervals(self):
        row=dict(token_events=[dict(first=True,time=1.,previous_time=0.,new_tokens=1),
            dict(first=False,time=1.04,previous_time=1.,new_tokens=1),
            dict(first=False,time=1.10,previous_time=1.04,new_tokens=2)])
        intervals,bundled=token_intervals(row)
        self.assertAlmostEqual(intervals[0],40.);self.assertEqual(bundled,1)
        self.assertEqual(len(intervals),1)
    def test_AR_intervals(self):
        values,n=token_intervals(dict(token_events=[dict(time=1.,count=1),dict(time=1.05,count=2)]))
        self.assertAlmostEqual(values[0],50.);self.assertEqual(n,0)
    def test_wilson(self):
        a,b=wilson(24,24)
        self.assertGreater(a,.8);self.assertAlmostEqual(b,1.)
        self.assertEqual(wilson(0,0),[None,None])
    def test_64_request_prompt_ids(self):
        from .phase_b_configs import pd_config
        cfg=pd_config('ouro','cpu_test_only',count=64)
        self.assertEqual(len(cfg['prompt_ids']),64)
        self.assertEqual(cfg['dataset'],'test64_L3072.json')

if __name__=='__main__':unittest.main()
