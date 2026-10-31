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
    def test_runtime_hash_scope(self):
        from .dispatch import runtime_sources
        names={p.name for p in runtime_sources()}
        self.assertIn('serving_worker.py',names)
        self.assertIn('capacity_worker.py',names)
        self.assertNotIn('additional_analysis.py',names)
    def test_dispatcher_failure_without_raw_status(self):
        import tempfile,json
        from pathlib import Path
        from .campaign import namespace_state
        with tempfile.TemporaryDirectory(prefix='bc_failed_start_') as d:
            path=Path(d);(path/'dispatcher_result.json').write_text(json.dumps(dict(returncode=1)))
            self.assertEqual(namespace_state(path),'FAILED')
            self.assertFalse((path/'status.json').exists())
    def test_prepared_gate_has_no_configuration_IO(self):
        import ast
        from .common import SOURCE
        tree=ast.parse((SOURCE/'serving_worker.py').read_text())
        gate=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='ready_cohort')
        calls=[n for n in ast.walk(gate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='config']
        self.assertFalse(calls,'The per-request scheduling gate must not read configuration files')

if __name__=='__main__':unittest.main()
