"""CPU checks for context, indivisible Prompt budgets, and Poisson traces."""
import unittest
from .common import *
from .configs import finish


class ContractTests(unittest.TestCase):
    def test_context_limits(self):
        for m,cells in GRID.items():
            for length,cap in cells:self.assertLessEqual(length+cap,CONTEXT_LIMITS[m])
    def test_full_prompt_budget(self):
        for m,cells in GRID.items():
            for length,cap in cells:
                c=base(m,'cpu_only',length,cap)
                self.assertGreaterEqual(c['token_budget'],length)
                self.assertEqual(c['token_budget']%length,0)
    def test_actual_poisson_offsets_change_with_rate(self):
        a=finish(base('qwen8','cpu_only',8000,200),count=8,rate=1.)
        b=finish(base('qwen8','cpu_only',8000,200),count=8,rate=2.)
        self.assertEqual(a['prompt_ids'],b['prompt_ids'])
        self.assertEqual(a['arrival_offsets'],[2*v for v in b['arrival_offsets']])
        self.assertTrue(all(a['arrival_offsets'][i]<a['arrival_offsets'][i+1] for i in range(7)))
    def test_full_is_not_a_wire_fragment_count(self):
        c=finish(base('huginn','cpu_only',3900,32),count=2)
        self.assertTrue(c['atomic_prompt']);self.assertEqual(c['depth'],32)
    def test_native_scheduler_updates_progress_before_return(self):
        from types import SimpleNamespace
        from .ar_worker import prompt_rows
        request=SimpleNamespace(prompt_token_ids=list(range(512)),num_computed_tokens=0)
        before={'request':request.num_computed_tokens}
        request.num_computed_tokens+=512
        result=SimpleNamespace(num_scheduled_tokens={'request':512})
        self.assertEqual(prompt_rows(result,{'request':request},before),[('request',0,512,512)])
    def test_atomic_evidence_excludes_warmup(self):
        from .evidence import atomic
        p=ROOT/'raw/gate_ouro_native_full_l512_pd_g1_v1/window.json'
        if not p.exists():self.skipTest('GPU evidence fixture is not present')
        evidence=atomic('gate_ouro_native_full_l512_pd_g1_v1')
        self.assertTrue(evidence['passed'],evidence)
        self.assertEqual(evidence['recorded_prompt_spans'],2)
    def test_local_stage_metrics_have_the_run_namespace(self):
        from unittest.mock import patch
        from .local_metrics import prepare
        with patch.dict(os.environ,{},clear=True):
            prepare({'name':'cpu_test'})
            self.assertEqual(os.environ['ABC_RUN_DIR'],str(ROOT/'raw/cpu_test'))
    def test_zero_arrival_span_profile_uses_the_tool_window(self):
        name='profile_tool_smoke_ouro_l512_v1';folder=ROOT/'raw'/name
        if not (folder/'nsys_report.sqlite').exists():self.skipTest('Recorded GPU profiler fixture is not present')
        from .profiles import parse
        value=parse(folder,json.loads((ROOT/'configs'/(name+'.json')).read_text()))
        self.assertTrue(all(r['kernel_count']>0 for r in value['devices']),value)


if __name__=='__main__':unittest.main()
