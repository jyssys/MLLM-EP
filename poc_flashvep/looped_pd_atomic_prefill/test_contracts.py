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
    def test_colocated_worker_does_not_require_a_nixl_fingerprint(self):
        from .native_mixed_worker import worker_mapping
        mapping=worker_mapping([{'cuda_uuid':UUIDS[2]},{'cuda_uuid':UUIDS[3]}],[2,3])
        self.assertEqual([r['physical_GPU'] for r in mapping],[2,3])
    def test_capacity_reader_skips_reference_metadata(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from . import analyze
        with TemporaryDirectory(prefix='atomic_capacity_test_') as folder:
            root=Path(folder);(root/'configs').mkdir()
            write(root/'configs/capacity_references.json',{'ouro_L8192':{'mu_P':2.}})
            with patch.object(analyze,'ROOT',root):
                self.assertEqual(analyze.capacity(),[])
    def test_ar_async_is_an_explicit_separate_option(self):
        from .ar_worker import llm_options
        c=base('qwen8','cpu_test',8000,200)
        self.assertFalse(llm_options(c,'prefill')['async_scheduling'])
        c['ar_async_scheduling']=True
        self.assertTrue(llm_options(c,'prefill')['async_scheduling'])
    def test_ar_eager_is_an_explicit_separate_option(self):
        from .ar_worker import llm_options
        cfg=base('qwen8','cpu_only',8000,200)
        self.assertNotIn('enforce_eager',llm_options(cfg,'prefill'))
        cfg['ar_enforce_eager']=True
        self.assertTrue(llm_options(cfg,'prefill')['enforce_eager'])
    def test_configured_base_does_not_disable_an_explicit_AR_invariance_request(self):
        from .search import configured_base
        c=configured_base('qwen8','cpu_only',8000,200,ar_batch_invariant=True)
        self.assertTrue(c['ar_batch_invariant'])
    def test_colocated_control_preserves_actual_scheduler_geometry(self):
        from .colocated_controls import controls
        cfg=base('huginn','cpu_only',3500,32,wavefront=False,token_budget=2048,kv_blocks=28288,scheduler_mode='mixed',native_mixed=True)
        self.assertEqual(controls(cfg),dict(wavefront=False,token_budget=2048,kv_blocks=28288,scheduler_mode='mixed',native_mixed=True))
    def test_final_TRUE_verdict_needs_three_independent_confirmations(self):
        from unittest.mock import patch
        from . import reporting
        runs=[dict(name='a'+str(i),model='ouro',phase='final_repeat',eligible_clean=True,independent_start=i,transport_control='same',target_QPS=1.) for i in range(3)]
        checked=[dict(name='a'+str(i),model='ouro',length=8192,output_cap=32,chunk=8192,category='TRUE_PREFILL_HEAVY') for i in range(3)]
        with patch.object(reporting,'write'):
            two=reporting.model_decisions(runs[:2],checked[:2],{})
            three=reporting.model_decisions(runs,checked,{})
        self.assertFalse(next(r for r in two if r['model']=='ouro')['repeated_TRUE_cells'])
        self.assertTrue(next(r for r in three if r['model']=='ouro')['repeated_TRUE_cells'])
    def test_final_cohort_has_thirty_seconds_in_its_central_window(self):
        from .search import final_count
        import random
        for rate in (.5,2.5,10.):
            for start in range(3):
                count=final_count(rate,start)
                rng=random.Random(20261111+start)
                duration=sum(rng.expovariate(rate) for _ in range(count-1))
                self.assertGreaterEqual(count,128)
                self.assertGreaterEqual(.7*duration,30.)
    def test_large_evidence_archive_is_lossless(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from . import archive
        with TemporaryDirectory(prefix='atomic_archive_test_') as directory:
            root=Path(directory);source=root/'sample.bin';source.write_bytes(bytes(range(256))*10)
            with patch.object(archive,'ROOT',root),patch.object(archive,'MANIFEST',root/'archives/archive_manifest.json'),patch.object(archive,'PART_BYTES',64):
                archive.create(minimum_bytes=1000)
                target=root/'restored.bin';archive.restore('sample.bin',target)
                self.assertEqual(source.read_bytes(),target.read_bytes())


if __name__=='__main__':unittest.main()
