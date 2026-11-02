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
    def test_explicit_dataset_pool_preserves_recheck_inputs(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from . import configs
        with TemporaryDirectory(prefix='atomic_dataset_recheck_') as folder:
            root=Path(folder);(root/'data').mkdir()
            write(root/'data/qualified_subset.json',[dict(prompt_id='qualified')])
            write(root/'data/test_qwen8_L8000.json',[dict(prompt_id='different_default')])
            cfg=base('qwen8','cpu_only',8000,200,dataset_pool='qualified_subset.json')
            with patch.object(configs,'ROOT',root):value=configs.finish(cfg,1)
        self.assertEqual(value['dataset'],'qualified_subset.json')
        self.assertEqual(value['prompt_ids'],['qualified'])
    def test_output_comparison_identifies_different_inputs(self):
        from unittest.mock import patch
        from . import evidence
        a=[dict(sample_id=0,input_ids=[1],output_ids=[2],seed=3)]
        b=[dict(sample_id=0,input_ids=[9],output_ids=[2],seed=3)]
        with patch.object(evidence,'rows',side_effect=[a,b]):value=evidence.compare('a','b')
        self.assertFalse(value['comparison_applicable']);self.assertEqual(value['matched_input_seed_rows'],0)
        self.assertFalse(value['passed'])
    def test_report_paragraphs_preserve_literal_blocks_and_values(self):
        from .reporting import prose_blocks,table
        source='One 95% CI.\nTwo 32/32.\nThree 128 requests.\nFour 1.2x.\n\n```bash\na\nb\nc\nd\n```\n'
        result=prose_blocks(source)
        self.assertIn('Three 128 requests.\n\nFour 1.2x.',result)
        self.assertIn('```bash\na\nb\nc\nd\n```',result)
        self.assertTrue(table(['Value'],[[.25]]).startswith('\n|'))
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
    def test_performance_family_cannot_silently_disable_BI(self):
        from .families import validate_runtime_family
        c=base('qwen8','cpu_only',8000,200,ar_execution_family='BI_EAGER',ar_enforce_eager=True,ar_batch_invariant=False)
        with self.assertRaises(AssertionError):validate_runtime_family(c)
        c['ar_batch_invariant']=True;validate_runtime_family(c)
        c.update(ar_execution_family='BI_COMPILED_NO_GRAPH',ar_enforce_eager=False,ar_cudagraph_mode='NONE')
        validate_runtime_family(c)
    def test_AR_capacity_batch_does_not_reduce_qualified_engine_request_capacity(self):
        from .ar_compiled_recovery import serving_P_slots
        self.assertEqual(serving_P_slots(1,4),4)
        self.assertEqual(serving_P_slots(8,4),8)
    def test_prepared_D_environment_preserves_the_requested_AR_arithmetic(self):
        from .families import ar_environment
        self.assertEqual(ar_environment({'ar_batch_invariant':True})['VLLM_BATCH_INVARIANT'],'1')
        self.assertEqual(ar_environment({'ar_batch_invariant':False})['VLLM_BATCH_INVARIANT'],'0')
    def test_P_only_profiler_failure_does_not_block_clean_serving(self):
        from unittest.mock import patch
        from . import live_audit
        cfg=dict(model='qwen8',name='cpu_test_missing_profile',phase='profile_control',
            nsys=False,nsys_P_only=True,ar_execution_family='BI_COMPILED_NO_GRAPH',length=8000)
        with patch('poc_flashvep.looped_pd_atomic_prefill.h_workspace_controls.advance'),patch.object(live_audit,'status',return_value='FAILED'),patch.object(live_audit,'write') as save:
            live_audit.verify(cfg)
        self.assertEqual(save.call_count,0)
    def test_process_observer_tolerates_a_process_exit_during_proc_read(self):
        from unittest.mock import patch,MagicMock
        from . import guard
        vanished=MagicMock();vanished.name='1234'
        vanished.__truediv__.return_value.read_bytes.side_effect=ProcessLookupError(3,'No such process')
        with patch.object(guard,'Path') as root:
            root.return_value.iterdir.return_value=[vanished]
            self.assertEqual(guard.processes('campaign'),[])
    def test_profile_gate_keeps_passes_and_one_negative_control_per_chunk(self):
        from .profile_prerequisite_gate import retained_profiles
        entries=[dict(name=f'c{c}s{s}',chunk=c,state='PENDING',near_P_service=False,P_completion_rate=float(s+1)) for c in (1,2) for s in range(3)]
        self.assertEqual(retained_profiles(entries),{'c1s2','c2s2'})
        entries[0].update(state='COMPLETE');entries[1].update(near_P_service=True)
        self.assertEqual(retained_profiles(entries),{'c1s0','c1s1','c2s2'})
    def test_profile_prerequisite_uses_measured_P_completions_and_excludes_warmup(self):
        from .profile_prerequisite_gate import completion_rate
        metrics=[dict(request_id=n,created_monotonic=0.,timings=dict(prefill_compute_completed=t)) for n,t in [('r0_q0',2.),('r0_q1',7.),('r0_q2',13.),('r-1_q0',5.)]]
        self.assertEqual(completion_rate(metrics,2.,10.),.25)
    def test_profile_no_progress_watchdog_does_not_stop_clean_or_completed_work(self):
        from .dispatch import profile_stalled
        self.assertTrue(profile_stalled(dict(nsys=True),'RUNNING',181.))
        self.assertTrue(profile_stalled(dict(nsys_P_only=True),'RUNNING',181.))
        self.assertFalse(profile_stalled(dict(nsys=True),'COMPLETE',181.))
        self.assertFalse(profile_stalled({},'RUNNING',181.))
        self.assertFalse(profile_stalled(dict(nsys=True),'RUNNING',None))
    def test_unfinished_window_cannot_report_a_completed_subset_rate_as_goodput(self):
        from .analyze import mask_incomplete_rates
        row=dict(elapsed_s=10.,requests_per_s=3.,SLO_goodput=1.,SLO_attainment=.25)
        mask_incomplete_rates(row)
        self.assertIsNone(row['SLO_goodput']);self.assertIsNone(row['requests_per_s'])
        self.assertEqual(row['SLO_attainment'],.25)
        self.assertEqual(row['completed_subset_elapsed_s_diagnostic'],10.)
    def test_requested_Huginn_async_is_not_reported_as_effective_async(self):
        from .families import effective_asynchronous,execution_family
        cfg=base('huginn','cpu_only',3500,32,asynchronous=True,execution_control_family='requested_async')
        self.assertFalse(effective_asynchronous(cfg))
        self.assertTrue(execution_family(cfg).startswith('IGNORED_ASYNC_REQUEST_'))
    def test_extension_final_cohorts_keep_the_original_size_and_duration(self):
        from .extension_confirmation import final_count
        import random
        for rate in (.5,2.5,10.):
            for start in range(3):
                count=final_count(rate,start);rng=random.Random(20261131+start)
                self.assertGreaterEqual(count,128)
                self.assertGreaterEqual(.7*sum(rng.expovariate(rate) for _ in range(count-1)),30.)
    def test_deadline_queue_preserves_the_looped_priority(self):
        from .extension_discovery import queue_priority
        loop=base('huginn','cpu_only',3500,32,extension=True,phase='qps_pilot')
        ar=base('qwen8','cpu_only',8000,200,extension=True,phase='qps_pilot',ar_execution_family='BI_COMPILED_NO_GRAPH')
        self.assertLess(queue_priority(loop),queue_priority(ar))
        center=dict(loop,phase='final_repeat',rate_index=1)
        outer=dict(loop,phase='final_repeat',rate_index=2)
        self.assertLess(queue_priority(center),queue_priority(ar))
        self.assertLess(queue_priority(ar),queue_priority(outer))
    def test_reserved_context_time_splits_into_pending_and_active(self):
        from .reservation_accounting import occupancy
        reserved=occupancy([(0.,4.),(1.,5.)],1.,4.)
        pending=occupancy([(0.,2.),(1.,3.)],1.,4.)
        active=occupancy([(2.,4.),(3.,5.)],1.,4.)
        self.assertEqual(reserved['mean'],2.)
        self.assertEqual(reserved['peak'],2)
        self.assertAlmostEqual(pending['request_seconds']+active['request_seconds'],reserved['request_seconds'])
        self.assertAlmostEqual(active['zero_fraction'],1./3)
    def test_KV_density_separates_valid_layers_and_uniform_storage(self):
        from .architecture_accounting import density
        q=density('qwen8',dict(num_hidden_layers=36,num_key_value_heads=8,head_dim=128),1)
        o=density('ouro',dict(num_hidden_layers=24,num_key_value_heads=16,head_dim=128),4)
        h=density('huginn',dict(n_layers=8,n_layers_in_prelude=2,n_layers_in_coda=2,n_layers_in_recurrent_block=4,num_key_value_heads=55,head_dim=96),32)
        self.assertEqual(q['fully_valid_KV_bytes_per_token'],147456)
        self.assertEqual(o['fully_valid_KV_bytes_per_token'],786432)
        self.assertEqual(h['fully_valid_KV_bytes_per_token'],2787840)
        self.assertEqual(h['uniform_reserved_KV_bytes_per_token'],5406720)
        self.assertEqual(h['unused_uniform_plane_fraction'],.484375)
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
    def test_failed_old_AR_family_does_not_relabel_a_qualified_new_family(self):
        from unittest.mock import patch
        from . import reporting
        runs=[dict(name='new_family_pilot',model='qwen8',phase='qps_pilot',eligible_clean=True)]
        blocks={'qwen8':dict(classification='RUNTIME_FAILURE_REQUIRES_TRIAGE')}
        with patch.object(reporting,'write'):
            result=reporting.model_decisions(runs,[],blocks)
        self.assertEqual(next(r for r in result if r['model']=='qwen8')['verdict'],'NOT_RUN_FINAL_CONFIRMATION')
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
    def test_publication_batches_bound_each_upload_without_dropping_files(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from . import publish
        with TemporaryDirectory(prefix='atomic_publication_test_') as directory:
            root=Path(directory)
            for name in ('one','two','three'):(root/name).write_bytes(b'x'*40)
            with patch.object(publish,'REPO',root):
                self.assertEqual(publish.file_batches(['one','two','three'],80),[['one','two'],['three']])


if __name__=='__main__':unittest.main()
