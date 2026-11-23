"""CPU tests for observer passthrough and safe source-credit transitions."""
import ast
from pathlib import Path
import textwrap
import unittest
from .instrumentation import rewrite
from .retirement import safe_completion,release_owned


class ProtocolTests(unittest.TestCase):
    def test_GPU_budget_uses_interval_union_not_overlapping_sum(self):
        from .verification import interval_union_seconds
        self.assertEqual(interval_union_seconds([]),0)
        self.assertEqual(interval_union_seconds([(1,3),(2,4),(5,6)]),4)
        self.assertEqual(interval_union_seconds([(1,4),(2,3),(4,5)]),4)
        with self.assertRaises(AssertionError):interval_union_seconds([(3,2)])

    def test_report_separates_warm_strict_proof_from_serving_results(self):
        from unittest.mock import patch
        from . import report_sections
        from .reports import table,estimate,number
        fixtures={
            'analysis/coordinator_hygiene_warm_regime_confirmation_analysis.json':None,
            'analysis/compiler_cache_regime_causal_audit.json':dict(strict_comparisons=[],prospective_warm_regime=dict(passed=True)),
            'configs/common_warm_regime_qualification_v9.json':dict(items=[dict(label='Fixture',gate='fixture.json')]),
            'state/fixture.json':dict(passed=True,full400_passed=True,quality_passed=True),
        }
        def read(path,default=None):return fixtures.get(path,default)
        with patch.object(report_sections,'load',side_effect=read):
            result=report_sections.sections(table,estimate,number)
        self.assertIn('새 Common Warm C3 확인',result['answers'])
        self.assertIn('Warm-only3회',result['answers'])
        self.assertIn('CORRECTNESS_ONLY',result['answers'])

    def test_cache_prime_identity_preserves_pair_source_and_model_action(self):
        import json
        from .common import ROOT
        from .warm_serving_followup import priming_identity,serving_prime_template
        for label in ('NativeStrong','NativeCommon2','StockToken','Layer','Layer8','Hybrid','HybridIso8Approx','HybridGCControl'):
            prime=json.loads((ROOT/'configs'/f'PC_WarmQual_{label}_Prime8.json').read_text())
            full=json.loads((ROOT/'configs'/f'PC_WarmQual_{label}_Natural400.json').read_text())
            if label=='StockToken':
                self.assertTrue(prime['controlled_pipeline']);self.assertFalse(full['controlled_pipeline'])
                self.assertNotEqual(priming_identity(prime),priming_identity(full))
            prime=serving_prime_template(label,full['gpus'],'fixture_'+label)
            self.assertEqual(priming_identity(prime),priming_identity(full),label)
            other=dict(full,gpus=[4,5]);self.assertNotEqual(priming_identity(prime),priming_identity(other))
            other=dict(full,source_capsule_SHA256='different');self.assertNotEqual(priming_identity(prime),priming_identity(other))
            other=dict(full,axis_cut=999);self.assertNotEqual(priming_identity(prime),priming_identity(other))
            annotated=dict(full,cache_priming_reference='only-a-provenance-pointer')
            self.assertEqual(priming_identity(full),priming_identity(annotated))

    def test_expected_roots_accepts_full_and_partial_caches_in_one_worker(self):
        import tempfile
        from unittest.mock import patch
        from . import warm_serving_followup
        log=('Using cache directory: /cache/torch_compile_cache/full/backbone for vLLM\n'
            'Using cache directory: /cache/prefix_compiler_v3/lower34/backbone for vLLM\n')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);folder=root/'raw/fixture';folder.mkdir(parents=True)
            for role in ('prefill','decode'):(folder/(role+'.log')).write_text(log)
            with patch.object(warm_serving_followup,'ROOT',root):
                result=warm_serving_followup.expected_roots('fixture')
            self.assertEqual(len(result['prefill']),2)
            self.assertEqual(len(result['decode']),2)

    def test_warm_partial_reader_tracks_each_real_compiler_region(self):
        from .warm_serving_followup import compiler_regions
        official='/cache/torch_compile_cache/full/backbone'
        partial='/cache/prefix_compiler_v3/lower34/backbone'
        log=(f'Using cache directory: {official} for vLLM\n'
            'Directly load the compiled graph(s) for compile range (1,8192)\n'
            'Directly load AOT compilation from path /cache/aot/model\n'
            f'Using cache directory: {partial} for vLLM\n'
            'Directly load the compiled graph(s) for compile range (1,8192)\n')
        regions=compiler_regions(log)
        self.assertEqual([row['cache_root'] for row in regions],[official,partial])
        self.assertTrue(all(row['graph_reused'] for row in regions))
        self.assertTrue(regions[0]['official_AOT_required'])
        self.assertFalse(regions[1]['official_AOT_required'])
        cold=compiler_regions(log.replace('Directly load the compiled graph(s)','Compiling a graph for compile range',1))
        self.assertTrue(cold[0]['new_compilation']);self.assertFalse(cold[0]['graph_reused'])

    def test_warm_regime_rejects_cold_compile_or_missing_AOT_loading(self):
        from .warm_cache_strict import cache_reuse_observation
        expected='/cache/fixture/backbone'
        good=('Using cache directory: '+expected+" for vLLM's torch.compile\n"
            'Directly load the compiled graph(s) for compile range (1,8192)\n'
            'Directly load AOT compilation from path /cache/aot/model\n')
        self.assertTrue(cache_reuse_observation(good,expected)['official_graph_reuse'])
        with self.assertRaises(AssertionError):cache_reuse_observation(good+'Compiling a graph for compile range (1,8192)',expected)
        with self.assertRaises(AssertionError):cache_reuse_observation(good.replace('Directly load AOT compilation','Unobserved AOT compilation'),expected)
        with self.assertRaises(AssertionError):cache_reuse_observation(good,'/wrong/cache')

    def test_fixed_port_strict_failure_blocks_performance_without_GPU_calls(self):
        import json
        import tempfile
        from unittest.mock import patch,MagicMock
        from . import hygiene_confirmation
        from .common import write
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'state').mkdir()
            for label in hygiene_confirmation.MAPPING.values():
                write(root/'state'/('pc_fixed_port_'+label.lower()+'_composition_v6.json'),
                    dict(passed=label!='StockToken',signature='fixture'))
            runner=MagicMock()
            with patch.object(hygiene_confirmation,'ROOT',root),patch.object(hygiene_confirmation,'FIXED_PORTS',True), \
                    patch.object(hygiene_confirmation,'bind',return_value=runner):
                self.assertIsNone(hygiene_confirmation.prepare_after_gates())
            runner.execute.assert_not_called();runner.save.assert_not_called()
            marker=json.loads((root/'state/coordinator_hygiene_fixed_ports_not_run_v6.json').read_text())
            self.assertEqual(marker['status'],'NOT_RUN');self.assertIn('StockToken',marker['invalid'])
            self.assertFalse((root/'jobs').exists())

    def test_native_reader_supplies_the_real_required_deflection_argument(self):
        from unittest.mock import patch
        from .port_qualification import native_state
        from ..dense_ar_longbench400_pd import decode_cap_discovery
        with patch.object(decode_cap_discovery,'full_native_state',autospec=True,return_value=dict(passed=True)) as reader:
            self.assertTrue(native_state('completed-fixture')['passed'])
            reader.assert_called_once_with('completed-fixture',False)

    def test_bind_repairs_a_reader_imported_before_the_campaign_root(self):
        from .common import ROOT,LEGACY,bind
        from ..dense_ar_longbench400_pd import qualification
        previous=qualification.ROOT
        try:
            qualification.ROOT=LEGACY
            bind()
            self.assertEqual(qualification.audit.__globals__['ROOT'],ROOT)
        finally:qualification.ROOT=previous

    def test_pinned_control_port_never_falls_back_after_an_address_conflict(self):
        from unittest.mock import patch,MagicMock
        from . import pinned_control_ports
        cfg=dict(gpus=[2,3],fixed_NIXL_control_ports={'2,3':[16082,16083]})
        probe=MagicMock();probe.__enter__.return_value=probe
        with patch.object(pinned_control_ports.Path,'read_text',return_value='32768 60999'), \
                patch.object(pinned_control_ports.socket,'socket',return_value=probe):
            self.assertEqual(pinned_control_ports.choose(cfg,'prefill'),16082)
            probe.bind.assert_called_with(('127.0.0.1',16082))
            probe.bind.side_effect=OSError('occupied')
            with self.assertRaises(OSError):pinned_control_ports.choose(cfg,'decode')
            with self.assertRaises(AssertionError):pinned_control_ports.choose(cfg,'prefill',[16082])

    def test_figure_safety_marks_observed_failure_without_inventing_N1_relative_failure(self):
        from .figures import safety_unqualified
        row=dict(independent_starts=1,quality_passed_all_starts=True,
            decode_gate=dict(qualified=False,absolute_decode_attainment_95percent_pass=True))
        self.assertFalse(safety_unqualified(row))
        row['runtime_policy_passed_all_starts']=False
        self.assertTrue(safety_unqualified(row))
        row['runtime_policy_passed_all_starts']=True
        row['decode_gate']['absolute_decode_attainment_95percent_pass']=False
        self.assertTrue(safety_unqualified(row))
        row.update(independent_starts=3,decode_gate=dict(qualified=True,absolute_decode_attainment_95percent_pass=True),
            additional_decode_gates=dict(NativeCommon4=dict(qualified=False)))
        self.assertTrue(safety_unqualified(row))

    def test_cached_head_transfer_rewrites_only_new_or_partial_KV_pages(self):
        from .attention_benchmark import updated_KV_blocks
        self.assertEqual(updated_KV_blocks(0,8192,128),(0,64))
        self.assertEqual(updated_KV_blocks(8192,16384,128),(64,128))
        self.assertEqual(updated_KV_blocks(16384,16477,128),(128,129))
        self.assertEqual(updated_KV_blocks(8181,8290,128),(63,65))
        with self.assertRaises(AssertionError):updated_KV_blocks(100,99,128)

    def test_attention_head_split_preserves_complete_GQA_groups(self):
        from .attention_benchmark import groups
        import torch
        query=torch.arange(5*32*128).view(5,32,128)
        for heads in (2,4):
            value=groups(heads)
            self.assertEqual(value['local_Q']+value['remote_Q'],32)
            self.assertEqual(value['local_KV']+value['remote_KV'],8)
            self.assertEqual(value['local_Q'],4*value['local_KV'])
            self.assertEqual(value['remote_Q'],4*value['remote_KV'])
            joined=torch.cat([query[:,:value['local_Q']],query[:,value['local_Q']:]],dim=1)
            self.assertTrue(torch.equal(joined,query))
        with self.assertRaises(AssertionError):groups(3)

    def test_stable_source_reuses_identical_bytes_and_rejects_corruption(self):
        import tempfile
        import zipfile
        from unittest.mock import patch
        from . import stable_source
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);archive=root/'source.zip'
            with zipfile.ZipFile(archive,'w') as stream:stream.writestr('poc_flashvep/test.py','value=1\n')
            with patch.object(stable_source,'ROOT',root):
                first=stable_source.source_tree(archive)
                self.assertEqual(stable_source.source_tree(archive),first)
                (first/'poc_flashvep/test.py').write_text('value=2\n')
                with self.assertRaises(AssertionError):stable_source.source_tree(archive)

    def test_distributed_pair_comparator_reads_both_axis_state_schemas(self):
        import json
        import tempfile
        import torch
        from unittest.mock import patch
        from . import coordinator_state_fix
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'raw').mkdir()
            for label in ('control','candidate'):
                folder=root/'raw'/label;folder.mkdir()
                tensor=folder/'state.pt';torch.save(dict(hidden=torch.tensor([1.,2.])),tensor)
                state=dict(KV=[dict(layer=0,bytes=4,SHA256='same')],tensor_file=str(tensor))
                (folder/'axis_events_decode.json').write_text(json.dumps(dict(states=dict(q0=state))))
                (folder/'requests.jsonl').write_text(json.dumps(dict(sample_id=0,output_ids=[1,2]))+'\n')
            with patch.object(coordinator_state_fix,'ROOT',root):
                result=coordinator_state_fix.compare('control','candidate')
                self.assertTrue(result['passed'])
                path=root/'raw/candidate/axis_events_decode.json'
                value=json.loads(path.read_text());value['states']['q0']['KV'][0]['SHA256']='different'
                path.write_text(json.dumps(value))
                self.assertFalse(coordinator_state_fix.compare('control','candidate')['passed'])

    def test_native_state_capture_uses_the_native_worker_seam(self):
        from .native_capture_fix import capture_flags
        flags=capture_flags(dict(method='Native',axis_state_capture=True,state_capture=False))
        self.assertTrue(flags['state_capture'])
        self.assertFalse(flags['axis_state_capture'])

    def test_real_capture_seam_accepts_backend_global_window_sentinel(self):
        import ast
        import re
        from types import SimpleNamespace
        from . import attention_capture
        class Tensor:
            def __init__(self,shape):self.shape=shape
            def __getitem__(self,key):return self
            def long(self):return self
            def detach(self):return self
            def cpu(self):return self
            def index_select(self,*args):return self
            def stride(self):return tuple(range(len(self.shape)))
        source=Path(attention_capture.__file__).read_text()
        node=next(node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef) and node.name=='observed')
        captures=[];namespace=dict(re=re,cfg=dict(layers=[0]),active=dict(prompt_id='real-window-seam',input_tokens=64),
            captures=captures,original=lambda *a,**kw:a[6],torch=SimpleNamespace(save=lambda *args:None),
            folder=Path('/attention_poc/raw/not_written_capture_fixture'),ROOT=Path('/'),digest=lambda path:'fixture')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<real_capture_seam>','exec'),namespace)
        metadata=SimpleNamespace(num_actual_tokens=64,seq_lens=[SimpleNamespace(item=lambda:64)],
            query_start_loc=SimpleNamespace(numel=lambda:2),block_table=Tensor((1,1)),causal=True,
            max_num_splits=0,scheduler_metadata=None)
        backend=SimpleNamespace(scale=.125,vllm_flash_attn_version=3,sliding_window=(-1,-1),
            logits_soft_cap=0,alibi_slopes=None,sinks=None)
        namespace['observed'](backend,SimpleNamespace(layer_name='model.layers.0.self_attn'),
            Tensor((64,32,128)),None,None,Tensor((1,8,128,256)),metadata,Tensor((64,32,128)))
        self.assertEqual(len(captures),1)
        self.assertEqual(captures[0]['sliding_window'],[-1,-1])

    def test_guard_recovery_never_restarts_a_live_or_failed_GPU_folder(self):
        import json
        import tempfile
        from unittest.mock import patch
        from . import supervision
        from .common import write
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for sub in ('jobs','raw','state'):(root/sub).mkdir()
            write(root/'jobs/old.json',dict(lanes=[['failed','missing','live'],[],[],[]]))
            (root/'raw/failed').mkdir();write(root/'raw/failed/dispatcher_result.json',dict(complete=False))
            (root/'raw/live').mkdir()
            with patch.object(supervision,'ROOT',root),patch.object(supervision,'bind',lambda:None):
                with self.assertRaises(AssertionError):supervision.recovery_plan('old.json',0)
                write(root/'raw/live/dispatcher_result.json',dict(complete=True))
                plan=supervision.recovery_plan('old.json',0)
            value=json.loads(plan.read_text())
            self.assertEqual(value['lanes'][0],['missing'])
            self.assertEqual(value['retained_terminal_runs'][0]['status'],'FAILED_GPU_RUN_NOT_RETRIED')

    def test_coordinator_GC_policy_restores_the_callers_state(self):
        import gc
        import tempfile
        from .coordinator_probe import Probe
        previous=gc.isenabled()
        try:
            gc.enable()
            with tempfile.TemporaryDirectory() as name:
                probe=Probe(dict(defer_coordinator_cyclic_gc=True),Path(name))
                probe.begin_measurement()
                self.assertFalse(gc.isenabled())
                probe.end_measurement()
                self.assertTrue(gc.isenabled())
                probe.close()
            gc.disable()
            with tempfile.TemporaryDirectory() as name:
                probe=Probe(dict(defer_coordinator_cyclic_gc=True),Path(name))
                probe.begin_measurement();probe.close()
                self.assertFalse(gc.isenabled())
        finally:
            gc.enable() if previous else gc.disable()

    def test_model_observer_sees_late_graph_guard_installation(self):
        from .instrumentation import execution_namespace
        values={'dispatch_cg_and_sync_dp':lambda:'ORIGINAL'}
        exec('def execute_model():\n    return dispatch_cg_and_sync_dp()\n',values)
        namespace=execution_namespace(values['execute_model'])
        values['dispatch_cg_and_sync_dp']=lambda:'PARTIAL_LAYER_GUARD'
        exec('def observed():\n    return dispatch_cg_and_sync_dp()\n',namespace)
        self.assertEqual(namespace['observed'](),'PARTIAL_LAYER_GUARD')

    def test_actual_installed_model_call_seam_is_reversible(self):
        path=Path('/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu/model_runner.py')
        source=path.read_text();module=ast.parse(source)
        cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='GPUModelRunner')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute_model')
        first=min([method.lineno]+[n.lineno for n in method.decorator_list])
        original=textwrap.dedent('\n'.join(source.splitlines()[first-1:method.end_lineno])+'\n')
        changed=rewrite(original);ast.parse(changed)
        self.assertEqual(changed.count('_pc_forward_call('),3)
        with self.assertRaises(AssertionError):rewrite(changed)

    def test_credit_needs_native_retirement_and_hidden_ack(self):
        self.assertTrue(safe_completion(True,True,True,True,True))
        for args in ((False,True,True,True,True),(True,False,True,True,True),
                     (True,True,True,False,True),(True,True,True,True,False)):
            self.assertFalse(safe_completion(*args))
        self.assertTrue(safe_completion(True,True,False,False,False))

    def test_client_fallback_and_retirement_cannot_release_twice(self):
        owned={'q0','q1'}
        self.assertTrue(release_owned(owned,'q0'))
        self.assertFalse(release_owned(owned,'q0'))
        self.assertEqual(owned,{'q1'})

    def test_retirement_has_no_manual_KV_free_call(self):
        import inspect
        from .retirement import Retirement
        source=inspect.getsource(Retirement)
        self.assertNotIn('._free_blocks(',source)
        self.assertNotIn('kv_cache_manager.free(',source)

    def test_prepared_strict_arrival_contract_reaches_real_coordinator_seam(self):
        import json
        from .common import ROOT
        for filename in ('retirement_qualification_r2.json','stock_observer_qualification_r2.json'):
            plan=json.loads((ROOT/'configs'/filename).read_text())
            for item in plan['items']:
                for name in item['strict']:
                    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                    self.assertEqual(len(cfg['arrival_offsets']),cfg['count'])
                    self.assertEqual(cfg['arrival_offsets'],[0.]*8)
                    self.assertTrue(cfg['serial'])

    def test_retirement_tick_waits_for_real_owner_and_native_fence_seams(self):
        import tempfile
        from types import SimpleNamespace
        from .retirement import Retirement
        with tempfile.TemporaryDirectory() as path:
            directory=Path(path);rid='q0-test';sent=[]
            transport=SimpleNamespace(directory=directory,parts={rid:1},owners=[rid],reap=lambda:None)
            value=Retirement.__new__(Retirement)
            value.enabled=True;value.pending={rid:dict(native_request_removed=True,required_fence=7)}
            value.sent=set();value.rows=[];value.adapter=SimpleNamespace(transport=transport)
            value.cfg=dict(source_credit_release_on_retirement=True,continuous_source_credit_limit=8)
            value.channel=SimpleNamespace(send=sent.append)
            scheduler=SimpleNamespace(processed_step_seq=6)
            value.tick(scheduler);self.assertFalse(sent)
            (directory/(rid+'.ACK.json')).touch();transport.parts.clear();transport.owners[0]=None
            value.tick(scheduler);self.assertFalse(sent)
            scheduler.processed_step_seq=7;value.tick(scheduler)
            self.assertEqual(len(sent),1);self.assertEqual(sent[0]['request_id'],'q0')
            self.assertFalse(value.pending);value.tick(scheduler);self.assertEqual(len(sent),1)

    def test_buffered_coordinator_never_writes_the_measured_append(self):
        import json
        import tempfile
        from .coordinator_probe import Probe
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);path=folder/'requests.jsonl'
            probe=Probe(dict(buffer_coordinator_evidence=True),folder);row={'request_id':'q0','value':[1]}
            probe.append(path,row,{})
            self.assertFalse(path.exists())
            row['value'].append(2)
            self.assertEqual(json.loads(probe.lines[path][0])['value'],[1])
            probe.close()
            self.assertEqual(json.loads(path.read_text())['value'],[1])

    def test_arrival_window_excludes_left_boundary_and_future_arrivals(self):
        from .arrivals import preceding_counts
        self.assertEqual(preceding_counts([0.,.1,.5,1.,1.1],1.),[1,2,3,3,3])
        self.assertEqual(preceding_counts([0.,.5,1.],.5),[1,1,1])
        with self.assertRaises(AssertionError):preceding_counts([1.,0.],1.)

    def test_buffer_flag_does_not_change_existing_selective_buffers(self):
        import tempfile
        from .coordinator_probe import Probe
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);path=folder/'credit_events.jsonl';original={};row={'request_id':'q0'}
            probe=Probe(dict(buffer_coordinator_evidence=True),folder)
            probe.append(path,row,original)
            self.assertIs(original[path][0],row);self.assertFalse(probe.lines)
            probe.close();self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
