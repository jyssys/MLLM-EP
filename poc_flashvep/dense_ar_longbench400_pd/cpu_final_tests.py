"""Run CPU regression tests without editing runtime files or GPU owners."""
import ast
from fractions import Fraction
import json
import random
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from .common import ROOT,SOURCE,write,preserve,digest
from .geometry import Partition,matched_suffixes,active_spans
from .qualification_gate import KV_equal
from .window_analysis import overlap,duration
from .c0_queue_model import nnls,simulate
from .decode_prefill_throttle import call
from .prefill_completion_guard import capped_call


class CPUChecks(unittest.TestCase):
    def test_python_syntax(self):
        for path in SOURCE.glob('*.py'):
            ast.parse(path.read_text(),filename=str(path))

    def test_two_dimensional_conservation(self):
        for length in (32,257,3987,8192,16384,32768):
            for upper in (0,2,4,8,12):
                for suffix in (0,length//16,length//8,length//3):
                    p=Partition(length,upper,suffix)
                    self.assertEqual(p.P_rows+p.D_rows,36*length)
                    self.assertEqual(p.P_attention_pairs+p.D_attention_pairs,36*length*(length+1)//2)
                    self.assertEqual(p.D_rows,length*upper+(36-upper)*suffix)
                    # Prefix upper layers precede all suffix upper queries.
                    for token in range(min(length,48)):
                        for layer in range(36):
                            if p.owners(token,layer)=='P':
                                self.assertLess(token,p.prefix)
                                self.assertLess(layer,p.lower)

    def test_alignment_optimal_residual(self):
        rng=random.Random(2026112481)
        for _ in range(100):
            lengths=[rng.randint(3000,36000) for _ in range(rng.randint(1,32))]
            for cut in (Fraction(2),Fraction(8),Fraction(25,4),Fraction(39,4),Fraction(27,2)):
                suffix,audit=matched_suffixes(lengths,cut)
                self.assertTrue(all(0<s<L and s%16==0 for s,L in zip(suffix,lengths)))
                self.assertLessEqual(abs(audit['signed_error_rows']),288)
                self.assertEqual(audit['actual_D_rows'],36*sum(suffix))
                self.assertEqual(audit['exact_match'],36*sum(suffix)==cut*sum(lengths))

    def test_mixed_row_span_ownership(self):
        args=([0,8,9],[12,128],[16,0],[32,0])
        self.assertEqual(active_spans(*args,0,'D'),[(0,4,8),(1,8,9)])
        self.assertEqual(active_spans(*args,32,'D'),[(0,0,8),(1,8,9)])

    def test_digest_algorithm_mismatch_is_not_tensor_inequality(self):
        raw=[dict(hash='a')]
        leaf=[dict(hash='a',digest_algorithm='SHA256_OF_SHA256_4096B_LEAVES_V1',chunk_bytes=4096)]
        with self.assertRaises(ValueError):
            KV_equal(raw,leaf)
        self.assertTrue(KV_equal(leaf,leaf))
        self.assertFalse(KV_equal(leaf,[dict(leaf[0],hash='b')]))

    def test_prefill_policy_restores_original_state(self):
        scheduler=SimpleNamespace(prefill_capacity_bound=True)
        def original(value,throttle_prefills=False):
            self.assertTrue(throttle_prefills)
            self.assertFalse(value.prefill_capacity_bound)
            return 7
        self.assertEqual(call(original,scheduler),7)
        self.assertTrue(scheduler.prefill_capacity_bound)
        def failed(value,throttle_prefills=False):
            raise RuntimeError('Expected fixture')
        with self.assertRaises(RuntimeError):
            call(failed,scheduler)
        self.assertTrue(scheduler.prefill_capacity_bound)

    def test_interval_union_against_integer_reference(self):
        rng=random.Random(2026112482)
        for _ in range(1000):
            a=[sorted((rng.randrange(100),rng.randrange(100))) for _ in range(8)]
            b=[sorted((rng.randrange(100),rng.randrange(100))) for _ in range(8)]
            aa={i for left,right in a for i in range(left,right)}
            bb={i for left,right in b for i in range(left,right)}
            self.assertEqual(duration(overlap(a,b)),len(aa&bb))

    def test_completion_cap_restoration(self):
        scheduler=SimpleNamespace(max_num_running_reqs=1024)
        def original(value):
            self.assertEqual(value.max_num_running_reqs,2)
            return 7
        self.assertEqual(capped_call(original,scheduler,2),7)
        self.assertEqual(scheduler.max_num_running_reqs,1024)
        def failed(value):
            raise RuntimeError('Expected fixture')
        with self.assertRaises(RuntimeError):
            capped_call(failed,scheduler,2)
        self.assertEqual(scheduler.max_num_running_reqs,1024)

    def test_registration_deferral_preserves_requests_controls_and_outputs(self):
        from .request_admission_deferral import test
        test()
        from .registration_deferral_campaign import registration_signature
        native=dict(method='Native',d_compute_throttle=True,defer_D_remote_setup=True)
        self.assertNotEqual(registration_signature(native),
            registration_signature(dict(native,defer_D_request_registration=True)))

    def test_decode_compiler_override_is_role_local(self):
        from .role_compile_options import cpu_test
        cpu_test()
        from .decode_compile_custom_ops import cpu_test as custom_ops_test
        custom_ops_test()
        from .decode_compile_ir_options import cpu_test as ir_test
        ir_test()
        from .scoped_compile_options import cpu_test as scoped_test
        scoped_test()
        from .eager_prefill_compiled_adapter import cpu_test as prompt_test
        prompt_test()
        from .decode_only_compile_followup import role_signature
        base=dict(method='Layer',axis_kind='Layer',axis_cut=8,
                  execution_family='MATCHED_NO_COMPILE')
        self.assertNotEqual(role_signature(base),role_signature(dict(base,
            preserve_compiled_decode_only=True,D_execution_family_override='INSTALLED_DEFAULT_COMPILE')))
        plan=json.loads((ROOT/'configs/decode_only_compile_correctness_frozen_v2.json').read_text())
        for item in plan['qualification'].values():
            for name in item['jobs']:
                cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                self.assertTrue(cfg['correctness_only'])
                if cfg['controlled_pipeline']:
                    self.assertEqual(cfg['continuous_source_credit_limit'],0)
                else:
                    self.assertEqual(cfg['continuous_source_credit_limit'],8)

    def test_role_local_performance_requires_its_exact_source_and_signature(self):
        from . import run
        from .scoped_decode_compile_followup import signature
        cfg=dict(method='Layer',axis_kind='Layer',axis_cut=8,
            preserve_compiled_decode_only=True,D_partial_IR_norm_priority=True,
            D_compile_role_gate='fixture.json',source_capsule_SHA256='source')
        with tempfile.TemporaryDirectory(prefix='role_gate_regression_',dir=ROOT/'tools') as directory:
            base=Path(directory);(base/'state').mkdir()
            write(base/'state/fixture.json',dict(passed=True,full400_passed=True,
                role_signature=signature(cfg),source_capsule_SHA256='source'))
            with patch.object(run,'ROOT',base):
                run.require_role_compile_gate(cfg)
                with self.assertRaises(AssertionError):
                    run.require_role_compile_gate(dict(cfg,axis_cut=12))
                with self.assertRaises(AssertionError):
                    run.require_role_compile_gate(dict(cfg,source_capsule_SHA256='different'))
                with self.assertRaises(FileNotFoundError):
                    run.require_role_compile_gate(dict(cfg,D_compile_role_gate='missing.json'))
                run.require_role_compile_gate(dict(cfg,correctness_only=True))

    def test_intermediate_metric_cache_is_reconstructed(self):
        from .final_confirmation_analysis import complete_metric
        final=dict(runtime_source_archive_SHA256='source',dataset_SHA256='dataset',actual_geometry={})
        intermediate=dict(TTFT_p95_ms=3.5)
        with patch('poc_flashvep.dense_ar_longbench400_pd.final_confirmation_analysis.summarize',return_value=final) as rebuild:
            self.assertIs(complete_metric(intermediate,'fixture'),final)
            rebuild.assert_called_once_with('fixture')
            self.assertIs(complete_metric(final,'fixture'),final)
            self.assertEqual(rebuild.call_count,1)

    def test_deflection_host_paths_have_no_fabricated_P_stages(self):
        from .service_accounting import analyze
        result=analyze('FinalC3_Heldout200_q2.8_s0_BestWholeDeflection_v1')
        self.assertEqual(sum(result['route_counts'].values()),200)
        self.assertGreater(result['route_counts']['D_LOCAL_FULL36'],0)
        for row in result['rows']:
            self.assertAlmostEqual(sum(row['stages_ms'].values()),row['TTFT_ms'],places=3)
            if row['route']=='D_LOCAL_FULL36':
                self.assertFalse(any('P_' in k for k in row['stages_ms']))
                self.assertNotIn('P_prepare_to_ready_ms',row)

    def test_decode_label_does_not_imply_post_first_output(self):
        import bisect
        from .final_compiler_artifacts import post_emission_step_indices
        from .common import ROOT
        folder=ROOT/'raw/EagerDecode_I3_O16_Cap1_s0_StockToken_v2'
        steps=json.loads((folder/'engine_steps_decode.json').read_text())['rows']
        queries=json.loads((folder/'actual_queries_decode.json').read_text())['rows']
        requests=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
        first={'q'+str(r['sample_id']):r['token_events'][0]['worker_time'] for r in requests}
        starts=[r['start'] for r in steps];groups={}
        for query in queries:
            i=bisect.bisect_right(starts,query['time'])-1
            if i>=0 and query['time']<=steps[i]['end']:groups.setdefault(i,[]).append(query)
        naive=[i for i,q in groups.items() if all(r['logical_phase']=='DECODE' for r in q)]
        selected=post_emission_step_indices(steps,groups,first)
        self.assertTrue(selected)
        self.assertGreater(len(naive),len(selected))
        self.assertTrue(any(steps[i]['start']<first[q['request_id']] for i in naive for q in groups[i]))
        self.assertTrue(all(steps[i]['start']>=first[q['request_id']] for i in selected for q in groups[i]))

    def test_final_composition_signature_preserves_meaningful_controls(self):
        from .final_confirmation_jobs import signature
        native=dict(method='Native',continuous_source_credit_limit=0,hidden_source_slots=4,
            hidden_destination_slots=4)
        same=dict(native,d_compute_throttle=False,hidden_source_slots=16,hidden_destination_slots=16)
        self.assertEqual(signature(native),signature(same))
        self.assertNotEqual(signature(native),signature(dict(native,continuous_source_credit_limit=8)))
        self.assertNotEqual(signature(native),signature(dict(native,D_running_admission_cap=2)))
        layer=dict(native,method='Layer',axis_kind='Layer',axis_cut=8)
        self.assertNotEqual(signature(layer),signature(dict(layer,hidden_source_slots=16)))

    def test_terminal_work_does_not_imply_scientific_success(self):
        from .final_work_accounting import terminal_jobs
        receipts={'raw/valid/dispatcher_result.json':dict(complete=True),
            'raw/failed/dispatcher_result.json':dict(complete=False)}
        stopped=[dict(name='blocked',status='NOT_RUN',reason='BLOCKED_CORRECTNESS')]
        with patch('poc_flashvep.dense_ar_longbench400_pd.final_work_accounting.read',receipts.get):
            result=terminal_jobs(['valid','failed','blocked'],stopped)
            self.assertTrue(result['passed'])
            self.assertEqual([r['status'] for r in result['results']],['COMPLETE','FAILED','NOT_RUN'])
            self.assertFalse(terminal_jobs(['unexplained_missing'])['passed'])

    def test_large_archive_ledger_uses_verified_compressed_metadata(self):
        from . import publication
        with tempfile.TemporaryDirectory(prefix='publication_metadata_regression_',dir=ROOT/'tools') as directory:
            base=Path(directory);target=base/'campaign';source=base/'source';source.mkdir()
            (target/'state').mkdir(parents=True)
            archive=target/'archives/evidence_archive_manifest_v1.json'
            document=dict(verified_lossless=True,public_archive=dict(parts=[]),
                local_members=[dict(path=f'raw/request{i}/fixture_tensor.pt',bytes=1024,SHA256='a'*64)
                               for i in range(1024)],fixture_only=True)
            write(archive,document)
            before=digest(archive)
            self.assertGreater(archive.stat().st_size,64*1024)
            with patch.multiple(publication,ROOT=target,REPO=base,SOURCE=source,DOCUMENTS=(),MAX_BLOB_BYTES=64*1024):
                selected=publication.prepare()
            self.assertNotIn(archive,selected)
            self.assertIn(archive.with_name(archive.name+'.gz'),selected)
            self.assertIn(archive.with_name(archive.stem+'.index.json'),selected)
            self.assertEqual(digest(archive),before)
            self.assertTrue(all(p.stat().st_size<64*1024 for p in selected))

    def test_nonnegative_fit_and_state_updating_replay(self):
        matrix=np.array([[1,0],[0,1],[1,1],[2,1]],float)
        weights,_=nnls(matrix,matrix@np.array([2.,3.]))
        np.testing.assert_allclose(weights,[2.,3.],atol=1e-10)
        model=dict(P=[100,0,0,0,0,0],D=[50,0,0,0,0,0,0],decode_step=[10,0])
        result=simulate([0]*8,[4096]*8,[Partition(4096)]*8,8,model)
        self.assertEqual(len(result['rows']),8)
        first=result['rows'][0]
        self.assertAlmostEqual(first['first'],.150)
        self.assertAlmostEqual(first['finish'],.220)
        self.assertGreater(result['rows'][7]['P_start'],result['rows'][1]['P_start'])
        self.assertTrue(all(r['P_start']<=r['P_ready']<=r['D_start']<=r['first']<=r['finish']
            for r in result['rows']))


def main():
    preserve()
    source_hashes={p.name:digest(p) for p in SOURCE.iterdir() if p.suffix in ('.py','.cc','.cu','.h')}
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CPUChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        from .cpu_regression import main as row_regression
        row_regression()
        from .remote_setup_deferral import test as metadata_test
        metadata_test()
    preserve()
    assert source_hashes=={p.name:digest(p) for p in SOURCE.iterdir() if p.suffix in ('.py','.cc','.cu','.h')},'Source changed during CPU tests'
    record=dict(time=time.time(),passed=result.wasSuccessful(),
        source_hashes=source_hashes,
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        failure_messages=[message for _,message in result.failures],
        error_messages=[message for _,message in result.errors],
        evidence='CPU_REGRESSION',scope='CPU ownership,alignment,interval,policy-state and replay tests. '
            'The actual row-aware forward seam uses CPU surrogate operators. '
            'These tests do not replace GPU causal attention,state,lifetime,quality or ITL gates.')
    write(ROOT/'analysis'/('cpu_regression_start_'+str(time.time_ns())+'.json'),record)
    write(ROOT/'analysis/cpu_final_regression_v1.json',record)
    assert result.wasSuccessful()


if __name__=='__main__':
    main()
