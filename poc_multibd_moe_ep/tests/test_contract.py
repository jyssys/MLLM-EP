"""CPU tests; synthetic fixture values are never serving results."""
import unittest
import sys
import numpy as np
from poc_multibd_moe_ep.contract import *
from poc_dllm_ep_comm.metrics import traffic


class ContractTests(unittest.TestCase):
    def test_conversion_uses_paired_requests(self):
        from poc_multibd_moe_ep.scripts.analyze_conversion import paired_conversion
        rows=[dict(request_id='a',backend='ep4',mode='clean',R=1,TPF=2,TPS=100,step_ms=20),
              dict(request_id='a',backend='ep4',mode='clean',R=2,TPF=4,TPS=150,step_ms=26.666),
              dict(request_id='b',backend='ep4',mode='clean',R=2,TPF=8,TPS=300,step_ms=26.666)]
        result=paired_conversion(rows)
        self.assertEqual(len(result),2)
        self.assertEqual(result[-1]['PCE'],.75)
        self.assertEqual(result[-1]['TPS_gain'],1.5)

    def test_no_layer_pseudoreplication(self):
        from poc_dllm_ep_comm.metrics import bootstrap_mean
        result=bootstrap_mean([.6,.8])
        self.assertEqual(result['n'],2)

    def test_stage_cost_normalizes_by_real_forward_count(self):
        from poc_multibd_moe_ep.scripts.analyze_conversion import summarize_ep
        result=summarize_ep([dict(D_ms=6,E_ms=2,C_ms=2,rank_max_mean=1.5,
                                  rank_cv=.2,max_rank_load=12)],committed=2,nfe=4)
        self.assertEqual(result['D_ms_per_forward'],1.5)
        self.assertEqual(result['EP_ms_per_commit'],5)
        self.assertEqual(result['EP_ms_per_forward'],2.5)

    def test_large_layer_csv_compression_is_lossless(self):
        import csv,gzip,json,tempfile
        from poc_multibd_moe_ep.scripts.analyze_conversion import write_csv
        with tempfile.TemporaryDirectory(prefix='multibd-csv-test-') as directory:
            path=Path(directory)/'ep_metrics.csv'
            write_csv(path,[dict(forward=2,rank_assignments=[1,2,3,4],D_ms=.25)])
            self.assertFalse(path.exists())
            with gzip.open(path.with_suffix('.csv.gz'),'rt') as inp:
                row=next(csv.DictReader(inp))
            self.assertEqual(json.loads(row['rank_assignments']),[1,2,3,4])
            self.assertEqual(float(row['D_ms']),.25)

    def test_repeats_aggregate_not_last_observation(self):
        from poc_multibd_moe_ep.scripts.analyze_conversion import paired_conversion
        rows=[dict(request_id='a',backend='ep4',mode='clean',R=1,TPF=2,TPS=100,step_ms=20),
              dict(request_id='a',backend='ep4',mode='clean',R=2,TPF=4,TPS=100,step_ms=40),
              dict(request_id='a',backend='ep4',mode='clean',R=2,TPF=4,TPS=200,step_ms=20)]
        self.assertEqual(paired_conversion(rows)[1]['TPS_gain'],1.5)

    def test_quality_normal_eos_not_cap_truncation(self):
        import json
        result=json.loads((ROOT/'artifacts/R1_ep4_calibration_clean/request0_repeat3.json').read_text())
        t=result['trajectories'][0]
        self.assertTrue(t['is_truncated'])  # Official name includes normal EOS!
        self.assertTrue(t['eos_token_generated'])
        self.assertFalse(t['max_new_tokens_reached'] or t['max_nfe_reached'])

    def test_quality_extraction_preserves_prefill_function(self):
        from poc_multibd_moe_ep.scripts.evaluate_quality import extract_code
        code=extract_code('Here is the completed code:\n```python\ndef f(x):\n',
                          '    return x+1\n```<|role_end|>')
        self.assertEqual(code,'def f(x):\n    return x+1\n')

    def test_destination_dedup_not_path_bytes(self):
        ids=np.array([[64,65,128,129,192,193,0,1]])
        counts=traffic(ids,source_rank=0)
        self.assertEqual(int(counts['remote_paths'][0]),6)
        self.assertEqual(int(counts['dispatch'][0]),3*4096)
        self.assertEqual(int(counts['total'][0]),2*3*4096)

    def test_actual_strided_source_layout(self):
        rows=np.arange(128)
        owned=[rows[r::4].tolist() for r in range(4)]
        self.assertEqual(sorted(sum(owned,[])),rows.tolist())
        self.assertTrue(all(p%4==r for r,positions in enumerate(owned) for p in positions))

    def test_ep_owner_partition(self):
        expert_sets=[list(range(r*64,(r+1)*64)) for r in range(4)]
        self.assertEqual(sum(expert_sets,[]),list(range(256)))
        self.assertEqual(64*2048*512*3*2,402653184)

    def test_fixed_R_is_capacity_not_active_count(self):
        import json
        result=json.loads((ROOT/'artifacts/cpu_state_machine.json').read_text())
        self.assertTrue(result['pass_all'])
        for row in result['rows']:
            self.assertEqual(row['physical_rows'],row['R']*32)
            self.assertLessEqual(row['active_after_29_of_32'],row['R'])

    def test_local_id_repair_keeps_source_metadata(self):
        sys.path.insert(0,str(UPSTREAM))
        import torch
        from poc_multibd_moe_ep.bounded_bridge import repair_normal_owner_local_ids
        from diffulex.moe.dispatcher.deepep_dispatcher import DeepEPDispatcher
        from diffulex.moe.metadata import DispatcherStage
        repair_normal_owner_local_ids()
        d=DeepEPDispatcher.__new__(DeepEPDispatcher)
        d._stage=DispatcherStage.AFTER_DISPATCH_A
        d.top_k=8;d.num_local_experts=64
        d.local_expert_start=192;d.local_expert_end=256
        h=torch.ones((1,4),dtype=torch.bfloat16)
        ids=torch.tensor([[0,1,-1,-1,-1,-1,-1,-1]])
        weights=torch.tensor([[.4,.6,0,0,0,0,0,0]])
        d._dispatch_intermediate_state=(h,True,1,h,ids,weights,object(),torch.tensor([0,0,0,1]))
        output=d.dispatch_b()
        self.assertEqual(output.metadata.total_recv_slots,2)
        self.assertEqual(output.recv_local_expert_ids.tolist(),[0,1])
        self.assertEqual(output.metadata.recv_token_indices.tolist(),[0,0])

    def test_real_acceptance_mismatch_is_not_hidden_by_token_parity(self):
        from poc_multibd_moe_ep.scripts.audit_parity import inspect
        audit=inspect(ROOT/'artifacts/R3_ep4_held_out_paired_gpu4567_20261013')
        failed=[r for r in audit['groups'] if not r['pass_all']]
        self.assertTrue(failed,'Recorded counterexample must remain visible')
        case=next(r for r in failed if r['domain']=='humaneval_plus' and r['selected_index']==5)
        self.assertTrue(all(t['final_tokens_equal'] and t['NFE_equal'] for t in case['comparisons']))
        self.assertTrue(any(not t['exact_acceptance_equal'] for t in case['comparisons']))

    def test_user_gpu_override_is_exact(self):
        self.assertEqual(MASK,'4,5,6,7')
        self.assertEqual(GPUS,[4,5,6,7])
        self.assertEqual(len(set(UUIDS)),4)

    def test_authoritative_spec_snapshot_is_exact(self):
        import json
        expected=json.loads((ROOT/'artifacts/pins.json').read_text())['spec_sha256']
        self.assertEqual(sha(ROOT/'authoritative_spec.md'),expected)
        self.assertEqual(sha(SPEC),expected)

    def test_optional_failed_tp_control_is_never_promoted(self):
        import json
        from poc_multibd_moe_ep.scripts.build_report import scientific_lane_status
        session='gpu4567_det_20261014'
        primary=[json.loads((ROOT/f'artifacts/R{r}_ep4_held_out_paired_{session}/status.json').read_text())
                 for r in [1,2,3,4]]
        self.assertTrue(all(case['success'] for case in primary))
        parity=json.loads((ROOT/f'results/{session}/tp_control_parity_audit.json').read_text())
        self.assertFalse(parity['pass_all'])
        state=scientific_lane_status(ROOT,session,dict(tp4_held_out_pass_all=parity['pass_all']))
        self.assertTrue(state['primary_ep_complete'])
        self.assertFalse(state['optional_tp_valid'])

    def test_live_burner_log_snapshot_does_not_interrupt_writer(self):
        import json,subprocess,tempfile
        from poc_multibd_moe_ep.scripts.run_guarded import freeze_restored_burner_log
        with tempfile.TemporaryDirectory(prefix='multibd-log-test-') as directory:
            root=Path(directory);output=root/'session';output.mkdir()
            live_parent=root/'live';live_parent.mkdir()
            snapshot=output/'restored_burner.log'
            with snapshot.open('w') as stream:
                stream.write('before\n');stream.flush()
                # CPU-only fixture with a matching argv; never starts a GPU burner.
                proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
                    'utilize.py','--gpus',MASK,'--duration-sec','30'],stdout=stream)
                try:
                    freeze_restored_burner_log(output,live_parent)
                    archive=json.loads((output/'burner_log_archive.json').read_text())
                    stream.write('after\n');stream.flush()
                    self.assertIsNone(proc.poll())
                    self.assertEqual(snapshot.read_text(),'before\n')
                    self.assertEqual(Path(archive['ongoing_log']).read_text(),'before\nafter\n')
                finally:
                    proc.terminate();proc.wait(timeout=5)


if __name__=='__main__':unittest.main()
