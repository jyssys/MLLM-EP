"""CPU checks for qualification gates, provenance, and evidence accounting."""
import ast
import csv
import unittest
from .contract import ROOT,PROJECT,OLD,TOL,UUIDS,cases,sha
from .analyze import load


class EvidenceTests(unittest.TestCase):
    def test_old_sources_unchanged(self):
        frozen=load(ROOT/'preflight.json')
        for name,value in frozen['old_source_hashes'].items():self.assertEqual(sha(PROJECT/name),value)
        for name,value in frozen['old_report_hashes'].items():self.assertEqual(sha(OLD/name),value)

    def test_limits_unchanged(self):
        self.assertEqual(TOL,load(ROOT/'preflight.json')['limits'])

    def test_frozen_spec(self):
        self.assertEqual(sha(ROOT/'spec_frozen.md'),load(ROOT/'preflight.json')['spec_sha256'])

    def test_R4_matrix(self):
        self.assertEqual(len(cases()),21)
        for c in cases():
            self.assertEqual(c['R'],4)
            self.assertEqual(sha(PROJECT/c['source']),c['source_sha256'])
            self.assertEqual(len(c['requests']),c['C'])
            for row in c['requests']:self.assertEqual(len(row['teacher_ids']),c['O'])

    def test_all_new_python_parses(self):
        for file in (PROJECT/'poc_flashvep/ouro_tp4').glob('*.py'):ast.parse(file.read_text())

    def test_true_tp_ownership(self):
        for k in range(4):
            row=load(ROOT/'ownership'/f'tp_fp32columns_operator_rank{k}.json')
            self.assertEqual(row['physical_GPU'],k+4)
            self.assertEqual(row['UUID'].removeprefix('GPU-'),UUIDS[k])
            self.assertEqual(row['QKV_weight_shape'],[512,2048])
            self.assertEqual(row['O_weight_shape'],[2048,512])
            self.assertEqual(row['MLP_gate_weight_shape'],[1408,2048])
            self.assertEqual(row['down_weight_shape'],[2048,1408])

    def test_operator_gate(self):
        for k in range(4):
            checks=load(ROOT/'correctness'/f'tp_fp32columns_operator_rank{k}.json')
            self.assertEqual(len(checks),48);self.assertTrue(all(v['PASS'] for v in checks))
            self.assertEqual({v['loop'] for v in checks},{0,1,2,3})
            self.assertEqual({v['step'] for v in checks},{0,1})

    def test_actual_operator_reductions(self):
        for k in range(4):
            ops=load(ROOT/'correctness'/f'clean2_tp_operator_diagnostic_operations_rank{k}.json')
            self.assertEqual(len(ops),32)
            self.assertEqual({v['kind'] for v in ops},{'all_reduce'})
            self.assertEqual({v['dtype'] for v in ops},{'torch.float32'})
        trace=load(ROOT/'profiles/clean2_tp_operator_diagnostic/summary.json')
        self.assertEqual([r['NCCL_kernel_count'] for r in trace['ranks']],[32]*4)

    def test_full_failure_preserved(self):
        for k in range(4):
            rows=load(ROOT/'correctness'/f'tp_fp32columns_full_rank{k}.json')
            failure=next(v for v in rows if v.get('P')==512 and v.get('teacher_forced'))
            self.assertFalse(failure['PASS']);self.assertTrue(failure['all_token_agreement'])
            self.assertGreater(max(v['relative_L2'] for v in failure['hidden_errors']),TOL['hidden_relative_L2'])

    def test_no_invalid_tp_benchmark(self):
        from .tp_driver import require_qualification
        with self.assertRaises(AssertionError):require_qualification()
        with (ROOT/'primary_matrix_status.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),21)
        self.assertTrue(all(r['TP4_matched']=='NOT_RUN_C2_NUMERICAL_GATE' for r in rows))
        self.assertFalse(any((ROOT/'raw').glob('tp_pilot*')))

    def test_graph_short_probe(self):
        for k in range(4):
            rows=load(ROOT/'correctness'/f'tp_graph_bounded_numeric_probe_rank{k}.json')
            self.assertTrue(all(v['PASS'] for v in rows if v.get('teacher_forced')))
            self.assertTrue(all(v['token_exact'] for v in rows if v.get('lane')=='free_running_greedy'))

    def test_MP_bitwise_and_overlap(self):
        for arm in ['layer_pp4','loop_pp4']:
            for k in range(4):
                checks=load(ROOT/'correctness'/f'clean_mp_correctness_{arm}_rank{k}.json')
                self.assertEqual(len(checks),4);self.assertTrue(all(v['PASS'] and v['KV_bitwise'] for v in checks))
                self.assertTrue(all(v['KV_slots']==24 for v in checks))
            trace=load(ROOT/'profiles'/f'clean_mp_correctness_{arm}/summary.json')
            self.assertGreater(trace['different_request_kernel_overlap_ms'],0)
            self.assertTrue(all(v['kernel_count']>0 for v in trace['ranks']))

    def test_PP_control_integrity(self):
        with (ROOT/'pp_controls_raw.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),24)
        for r in rows:
            self.assertEqual(int(r['graph_captures_in_clock']),0)
            self.assertEqual(int(r['hidden_peer_receives']),int(r['C'])*32*(15 if r['arm']=='layer_pp4' else 3))
            self.assertNotIn('PP_transport_layer',r['source'])
        for arm in ['layer_pp4','loop_pp4']:
            for rep in range(3):
                for index in [0,1]:
                    directory=ROOT/'raw'/f'PP_transport_clean_{arm}_r{rep}'
                    for k in range(4):
                        r=load(directory/f'case_{index:02d}_rank{k}.json.gz')
                        self.assertEqual(r['graph_captures_during_run'],0)
                        if k==3:
                            for v in r['outputs'].values():
                                self.assertEqual(len(v['computed_ids']),32)
                                self.assertEqual(v['computed_ids'],v['ids'])

    def test_external_evidence_limits(self):
        rows=load(ROOT/'external_numerical_comparison.json')
        self.assertEqual(len(rows),12)
        self.assertTrue(all(v['token_exact'] for v in rows))
        self.assertTrue(all(v['PASS'] for v in rows if v['P']==128))
        self.assertTrue(all(not v['PASS'] for v in rows if v['P']!=128))
        for P,O in [(128,8),(512,32),(2048,32)]:
            r=load(ROOT/'correctness'/f'external_natural_P{P}_O{O}.json')
            self.assertTrue(r['token_exact']);self.assertEqual(r['finish_reason'],'length')

    def test_exclusions(self):
        rows=load(ROOT/'excluded_runs.json')['excluded']
        self.assertEqual(sum(v.startswith('raw/') for v in rows),6)
        self.assertTrue(all((ROOT/p).exists() for p in rows))
        self.assertTrue(all('clean_' not in v for v in rows))


if __name__=='__main__':unittest.main()
