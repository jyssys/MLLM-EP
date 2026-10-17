import json
import unittest
from poc_recurrent_asymmetry.contract import ROOT,read,sha


class EvidenceContract(unittest.TestCase):
    def test_authoritative_spec_and_reference(self):
        frozen=read(ROOT/'metrics/frozen_contract.json')
        self.assertEqual(sha(ROOT/'reports/spec.md'),frozen['spec_sha256'])
        self.assertTrue(read(ROOT/'outputs/baseline_v1/status.json')['exact_reference'])
        rows=read(ROOT/'outputs/baseline_v1/integrity.json')['rows']
        self.assertEqual(len(rows),8)
        self.assertTrue(all(r['logical_executions']==96 and r['all_KV_bitwise'] for r in rows))

    def test_all_recorded_jobs_use_only_assigned_devices(self):
        for path in (ROOT/'outputs').glob('*/run_contract.json'):
            row=read(path)
            self.assertLessEqual(set(map(int,row['mask'].split(','))),{4,5,6,7})

    def test_negative_numerical_checks_are_not_promoted(self):
        analysis=ROOT/'metrics/analysis_final_v1'
        rows=read(analysis/'native_fidelity.json')
        self.assertEqual({r['TP'] for r in rows},{1,2,4})
        self.assertTrue(all(r['qualification']=='REFERENCE_PARTIAL' for r in rows))
        chunk=read(analysis/'chunk_correctness_summary.json')
        self.assertEqual(chunk['qualification'],'REFERENCE_PARTIAL')
        self.assertLess(chunk['numerical_pass'],chunk['cases'])
        self.assertEqual(chunk['top1_pass'],chunk['cases'])
        self.assertFalse(read(analysis/'summary.json')['architecture_GO'])

    def test_ipc_red_green_preserves_allocator_cause(self):
        cases=read(ROOT/'outputs/ipc_probe_v1/cases.json')
        evidence={r['case']:json.loads(r['stdout']) for r in cases}
        self.assertEqual(evidence['expandable']['cuda_IPC_return_code'],1)
        self.assertEqual(evidence['cuda_malloc']['cuda_IPC_return_code'],0)

    def test_trace_partition_does_not_double_count(self):
        for row in read(ROOT/'metrics/nsight/summary.json')['rows']:
            self.assertAlmostEqual(sum(row['exclusive_ms'].values()),row['activity_envelope_ms'],places=6)

    def test_communication_accounting_stays_separate(self):
        rows=read(ROOT/'outputs/physical_v1/physical/p2p.json')
        self.assertEqual(len(rows),108)
        self.assertEqual({(r['source'],r['destination']) for r in rows},
                         {(a,b) for a in range(4) for b in range(4) if a!=b})
        self.assertTrue(all(r['median_ms']>0 for r in rows))
        self.assertTrue(all(r['correctness'] for r in read(ROOT/'outputs/physical_v1/physical/cp_operator.json')))


if __name__=='__main__':unittest.main()
