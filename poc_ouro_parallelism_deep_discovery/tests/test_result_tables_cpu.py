import copy
import unittest

from poc_ouro_parallelism_deep_discovery.scripts.build_result_tables_cd import (
    extension_panels, native_extension_panels, native_panels, quality_panels)


class ResultTablesTest(unittest.TestCase):
    def test_different_endpoints_do_not_create_speedup(self):
        values=[]
        for mode,wall,GPUs in [('TP1',10,1),('TP4',5,4),('DP4',4,4),('TP2_DP2',3,4)]:
            values.append(dict(status='AVAILABLE',size='1.4B',mode=mode,C=1,context=1024,
                median_wall_ms=wall,reserved_GPUs=GPUs,repetitions=5,
                boundary='API' if mode.startswith('TP') and mode!='TP2_DP2' else 'PARENT',
                median_output_TPS=128000/wall))
        rows=native_panels(dict(summaries=values))
        self.assertEqual(rows[0]['speedup_vs_native_TP1'],1)
        self.assertEqual(rows[1]['speedup_vs_native_TP1'],2)
        self.assertIsNone(rows[2]['speedup_vs_native_TP1'])
        self.assertIsNone(rows[3]['speedup_vs_native_TP1'])
        self.assertTrue(all(r['speedup_vs_strongest_same_P'] is None for r in rows))

    def test_quality_row_is_not_timing(self):
        row=dict(size='1.4B',mode='fixed',runtime='native',TP=4,dataset='gsm8k',N=8,correct=4,
            accuracy_percent=50,accuracy_CI_percent=[12.5,87.5],parsing_failures=0,truncated=0)
        value=quality_panels(dict(summaries=[row]))[0]
        self.assertEqual(value['evidence_class'],'QUALITY_ONLY')
        self.assertEqual(value['speedup_vs_native_1GPU'],'NOT_RUN')

    def test_partial_extension_rejected(self):
        with self.assertRaises(ValueError):
            extension_panels(dict(evidence_class='QUALITY_ONLY',summaries=[dict(N=31)]))

    def test_same_mode_name_does_not_override_endpoint(self):
        rows=[dict(status='AVAILABLE',size='1.4B',mode=mode,C=1,context=1024,
            median_wall_ms=wall,reserved_GPUs=GPUs,repetitions=5,boundary=boundary,median_output_TPS=128000/wall)
            for mode,wall,GPUs,boundary in [('TP1',10,1,'API'),('TP4',5,4,'DIFFERENT')]]
        self.assertIsNone(native_panels(dict(summaries=rows))[1]['speedup_vs_native_TP1'])

    def test_quality_summary_requires_an_admitted_complete_arm(self):
        with self.assertRaises(ValueError):
            extension_panels(dict(evidence_class='QUALITY_ONLY',summaries=[dict(N=32,size='1.4B',mode='fixed')]))

    def native_extension(self):
        records=[dict(size='1.4B',dataset=domain,id=f'{domain}/{i}',correct_completed=i<16)
                 for domain in ('gsm8k','math500') for i in range(32)]
        summaries=[dict(size='1.4B',dataset=domain,N=32,correct=16,accuracy_percent=50,
                       accuracy_CI_percent=[31.25,68.75],parsing_failures=0,truncated=0)
                   for domain in ('gsm8k','math500')]
        return dict(evidence_class='QUALITY_ONLY',candidate_E2E='NOT_RUN',candidate_quality='NOT_RUN',
                    summaries=summaries,records=records)

    def test_native_extension_is_quality_only(self):
        rows=native_extension_panels(self.native_extension())
        self.assertEqual(len(rows),2)
        self.assertTrue(all(row['GPUs']==4 and row['evidence_class']=='QUALITY_ONLY' for row in rows))
        self.assertTrue(all(row['speedup_vs_native_1GPU']=='NOT_RUN' for row in rows))

    def test_native_extension_rejects_partial_arm(self):
        value=self.native_extension();value['records'].pop()
        with self.assertRaises(ValueError):native_extension_panels(value)

    def test_native_extension_rejects_duplicate_ID(self):
        value=self.native_extension();value['records'][-1]=copy.deepcopy(value['records'][0])
        with self.assertRaises(ValueError):native_extension_panels(value)

    def test_native_extension_rejects_numerator_mismatch(self):
        value=self.native_extension();value['summaries'][0]['correct']=17
        with self.assertRaises(ValueError):native_extension_panels(value)

    def test_native_extension_rejects_candidate_relabel(self):
        value=self.native_extension();value['candidate_E2E']='PASS'
        with self.assertRaises(ValueError):native_extension_panels(value)


if __name__=='__main__':
    unittest.main()
