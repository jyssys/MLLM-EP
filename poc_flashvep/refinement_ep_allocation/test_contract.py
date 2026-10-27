"""CPU-only checks for partition, identity, and statistical accounting."""
import unittest
from .common import ROOT, rows
from .audit import wilson
from .postprocess import paired_bootstrap,normalize_native_record

class ContractTests(unittest.TestCase):
    def test_train_disjoint(self):
        cal=rows(ROOT/'manifests/calibration.jsonl')
        val=rows(ROOT/'manifests/validation.jsonl')
        self.assertEqual(len(cal),256)
        self.assertEqual(len(val),128)
        self.assertTrue({r['id'] for r in cal}.isdisjoint(r['id'] for r in val))
        self.assertTrue(all(r['answer'] is None for r in cal+val))

    def test_full_evaluation_counts(self):
        self.assertEqual(len(rows(ROOT/'manifests/gsm8k_test.jsonl')),1319)
        self.assertEqual(len(rows(ROOT/'manifests/humaneval_test.jsonl')),164)

    def test_wilson(self):
        low,high=wilson(1211,1319)
        self.assertAlmostEqual(low,0.9020805854049858)
        self.assertAlmostEqual(high,0.931730596309351)

    def test_paired_identity(self):
        self.assertEqual(paired_bootstrap([True,False],[True,False],repetitions=100),[0.,0.])

    def test_native_counter_and_eos_correction(self):
        raw=dict(nfe=100,prompt_tokens=2000,gen_length_cap=2048,generation_tokens=180,
                 termination_reason='GEN_LENGTH_CAP')
        corrected=normalize_native_record(raw)
        self.assertEqual(corrected['nfe'],158)
        self.assertEqual(corrected['num_explicit_prefill_forwards'],59)
        self.assertEqual(corrected['termination_reason'],'EOS_FILTERED_SOURCE_DERIVED')
        self.assertEqual(raw['nfe'],100)
        self.assertEqual(raw['termination_reason'],'GEN_LENGTH_CAP')
        self.assertEqual(normalize_native_record(corrected),corrected)

    def test_measured_native_counters_are_not_corrected(self):
        raw=dict(nfe=158,prompt_tokens=2000,gen_length_cap=2048,generation_tokens=180,
            termination_reason='EOS',nfe_evidence='MEASURED_MODEL_FORWARD_COUNTER',
            termination_evidence='MEASURED_TERMINAL_BUFFER')
        self.assertEqual(normalize_native_record(raw),raw)

    def test_test_prompt_uses_no_solution(self):
        for r in rows(ROOT/'manifests/gsm8k_test.jsonl'):
            tail=r['raw_prompt'].rsplit('Question: ',1)[-1]
            self.assertEqual(tail,r['question']+"\nLet's think step by step\nAnswer:")

if __name__=='__main__':
    unittest.main()
