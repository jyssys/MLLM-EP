"""CPU-only checks for partition, identity, and statistical accounting."""
import unittest
from .common import ROOT, rows
from .audit import wilson
from .postprocess import paired_bootstrap

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

    def test_test_prompt_uses_no_solution(self):
        for r in rows(ROOT/'manifests/gsm8k_test.jsonl'):
            tail=r['raw_prompt'].rsplit('Question: ',1)[-1]
            self.assertEqual(tail,r['question']+"\nLet's think step by step\nAnswer:")

if __name__=='__main__':
    unittest.main()
