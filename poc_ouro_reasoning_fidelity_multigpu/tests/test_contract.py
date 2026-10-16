"""CPU regression tests for reporting denominators and input provenance."""
import unittest
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer,aime_integer

class ContractTests(unittest.TestCase):
    def test_frozen_primary_prompt(self):
        rows=read(ROOT/'results/frozen_questions.json')
        self.assertEqual(len(rows),130)
        self.assertEqual(sum(r['dataset']=='aime24' for r in rows),30)
        for row in rows:
            self.assertEqual(row['arms']['P1']['text'],row['arms']['P0']['text']+'<think>\n')
            self.assertEqual(row['arms']['P1']['ids'][-2:],[3,198])
        self.assertEqual(sha(ROOT/'results/frozen_questions.json'),read(ROOT/'protocol_lock.json')['inputs_sha256'])

    def test_legal_caps(self):
        lock=read(ROOT/'protocol_lock.json')
        self.assertEqual(lock['quality']['caps'],[8192,16384,24576])
        for row in read(ROOT/'results/frozen_questions.json'):
            self.assertLess(len(row['arms']['P1']['ids'])+24576,32769)

    def test_exact_integer_and_wrong_answer(self):
        self.assertEqual(aime_integer(final_answer('\\boxed{007}')[0]),7)
        self.assertNotEqual(aime_integer(final_answer('\\boxed{8}')[0]),7)
        self.assertIsNone(aime_integer(final_answer('We considered 7 during reasoning.')[0]))
        self.assertIsNone(aime_integer('1000'))

    def test_completed_denominator(self):
        records=[('\\boxed{7}','stop'),('\\boxed{7}','length'),('\\boxed{8}','stop')]
        scores=[aime_integer(final_answer(text)[0])==7 and finish=='stop' for text,finish in records]
        self.assertEqual(sum(scores),1)
        self.assertEqual(len(scores),3)

    def test_teacher_positions_and_splits(self):
        inputs=read(ROOT/'results/fidelity_inputs.json')
        self.assertEqual(len(inputs),8)
        self.assertEqual(sum(r['split']=='calibration' for r in inputs),4)
        self.assertEqual(sum(r['split']=='confirmation' for r in inputs),4)
        self.assertTrue(all(len(r['teacher_ids'])==1025 for r in inputs))

if __name__=='__main__':
    unittest.main()
