"""Exercise JSON serialization at the actual task-quality reduction seam."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import quality


class QualityTests(unittest.TestCase):
    def test_reduction_result_is_serializable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for path in (root/'data',root/'raw/test',root/'raw'/quality.GOLD,root/'analysis'):
                path.mkdir(parents=True)
            records=[dict(prompt_id='a',task='hotpotqa',output_ids=[1]),
                     dict(prompt_id='b',task='hotpotqa',output_ids=[2])]
            (root/'data/input.json').write_text(json.dumps(records))
            (root/'raw/test/configuration.json').write_text(json.dumps(dict(dataset='input.json',count=2)))
            (root/'raw/test/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (root/'raw/test/KV_lifetime_single.json').write_text(json.dumps(dict(passed=True)))
            (root/'raw'/quality.GOLD/'natural_reference.json').write_text(json.dumps(dict(rows=records)))
            def score(rows,data):
                return [dict(r,score=100.,text='answer') for r in rows]
            with patch.object(quality,'ROOT',root),patch.object(quality,'scores',score):
                result=quality.quality_gate('test')
            self.assertIs(result['tasks']['hotpotqa']['point_noninferiority'],True)
            self.assertTrue(json.loads((root/'analysis/test_quality_gate.json').read_text())['CI_passed'])

    def test_official_english_normalization(self):
        self.assertEqual(quality.qa_f1('The New-York!','newyork'),1.)
        self.assertAlmostEqual(quality.qa_f1('red blue','blue green'),.5)


if __name__=='__main__':
    unittest.main()
