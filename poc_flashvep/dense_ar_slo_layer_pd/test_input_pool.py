"""Do not confuse storage reuse with prefix-cache or computation reuse."""
import unittest
from .input_pool import intern


class InputPoolTests(unittest.TestCase):
    def test_same_tokens_and_distinct_request_rows(self):
        rows=[dict(prompt_id='a',input_ids={'qwen8':[1,2]},index=i) for i in range(4)]
        out=intern(rows)
        self.assertEqual(out,rows)
        self.assertIsNot(out[0],out[1])
        self.assertIs(out[0]['input_ids']['qwen8'],out[3]['input_ids']['qwen8'])
        self.assertEqual([r['index'] for r in out],list(range(4)))
    def test_different_prompts_remain_different(self):
        rows=[dict(prompt_id=p,input_ids={'qwen8':[1,2]}) for p in ['a','b']]
        out=intern(rows)
        self.assertIsNot(out[0]['input_ids']['qwen8'],out[1]['input_ids']['qwen8'])
    def test_inconsistent_source_id_is_an_error(self):
        rows=[dict(prompt_id='a',input_ids={'qwen8':ids}) for ids in ([1,2],[1,3])]
        with self.assertRaises(AssertionError):intern(rows)


if __name__=='__main__':unittest.main()
