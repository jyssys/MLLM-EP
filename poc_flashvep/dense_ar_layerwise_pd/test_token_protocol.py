import unittest
from .token_protocol import valid_prefix_blocks,continuation_input


class TokenProtocolTests(unittest.TestCase):
    def test_exact_original_blocks_and_new_destination_reservation(self):
        local=list(range(1000,1501));remote=list(range(500))
        self.assertEqual(valid_prefix_blocks(local,remote,8000,16),local[:500])
        self.assertEqual(len(local),501)

    def test_uncomputed_source_block_is_rejected(self):
        with self.assertRaises(AssertionError):valid_prefix_blocks(list(range(501)),list(range(501)),8000,16)

    def test_short_destination_and_unaligned_prefix_are_rejected(self):
        with self.assertRaises(AssertionError):valid_prefix_blocks(list(range(499)),list(range(500)),8000,16)
        with self.assertRaises(AssertionError):valid_prefix_blocks([0],[0],17,16)

    def test_original_prompt_and_full_output_cap_stay_unchanged(self):
        row=dict(input_ids=dict(qwen8=[10,20]),prompt_id='p0');kv=dict(remote='p')
        extended,cap,params=continuation_input(row,30,16,2,kv)
        self.assertEqual(extended['input_ids']['qwen8'],[10,20,30])
        self.assertEqual(row['input_ids']['qwen8'],[10,20])
        self.assertEqual(cap+1,16);self.assertEqual(params['dense_source_prefix_tokens'],2)
        self.assertNotIn('dense_source_prefix_tokens',kv)


if __name__=='__main__':unittest.main()
