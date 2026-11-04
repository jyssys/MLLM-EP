import unittest
from .split_accounting import account, fixed_projection_rows


class StructuralAccounting(unittest.TestCase):
    def test_matched_rows_do_not_match_attention(self):
        x = account(4096, 4, 3, 3072)
        self.assertEqual(x['loop_D_rows'], x['token_D_rows'])
        self.assertGreater(x['token_D_pairs'], x['loop_D_pairs'])

    def test_full_and_local_endpoints(self):
        for length, depth in ((2048, 32), (4096, 4)):
            full = account(length, depth, depth, length)
            self.assertEqual(full['loop_D_rows'] + full['token_D_rows'], 0)
            local = account(length, depth, 0, 0)
            self.assertEqual(local['loop_D_pairs'], local['full_causal_attention_pairs'])
            self.assertEqual(local['token_D_pairs'], local['full_causal_attention_pairs'])

    def test_numerical_padding_is_not_loop_novelty(self):
        small = fixed_projection_rows(2048, 32, 31, 1984)
        self.assertEqual((small['loop_rows'], small['token_rows']), (2048, 8192))
        aligned = fixed_projection_rows(2048, 32, 28, 1792)
        self.assertEqual((aligned['loop_rows'], aligned['token_rows']), (8192, 8192))


if __name__ == '__main__':
    unittest.main()
