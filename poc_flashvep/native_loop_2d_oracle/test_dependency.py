import unittest
from .dependency import Frontier, invalid_checker


class DependencyTests(unittest.TestCase):
    def test_every_small_staircase(self):
        for tokens in range(1, 9):
            for loops in range(1, 5):
                for c in range(loops + 1):
                    for k in range(tokens + 1):
                        f = Frontier(tokens, loops, c, k)
                        owners = {(d, t) for d in range(loops) for t in range(tokens) if f.p_owned(d, t)}
                        self.assertTrue(invalid_checker(owners, loops, tokens))
                        self.assertTrue(f.verify())

    def test_invalid_cross_dependencies(self):
        self.assertFalse(invalid_checker({(1, 0)}, 2, 2))
        self.assertFalse(invalid_checker({(0, 1)}, 2, 2))

    def test_equal_rows_imply_equal_recurrent_kv_not_equal_attention(self):
        token = Frontier(4096, 4, 0, 3072).accounting('ouro')
        loop = Frontier(4096, 4, 3, 0).accounting('ouro')
        self.assertEqual(token['D_recurrent_token_rows'], loop['D_recurrent_token_rows'])
        self.assertEqual(token['recurrent_KV_bytes'], loop['recurrent_KV_bytes'])
        self.assertLess(loop['D_causal_pairs_per_recurrent_attention_layer'],
                        token['D_causal_pairs_per_recurrent_attention_layer'])
        self.assertGreater(loop['minimum_hidden_bytes'], token['minimum_hidden_bytes'])


if __name__ == '__main__':
    unittest.main()
