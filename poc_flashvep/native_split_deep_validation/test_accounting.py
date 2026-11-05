import unittest
from .analyze import ratio
from .granularity import structural_attention


class AccountingTests(unittest.TestCase):
    def test_zero_goodput_is_not_removed(self):
        result = ratio([0., 1., 1.])
        self.assertEqual(result['starts'], 3)
        self.assertEqual(result['geometric_mean_ratio'], 0.)
        self.assertNotIn('ratio_CI95', result)

    def test_equal_rows_have_different_causal_attention(self):
        cfg = dict(model='huginn', length=2048, depth=32)
        loop = structural_attention(dict(cfg, method='loop31', native_partial_cut=31))
        token = structural_attention(dict(cfg, method='token1984', native_frontier=[0, 1984]))
        self.assertEqual(loop['D_core_recurrent_rows'], token['D_core_recurrent_rows'])
        self.assertGreater(token['D_core_causal_pairs_per_core_layer'], loop['D_core_causal_pairs_per_core_layer'])

    def test_2d_respects_the_causal_prefix(self):
        cfg = dict(model='ouro', length=4096, depth=4, method='2d', native_frontier=[3, 3072])
        result = structural_attention(cfg)
        self.assertEqual(result['D_core_recurrent_rows'], 1024)
        self.assertEqual(result['D_core_causal_pairs_per_core_layer'], sum(range(3073, 4097)))


if __name__ == '__main__':
    unittest.main()
