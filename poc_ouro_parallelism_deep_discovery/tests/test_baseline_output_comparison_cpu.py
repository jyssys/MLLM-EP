import unittest

from poc_ouro_parallelism_deep_discovery.scripts.baseline_output_comparison import compare_tokens


class TokenComparisonTest(unittest.TestCase):
    def test_identical(self):
        value=compare_tokens([1,2,0],[1,2,0])
        self.assertTrue(value['output_token_ids_equal'])
        self.assertIsNone(value['first_divergent_token'])
        self.assertEqual(value['common_prefix_tokens'],3)
    def test_value_difference(self):
        value=compare_tokens([1,2,0],[1,3,0])
        self.assertEqual(value['first_divergent_token'],1)
    def test_length_difference(self):
        value=compare_tokens([1,2],[1,2,3])
        self.assertFalse(value['output_token_ids_equal'])
        self.assertEqual(value['first_divergent_token'],2)


if __name__=='__main__':
    unittest.main()
