import unittest
from poc_ouro_parallelism_deep_discovery.scripts.backend_divergence_diagnostic import token_relation, terminal_flags


class Tests(unittest.TestCase):
    def test_exact(self):
        row = token_relation([1, 2], [1, 2])
        self.assertEqual(row['relation'], 'EXACT')
        self.assertEqual(row['common_prefix_tokens'], 2)
        self.assertTrue(row['first_output_agrees'])

    def test_prefix_only_is_not_internal_difference(self):
        row = token_relation([1], [1, 2])
        self.assertEqual(row['relation'], 'PREFIX_ONLY')
        self.assertIsNone(row['first_divergent_token'])

    def test_internal_position_is_zero_based(self):
        row = token_relation([1, 2, 3], [1, 4, 3])
        self.assertEqual(row['relation'], 'INTERNAL_DIVERGENCE')
        self.assertEqual(row['first_divergent_token'], 1)

    def test_first_output_can_differ(self):
        row = token_relation([1], [2])
        self.assertEqual(row['first_divergent_token'], 0)
        self.assertFalse(row['first_output_agrees'])

    def test_empty_sides(self):
        self.assertEqual(token_relation([], [])['relation'], 'EXACT')
        self.assertEqual(token_relation([], [1])['relation'], 'PREFIX_ONLY')

    def test_terminal_eos_is_not_only_a_length_difference(self):
        flags = terminal_flags([1, 2, 3], [1, 0], 0)
        self.assertTrue(flags['first_difference_touches_EOS'])
        self.assertTrue(flags['first_difference_at_shorter_terminal'])
        self.assertEqual(token_relation([1, 2, 3], [1, 0])['relation'], 'INTERNAL_DIVERGENCE')

    def test_early_internal_difference(self):
        flags = terminal_flags([1, 2, 3], [1, 4, 3], 0)
        self.assertFalse(flags['first_difference_touches_EOS'])
        self.assertFalse(flags['first_difference_at_shorter_terminal'])

    def test_terminal_flags_on_empty_inputs(self):
        self.assertFalse(any(terminal_flags([], [], 0).values()))


if __name__ == '__main__':
    unittest.main()
