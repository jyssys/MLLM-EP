import unittest
from .structural_work import causal_pairs, dense_work


class StructuralWorkTests(unittest.TestCase):
    def test_causal_suffix_is_not_uniform_token_work(self):
        self.assertEqual(causal_pairs(0, 4), 10)
        self.assertEqual(causal_pairs(2, 2), 7)

    def test_aligned_rows_add_no_padding(self):
        ideal, padded = dense_work(256, 4, 8, 16, True)
        self.assertEqual(ideal, padded)

    def test_only_sensitive_matrices_are_padded(self):
        ideal, padded = dense_work(1, 4, 8, 16, True)
        self.assertEqual(padded-ideal, 2*255*(4*8*16+2*8*8))

    def test_empty_coda_has_no_work(self):
        self.assertEqual(dense_work(0, 2, 8, 16), (0, 0))


if __name__ == '__main__':
    unittest.main()
