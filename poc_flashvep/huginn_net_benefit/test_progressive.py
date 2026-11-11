"""Exercise exact early/final plane accounting before GPU use."""
import unittest
from .progressive import delta_layout, disjoint_publication


class ProgressiveTests(unittest.TestCase):
    def test_full_frontier_preserves_boundary_planes(self):
        published = {}
        early = [(0, 0, 2048), (0, 1, 2048), (1, 1, 2048)]
        disjoint_publication(early, published)
        final = early + [(0, 2, 2047), (2, 1, 2048)]
        self.assertEqual(delta_layout(final, published), [(0, 2, 2047), (2, 1, 2048)])

    def test_immediate_cut_has_no_duplicate_kv(self):
        early = [(0, 0, 2048), (0, 1, 2048)]
        published = {}
        disjoint_publication(early, published)
        self.assertEqual(delta_layout(early, published), [])

    def test_earlier_final_cut_is_invalid(self):
        with self.assertRaisesRegex(ValueError, 'excludes'):
            delta_layout([(0, 1, 2048)], {(1, 1): 2048})

    def test_changed_valid_range_is_invalid(self):
        with self.assertRaisesRegex(ValueError, 'changed'):
            delta_layout([(0, 1, 2049)], {(0, 1): 2048})

    def test_duplicate_publication_is_invalid(self):
        with self.assertRaisesRegex(ValueError, 'twice'):
            disjoint_publication([(0, 1, 2048)], {(0, 1): 2048})


if __name__ == '__main__':
    unittest.main()
