import unittest
from .debug_rejected_graph_shapes import preferred


class GraphShapeTests(unittest.TestCase):
    def test_large_token_prefix_is_not_forced_to_full_prompt_padding(self):
        cfg = dict(length=4096, D_token_budget=4096, native_frontier=[0, 3072])
        shapes = preferred(cfg, 'prefill', (1024, 2048, 4096, 8192), 8192, 4)
        self.assertIn(3072, shapes)
        self.assertIn(6144, shapes)
        self.assertEqual(len(shapes), 4)

    def test_small_huginn_token_suffix_has_exact_graph(self):
        cfg = dict(length=2048, D_token_budget=2048, native_frontier=[0, 2016])
        self.assertEqual(preferred(cfg, 'decode', (256, 512, 1024, 2048), 2048, 4),
                         (32, 64, 128, 256))

    def test_2d_uses_no_larger_cache_budget(self):
        cfg = dict(length=4096, D_token_budget=4096, native_frontier=[3, 3072])
        self.assertEqual(preferred(cfg, 'prefill', (1024, 2048, 4096, 8192), 8192, 4),
                         (3072, 4096, 6144, 8192))


if __name__ == '__main__':
    unittest.main()
