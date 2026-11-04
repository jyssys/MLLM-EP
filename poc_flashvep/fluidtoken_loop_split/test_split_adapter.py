import unittest
from .split_adapter import static_plan


class Boundaries(unittest.TestCase):
    def test_large_chunk_does_not_round_away_token_split(self):
        cfg = dict(depth=32, axis='token', chunk=2048, prefix_fraction=.9)
        self.assertEqual(static_plan(cfg, 2048), (32, 1843))
        self.assertEqual(static_plan(dict(cfg, prefix_tokens=1984), 2048), (32, 1984))

    def test_endpoints(self):
        for depth in (4, 32):
            cfg = dict(depth=depth, axis='token', prefix_tokens=0)
            self.assertEqual(static_plan(cfg, 2048), (0, 0))
            self.assertEqual(static_plan(dict(cfg, prefix_tokens=2048), 2048), (depth, 2048))
            self.assertEqual(static_plan(dict(depth=depth, axis='loop', cut=depth-1), 2048), (depth-1, 2048))

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            static_plan(dict(depth=32, axis='token', prefix_tokens=2049), 2048)
        with self.assertRaises(NotImplementedError):
            static_plan(dict(depth=32, axis='loop', cut=16, dynamic=True), 2048)


if __name__ == '__main__':
    unittest.main()
