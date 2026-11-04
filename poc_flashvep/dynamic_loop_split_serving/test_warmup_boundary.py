import unittest
from ..fluidtoken_loop_split.split_adapter import static_plan
from .warmup_boundary import startup_plan


class StartupBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(axis='token', depth=32, length=2048, prefix_tokens=1984,
                        startup_relative_token_boundary=True, prime_all_graph_shapes=True)
        self.phase = dict(state='RUNNING', kind='short_D_geometry', seeds=[123])
        self.rid = 'recovery_warm_short_D_geometry_0'

    def test_original_warmup_reproduces(self):
        with self.assertRaisesRegex(ValueError, 'Illegal exact token prefix'):
            static_plan(self.cfg, 512)

    def test_short_startup_only(self):
        self.assertEqual(startup_plan(self.cfg, 512, self.rid, 123, self.phase), (32, 496))
        self.cfg.update(depth=4, length=4096, prefix_tokens=3072)
        self.assertEqual(startup_plan(self.cfg, 512, self.rid, 123, self.phase), (4, 384))

    def test_primary_and_unrelated_short_requests_unchanged(self):
        self.assertEqual(static_plan(self.cfg, 2048), (32, 1984))
        for rid, seed, phase, length in ((self.rid, 123, self.phase, 2048),
                ('measured_0', 123, self.phase, 512), (self.rid, 999, self.phase, 512),
                (self.rid, 123, dict(self.phase, state='COMPLETE'), 512)):
            self.assertIsNone(startup_plan(self.cfg, length, rid, seed, phase))


if __name__ == '__main__':
    unittest.main()
