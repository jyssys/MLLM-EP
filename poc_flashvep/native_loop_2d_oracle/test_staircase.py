import unittest
from .staircase import resolve_prefix


class WarmupIdentityTests(unittest.TestCase):
    def test_primary_boundary_stays_absolute(self):
        cfg = dict(length=4096, native_frontier=[0, 3072])
        self.assertEqual(resolve_prefix(cfg, 4096, 'r0_q0', 1, {}), 3072)

    def test_only_original_coordinator_identity_can_scale(self):
        cfg = dict(length=4096, native_frontier=[0, 3072])
        phase = dict(state='RUNNING', kind='short_D_geometry', seeds=[11])
        self.assertEqual(resolve_prefix(cfg, 512, 'recovery_warm_short_D_geometry_0', 11, phase), 384)
        for name, seed in (('a_stage_transfer_uuid', 11), ('r0_q0', 11),
                           ('recovery_warm_short_D_geometry_0', 12)):
            with self.assertRaises(ValueError):
                resolve_prefix(cfg, 512, name, seed, phase)


if __name__ == '__main__':
    unittest.main()
