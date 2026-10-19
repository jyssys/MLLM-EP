import unittest

from poc_ouro_parallelism_deep_discovery.scripts.depth_affinity_extension_screen import convert
from poc_ouro_parallelism_deep_discovery.scripts.depth_affinity_capacity_screen import visits,screen


class ExtensionScreenTests(unittest.TestCase):
    def test_counter_identity(self):
        row=convert(dict(recorded_decode_positions=6,R2_positions=1,R3_positions=3,R4_positions=2,
                         recorded_decode_core_pass_work=19))
        self.assertEqual(visits([row]),[1,1,5/6,2/6])
        self.assertLessEqual(screen(visits([row]),4)['pinned_over_pooled_saturated_capacity'],1)
    def test_bad_counter_identity_rejects(self):
        row=convert(dict(recorded_decode_positions=6,R2_positions=1,R3_positions=3,R4_positions=2,
                         recorded_decode_core_pass_work=20))
        with self.assertRaises(AssertionError):
            visits([row])


if __name__=='__main__':
    unittest.main()
