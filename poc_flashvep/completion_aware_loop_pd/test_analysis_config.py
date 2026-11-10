import unittest
from .analyze import qualification_config_view


class AnalysisConfigTests(unittest.TestCase):
    def test_native_null_means_full_depth_without_mutation(self):
        cfg=dict(name='sample',native_path=True,feature_native_wire_v5=True,native_partial_cut=None,depth=32)
        self.assertEqual(qualification_config_view(cfg,'sample')['native_partial_cut'],32)
        self.assertIsNone(cfg['native_partial_cut'])

    def test_partial_cut_does_not_change(self):
        cfg=dict(name='sample',native_path=True,feature_native_wire_v5=True,native_partial_cut=24,depth=32)
        self.assertEqual(qualification_config_view(cfg,'sample'),cfg)
