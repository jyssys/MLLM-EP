import unittest
from .analyze import completed_D_layers


class WorkTests(unittest.TestCase):
    def test_unused_native_cut_is_not_migration(self):
        self.assertEqual(completed_D_layers({},dict(layer_cut=28,layer_mode=None)),0)
    def test_fixed_static_contract_fills_legacy_field(self):
        self.assertEqual(completed_D_layers({},dict(layer_cut=34,layer_mode='MIGRATE')),2)
    def test_actual_request_cut_overrides_default(self):
        self.assertEqual(completed_D_layers(dict(completed_D_prefill_layers=12),dict(layer_cut=28,layer_mode='MIGRATE')),12)


if __name__=='__main__':unittest.main()
