import unittest
from poc_ouro_parallelism_deep_discovery.scripts.adaptive_alias_capacity_screen import payload_model, sealed_page_model


class Tests(unittest.TestCase):
    def test_fixed_has_no_saving(self):
        row = payload_model(32, [4, 4], 24)
        self.assertEqual(row['ideal_payload_saving_percent'], 0)

    def test_loaded_per_depth_geometry(self):
        self.assertEqual(payload_model(1, [3], 24)['bytes_per_depth_position'], 196608)
        self.assertEqual(payload_model(1, [3], 48)['bytes_per_depth_position'], 393216)

    def test_prompt_remains_full_depth(self):
        row = payload_model(1, [2], 24)
        self.assertEqual(row['full_depth_payload_bytes'], 8*196608)
        self.assertEqual(row['ideal_distinct_payload_bytes'], 6*196608)
        self.assertEqual(row['ideal_payload_saving_percent'], 25)
        self.assertEqual(row['decode_only_payload_saving_percent'], 50)

    def test_long_prefix_dilutes_decode_saving(self):
        self.assertLess(payload_model(1000, [2], 24)['ideal_payload_saving_percent'], 0.05)

    def test_invalid_geometry_or_depth(self):
        for args in [(0, [3], 24), (1, [], 24), (1, [1], 24), (1, [3], 12)]:
            with self.assertRaises(ValueError):
                payload_model(*args)

    def test_only_sealed_uniform_source_pages_alias(self):
        self.assertEqual(sealed_page_model(2, [3, 3], 2)['aliased_sealed_depth_pages'], 1)
        self.assertEqual(sealed_page_model(2, [3, 4], 2)['aliased_sealed_depth_pages'], 0)
        self.assertEqual(sealed_page_model(2, [2, 2], 2)['aliased_sealed_depth_pages'], 2)
        self.assertEqual(sealed_page_model(2, [3], 2)['aliased_sealed_depth_pages'], 0)


if __name__ == '__main__':
    unittest.main()
