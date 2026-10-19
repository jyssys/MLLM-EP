import unittest
from poc_ouro_parallelism_deep_discovery.scripts.hybrid_collective_pair_probe import layout,check_mask


class LayoutTest(unittest.TestCase):
    def test_padding_remains_charged(self):
        value=layout(1,4)
        self.assertEqual(value['padded_rows'],4)
        self.assertEqual(value['AR_input_bytes_per_rank_per_call'],4096)
        self.assertEqual(value['AG_output_bytes_per_rank'],16384)
    def test_balanced_rows(self):
        value=layout(8,4)
        self.assertEqual(value['local_rows'],2)
        self.assertEqual(value['padded_rows'],8)
        self.assertEqual(value['RS_input_bytes_per_rank'],32768)
    def test_invalid_world(self):
        with self.assertRaises(ValueError):
            layout(1,8)
    def test_exact_masks(self):
        check_mask([4,5])
        check_mask([4,5,6,7])
        for mask in ([4,6],[5,4],[7,6,5,4],[0,1],[4]):
            with self.assertRaises(ValueError):
                check_mask(mask)


if __name__=='__main__':
    unittest.main()
