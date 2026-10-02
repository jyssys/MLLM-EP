import unittest

from poc_native_lp_ep4.scripts.verify_reanalysis import canonical, compare


class ArchiveReanalysisTest(unittest.TestCase):
    def test_only_run_prefix_and_metadata_order_are_normalized(self):
        a = {'rows': [{'run': 'local/run1', 'wave': 2, 'ms': 4.0},
                      {'run': 'local/run1', 'wave': 1, 'ms': 3.0}]}
        b = {'rows': [{'run': 'restored/run1', 'wave': 1, 'ms': 3.0},
                      {'run': 'restored/run1', 'wave': 2, 'ms': 4.0}]}
        self.assertEqual(compare(canonical(a), canonical(b)), 4)

    def test_a_real_numeric_change_cannot_be_hidden(self):
        with self.assertRaises(AssertionError):
            compare(canonical({'rows': [{'arm': 'LP', 'ms': 3.0}]}),
                    canonical({'rows': [{'arm': 'LP', 'ms': 3.1}]}))


if __name__ == '__main__':
    unittest.main()
