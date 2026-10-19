"""CPU summary checks against immutable captured metadata. No model execution."""
import copy
import json
from pathlib import Path
import unittest

from poc_ouro_parallelism_deep_discovery.scripts.summarize_typed_graph_edges import summarize


class TypedSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.path = cls.root / 'outputs/native_graph_typed_edges_base14_P1_v1/rank0_replayed_graph0_edges.json'
        cls.payload = json.loads(cls.path.read_bytes())
        cls.dot = Path(cls.payload['source_dot']).read_bytes()

    def test_actual_counts_and_default_interior_cuts(self):
        summary, rows = summarize(self.payload, self.dot, 24)
        self.assertEqual((summary['node_count'], summary['edge_count']), (1154, 1153))
        self.assertEqual(summary['metadata_counts'], {'0000000000000000': 1057, '0100010000000000': 96})
        self.assertEqual(summary['reserved_nonzero_count'], 0)
        self.assertEqual(len(rows), 1153)
        self.assertTrue(all(row['source_kernel'] and row['target_kernel'] for row in rows))
        self.assertEqual([cut['on_nondefault_edge'] for cut in summary['cut_analysis']['cuts']],
                         [None, False, False, False, None])
        self.assertIsNone(summary['overlap_duration_ms'])
        self.assertEqual(summary['kernel_overlap'], 'NOT_MEASURED')

    def test_corrupt_named_field_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload['snapshot']['edges'][0]['from_port'] ^= 1
        with self.assertRaisesRegex(ValueError, 'Named edge fields'):
            summarize(payload, self.dot, 24)

    def test_corrupt_raw_byte_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload['snapshot']['edges'][0]['raw_hex'] = '0000000000000001'
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            summarize(payload, self.dot, 24)

    def test_wrong_dot_rejected(self):
        with self.assertRaisesRegex(ValueError, 'DOT hash'):
            summarize(self.payload, self.dot + b'\n', 24)


if __name__ == '__main__':
    unittest.main()
