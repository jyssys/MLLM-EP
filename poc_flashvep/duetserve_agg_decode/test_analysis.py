"""CPU checks for independent load and budget controls."""
import unittest

from .configuration import offsets
from .statistics import mean_ci, paired_ratio, spearman


class ArrivalTest(unittest.TestCase):
    def test_same_seed_preserves_scaled_poisson_trace(self):
        a = offsets(128, 2.0, 41)
        b = offsets(128, 10.0, 41)
        self.assertEqual(a[0], 0)
        self.assertTrue(all(y > x for x, y in zip(a, a[1:])))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, 5 * y)

    def test_seed_changes_trace(self):
        self.assertNotEqual(offsets(32, 4.0, 41), offsets(32, 4.0, 42))

    def test_no_confidence_interval_from_one_start(self):
        self.assertIsNone(mean_ci([42])['CI95'])
        self.assertIsNone(paired_ratio([4], [2])['CI95'])

    def test_constant_paired_ratio(self):
        result = paired_ratio([4, 6, 8], [2, 3, 4])
        self.assertEqual(result['CI95'], [2, 2])
        self.assertEqual(result['bootstrap_draws'], 27)

    def test_rank_correlation(self):
        self.assertAlmostEqual(spearman([2, 3, 4], [4, 6, 8]), 1)
        self.assertAlmostEqual(spearman([2, 3, 4], [8, 6, 4]), -1)

    def test_preloaded_analysis_uses_new_evidence_root(self):
        from ..dense_ar_slo_layer_pd import analyze as inherited
        from .common import PRIOR, ROOT, bind_previous
        inherited.ROOT = PRIOR
        bind_previous()
        self.assertEqual(inherited.ROOT, ROOT)

    def test_recompute_does_not_create_fictitious_output(self):
        from .interference import decode_output_frontier
        self.assertIsNone(decode_output_frontier(8000, 154, 8198))
        self.assertEqual(decode_output_frontier(8000, 1, 8000), 2)
        self.assertEqual(decode_output_frontier(8154, 44, 8198), 199)

    def test_kernel_union_does_not_sum_stream_overlap(self):
        from .kernel_trace import union, gaps, duration
        busy = union([(0, 4), (2, 6), (8, 12)], 1, 10)
        self.assertEqual(busy, [[1, 6], [8, 10]])
        self.assertEqual(duration(busy), 7)
        self.assertEqual(gaps(busy, 1, 10), [[6, 8]])

    def test_actual_process_ancestry(self):
        import os
        from .ownership_watch import lineage
        self.assertEqual(lineage(os.getpid(), os.getpid())[0]['pid'], os.getpid())
        self.assertIsNone(lineage(os.getpid(), 999999999))

    def test_collector_wrapper_changes_worker_module_only(self):
        from .cupti28_entry import worker_command
        worker = ['python', '-m', 'poc_flashvep.dense_ar_slo_layer_pd.native', 'cfg', '--role', 'decode']
        actual = worker_command(worker)
        self.assertEqual(actual[2], 'poc_flashvep.duetserve_agg_decode.cupti28_entry')
        self.assertEqual(actual[:2] + actual[3:], worker[:2] + worker[3:])
        coordinator = worker[:4]
        self.assertEqual(worker_command(coordinator), coordinator)

    def test_native_TTFT_host_identity(self):
        from .native_ttft_attribution import components
        result = components(10., 10.6, 13.5, 13.75)
        self.assertAlmostEqual(sum(result.values()), 3750.)
        self.assertAlmostEqual(result['source_ready_to_D_socket_end_ms'], 2900.)


if __name__ == '__main__':
    unittest.main()
