import os
import unittest
from unittest.mock import patch
from poc_ouro_parallelism_deep_discovery.process_lease_diagnostics import snapshot


class LeaseDiagnosticsTests(unittest.TestCase):
    def test_only_allowlisted_values_are_compared_not_exported(self):
        expected = dict(LOOPED_KV_OUTPUT='CPU_PUBLIC_GUARD', CUDA_VISIBLE_DEVICES='4,5',
                        VLLM_WORKER_MULTIPROC_METHOD='spawn')
        with patch.dict(os.environ, dict(expected, DUMMY_PRIVATE_VALUE='DO_NOT_EXPORT')):
            row = snapshot(os.getpid(), expected, 'CPU_SELF')
        self.assertTrue(all(value['matches'] for value in row['logical_environment'].values()))
        self.assertNotIn('DO_NOT_EXPORT', str(row))
        self.assertNotIn('DUMMY_PRIVATE_VALUE', str(row))
        self.assertNotIn('CPU_PUBLIC_GUARD', str(row))
        self.assertEqual(row['ppid'], os.getppid())
        self.assertGreater(row['start_time_ticks'], 0)

    def test_extra_key_is_rejected(self):
        with self.assertRaises(ValueError):
            snapshot(os.getpid(), {'ANY_PRIVATE_KEY': 'not allowed'}, 'CPU_SELF')

    def test_string_extension_retains_the_allowlisted_snapshot(self):
        from poc_ouro_parallelism_deep_discovery.hybrid_cohort_probe import HybridCohortProbe
        expected = dict(LOOPED_KV_OUTPUT='CPU_STRING_GUARD', CUDA_VISIBLE_DEVICES='4,5',
                        VLLM_WORKER_MULTIPROC_METHOD='spawn')
        with patch.dict(os.environ, expected):
            result = HybridCohortProbe().inspect_guard_identity(expected)
        self.assertEqual(result['role'], 'ACTUAL_TP_RANK_RPC_SELF')
        self.assertTrue(all(v['matches'] for v in result['logical_environment'].values()))


if __name__ == '__main__':
    unittest.main()
