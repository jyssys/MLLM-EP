"""CPU regressions for evidence qualification and exact-work controls."""
import unittest
from .token_axis import matched_token_plan, suffix_for, truncate_command


class Contracts(unittest.TestCase):
    def test_exact_work_all_prespecified_cuts(self):
        for layers in (2, 4, 8, 12):
            plan = matched_token_plan(layers, 144)
            self.assertEqual(36 * sum(plan), 144 * 8000 * layers)
            self.assertTrue(all(value % 16 == 0 for value in plan))
            self.assertTrue(all(len(set(plan[i:i + 4])) == 1 for i in range(0, 144, 4)))

    def test_inexact_period_rejected(self):
        with self.assertRaises(AssertionError):
            matched_token_plan(8, 128)

    def test_actual_source_query_partition(self):
        command = dict(kind='add', request_id='q0', input_ids=list(range(8000)), cap=16)
        prepared = truncate_command(command, 8000, 5952)
        self.assertEqual(prepared['input_ids'], list(range(5952)))
        self.assertEqual(command['input_ids'], list(range(8000)))
        self.assertEqual(prepared['cap'], 16)

    def test_bad_boundary_rejected(self):
        command = dict(kind='add', request_id='q0', input_ids=list(range(8000)))
        for prefix in (0, 8000, 7999):
            with self.assertRaises(AssertionError):
                truncate_command(command, 8000, prefix)

    def test_internal_identity_and_warmup(self):
        cfg = dict(token_suffix=2048, token_suffix_by_request=[1776, 1792])
        self.assertEqual(suffix_for(cfg, 'q1-internal'), 1792)
        self.assertEqual(suffix_for(cfg, 'warm_0'), 2048)


if __name__ == '__main__':
    unittest.main()
