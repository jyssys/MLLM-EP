"""Test the prospective matched-batch design without a GPU."""
import unittest
from .batch_matched import matrix, state_configuration
from .configuration import destination_limit


class BatchMatchedTests(unittest.TestCase):
    def test_every_actual_admission_limit_is_one(self):
        values = matrix()
        self.assertEqual(len(values), 98)
        self.assertEqual(len({v['name'] for v in values}), 98)
        for cfg in values:
            self.assertEqual(cfg['P_group_size'], 1)
            self.assertEqual(cfg['D_admission_batch'], 1)
            self.assertEqual(cfg['token_budget'], 8192)
            self.assertEqual(cfg['D_token_budget'], 8192)
            self.assertEqual(cfg['P_slots'], 1024)
            self.assertEqual(cfg['D_slots'], 1024)
            self.assertTrue(cfg['common_prospective_KV_reservation'])
            self.assertEqual(destination_limit(list(range(8)), cfg, 1), 1)
            if cfg['evaluation_arm'] == 'Token':
                self.assertEqual(sum(cfg['token_suffix_by_request'])*36,
                                 cfg['count']*8000*cfg['static_layers'])

    def test_same_trace_in_each_comparison(self):
        grouped = {}
        for cfg in matrix():
            key = (cfg['matrix_stage'], cfg['count'], cfg['cap'], cfg['arrival_rate'], cfg['arrival_seed'])
            fields = {f:cfg[f] for f in ('arrival_offsets', 'request_caps', 'dataset', 'gpus')}
            self.assertEqual(grouped.setdefault(key, fields), fields)

    def test_new_full_state_gate_uses_the_same_small_batches(self):
        for method, layers in [('Native', 0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]:
            cfg = state_configuration(method, layers)
            self.assertTrue(cfg['correctness_only'])
            self.assertEqual(cfg['count'], 18)
            self.assertEqual(cfg['P_group_size'], cfg['D_admission_batch'], 1)
            self.assertEqual(cfg['P_group_size'], 1)
            if method == 'Token':
                self.assertEqual(sum(cfg['token_suffix_by_request'])*36, 18*8000*layers)


if __name__ == '__main__':
    unittest.main()
