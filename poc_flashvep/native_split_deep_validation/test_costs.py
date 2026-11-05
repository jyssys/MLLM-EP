import unittest
from .costs import preparation_enclosures


class PreparationCostTests(unittest.TestCase):
    def test_role_local_ids_join_by_seed_and_exclude_GPU_queue_time(self):
        frontiers = [dict(event='P_FIRST_SUBMISSION', request_id=rid, seed=seed, time=10.)
                     for rid, seed in [('native-a', 1), ('native-b', 2)]]
        device = [dict(request_id=rid, seed=seed, first_host_submit=10.003,
                       first_model_stream_entry=dict(time=10.5))
                  for rid, seed in [('native-a', 1), ('native-b', 2)]]
        rows = preparation_enclosures(frontiers, device, {1, 2}, 9.)
        self.assertEqual(rows[0][0], 2)
        self.assertAlmostEqual(rows[0][1], 3.)

    def test_missing_preparation_is_not_a_zero_cost(self):
        with self.assertRaises(ValueError):
            preparation_enclosures([], [], {1}, 9.)


if __name__ == '__main__':
    unittest.main()
