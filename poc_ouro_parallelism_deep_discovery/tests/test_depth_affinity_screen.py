"""CPU model checks. No GPU inference occurs."""
import unittest
from poc_ouro_parallelism_deep_discovery.scripts.depth_affinity_capacity_screen import screen, visits


class AffinityTests(unittest.TestCase):
    def test_fixed_balanced(self):
        for P in [2, 4]:
            self.assertEqual(screen([1, 1, 1, 1], P)['pinned_over_pooled_saturated_capacity'], 1)

    def test_R3_imbalance(self):
        self.assertEqual(screen([1, 1, 1, 0], 4)['pinned_over_pooled_saturated_capacity'], .75)
        self.assertEqual(screen([1, 1, 1, 0], 2)['pinned_over_pooled_saturated_capacity'], .75)

    def test_low_population(self):
        row = screen([1, 1, 1, 0], 4)
        self.assertEqual(row['finite_C'][0]['pinned_over_pooled_capacity'], 1)

    def test_only_four_depth_tasks(self):
        self.assertEqual(screen([1, 1, 1, 1], 8)['pinned_over_pooled_saturated_capacity'], .5)

    def test_visit_accounting(self):
        self.assertEqual(visits([dict(N_decode_events=2, R3_events=1, R4_events=1,
                                     executed_depth_steps=7)]), [1, 1, 1, .5])


if __name__ == '__main__':
    unittest.main()
