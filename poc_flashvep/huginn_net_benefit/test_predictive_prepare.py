import unittest
from .predictive_prepare import preparation_advertisement as prepare


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.ad = dict(observed=10., available=True, budget_ms=0.,
            future_quantum_budget_ms=3., known_decode_drain_forecast_ms=100.)
        self.cfg = dict(empty_quantum_cap_ms=200.)

    def test_forecast_is_a_new_object_and_does_not_change_admission(self):
        result = prepare(self.ad, 10.01, 300., self.cfg)
        self.assertIsNot(result, self.ad)
        self.assertTrue(result['checkpoint_preparation_forecast_only'])
        self.assertEqual(self.ad['budget_ms'], 0.)

    def test_unknown_or_late_drain_does_not_expose_checkpoint(self):
        self.assertIs(prepare(self.ad, 10.01, 50., self.cfg), self.ad)
        self.assertIs(prepare(dict(self.ad, known_decode_drain_forecast_ms=None),
            10.01, 300., self.cfg).get('known_decode_drain_forecast_ms'), None)

    def test_stale_unavailable_and_backward_advertisements(self):
        self.assertIs(prepare(self.ad, 10.1, 300., self.cfg), self.ad)
        self.assertIs(prepare(self.ad, 9.99, 300., self.cfg), self.ad)
        unavailable = dict(self.ad, available=False)
        self.assertIs(prepare(unavailable, 10.01, 300., self.cfg), unavailable)


if __name__ == '__main__':
    unittest.main()
