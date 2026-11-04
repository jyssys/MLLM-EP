import unittest
from .metrics import individual_slo_attainment, disjoint_latency_shares


class IndividualSLOTests(unittest.TestCase):
    def test_missing_requests_remain_in_each_offered_denominator(self):
        rows = [dict(ttft_ms=10, tpot_ms=30), dict(ttft_ms=40, tpot_ms=5)]
        result = individual_slo_attainment(rows, 4, dict(TTFT_ms=20, TPOT_ms=20))
        self.assertEqual(result['TTFT_SLO_pass_count'], 1)
        self.assertEqual(result['TPOT_SLO_pass_count'], 1)
        self.assertEqual(result['TTFT_SLO_attainment'], .25)
        self.assertEqual(result['TPOT_SLO_attainment'], .25)
        self.assertEqual(result['SLO_attainment_denominator'], 4)

    def test_equal_deadlines_pass_and_invalid_populations_fail(self):
        row = dict(ttft_ms=20, tpot_ms=20)
        self.assertEqual(individual_slo_attainment([row], 1,
            dict(TTFT_ms=20, TPOT_ms=20))['TTFT_SLO_attainment'], 1)
        with self.assertRaises(ValueError):
            individual_slo_attainment([row], 0, dict(TTFT_ms=20, TPOT_ms=20))

    def test_component_shares_use_summed_request_latency(self):
        records = [dict(queue_ms=200, service_TTFT_ms=300),
                   dict(queue_ms=100, service_TTFT_ms=400)]
        outputs = [dict(arrival=1, finish=2), dict(arrival=3, finish=6)]
        row = disjoint_latency_shares(records, outputs)
        self.assertEqual(row['summed_request_queue_E2E_percent'], 7.5)
        self.assertEqual(row['summed_request_service_TTFT_E2E_percent'], 17.5)
        self.assertEqual(row['summed_request_postfirst_Decode_E2E_percent'], 75)
        with self.assertRaises(ValueError):
            disjoint_latency_shares(records, [dict(arrival=1, finish=1.1)])


if __name__ == '__main__':
    unittest.main()
