import unittest
from unittest.mock import patch
from .oracle import search_family


class OracleEndpointTests(unittest.TestCase):
    def test_every_family_can_use_the_zero_offload_endpoint(self):
        seen = []

        class FakeSimulation:
            def __init__(self, model, arrivals, choices, slo):
                self.choices = choices
                seen.extend(choices)

            def run(self):
                return dict(metrics=dict(graph_budget_screen=True, Decode_SLO_screen=True,
                    joint_SLO_goodput=self.choices.count('zero'), TTFT_p95_ms=1.))

        with patch('poc_flashvep.native_split_deep_validation.oracle.Simulation', FakeSimulation):
            result = search_family(dict(actions=dict(native={}, zero={}, token={})),
                                   [0.] * 8, {}, ['token'])
        self.assertIn('zero', seen)
        self.assertEqual(result['choices'], ['zero'] * 8)


if __name__ == '__main__':
    unittest.main()
