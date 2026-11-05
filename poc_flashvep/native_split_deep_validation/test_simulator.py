import copy
import unittest
from .simulator import Simulation


def curve(one, two):
    return {'1': dict(median=one), '2': dict(median=two)}


def fixture():
    action = dict(P_steps={'prelude': curve(10., 18.)}, D_prompt={},
                  handoff_ms=dict(median=3.), ACK_lag_ms=dict(median=1.),
                  serial_TPOT_ms=dict(median=6.), serial_service_ms=dict(median=14.),
                  P_window_gap_ms=dict(median=0.), D_window_gap_ms=dict(median=0.))
    partial = copy.deepcopy(action)
    partial.update(P_steps={'prelude': curve(8., 15.)}, D_prompt=curve(3., 5.), D_max_batch=2)
    return dict(actions=dict(native=action, loop1=partial), depth=2, cap=3, P_slots=2,
                D_slots=2, D_credit_limit=3,
                decode=dict(prelude=curve(1., 1.), recurrent=curve(2., 2.), coda=curve(1., 1.)))


class SimulatorTests(unittest.TestCase):
    def run_model(self, methods):
        return Simulation(fixture(), [0.] * len(methods), methods,
                          dict(TTFT_ms=1000., TPOT_ms=100., ITL_ms=100.)).run()

    def test_source_ack_and_causal_handoff_are_not_free(self):
        result = self.run_model(['native'] * 8)
        self.assertEqual(len(result['requests']), 8)
        for request in result['requests']:
            self.assertEqual(len(request['tokens']), 3)
            self.assertGreaterEqual(request['start'], request['arrival'])
            self.assertGreater(request['D_received'], request['P_completed'])
            self.assertGreater(request['source_ACK'], request['D_received'])
            self.assertGreaterEqual(request['P_credit_released'], request['source_ACK'])
            self.assertGreater(request['tokens'][0], request['D_received'])

    def test_mixed_actions_complete_all_output_work(self):
        result = self.run_model(['native', 'loop1'] * 4)
        self.assertTrue(all(row['finished'] for row in result['requests']))
        self.assertEqual(len(result['requests']), 8)

    def test_unmeasured_actions_cannot_enter(self):
        with self.assertRaises(ValueError):
            self.run_model(['unmeasured'])

    def test_source_credit_does_not_retire_at_an_earlier_ACK(self):
        model = fixture()
        model['actions']['native']['P_credit_release_lag_ms'] = dict(median=9.)
        result = Simulation(model, [0.] * 6, ['native'] * 6,
                            dict(TTFT_ms=1000., TPOT_ms=100., ITL_ms=100.)).run()
        for request in result['requests']:
            self.assertGreater(request['P_credit_released'], request['source_ACK'])

    def test_graph_budget_is_a_necessary_geometry_screen(self):
        model = fixture()
        model['graph_budget'] = 1
        model['actions']['native']['P_query_rows'] = {'prelude': 4}
        model['actions']['loop1'].update(P_query_rows={'prelude': 3}, D_query_rows=4)
        result = Simulation(model, [0., 100.], ['native', 'loop1'],
                            dict(TTFT_ms=1000., TPOT_ms=100., ITL_ms=100.)).run()
        self.assertFalse(result['metrics']['graph_budget_screen'])
        self.assertEqual(result['metrics']['graph_keys_per_role']['prefill'], 2)
        self.assertEqual(result['metrics']['graph_keys_per_role']['decode'], 1)

    def test_missing_graph_geometry_is_not_a_pass(self):
        result = self.run_model(['native'])
        self.assertIsNone(result['metrics']['graph_budget_screen'])

    def test_input_preparation_delays_actual_model_start_not_service_start(self):
        model = fixture()
        model['actions']['native']['P_prepare'] = curve(7., 9.)
        result = Simulation(model, [0.], ['native'],
                            dict(TTFT_ms=1000., TPOT_ms=100., ITL_ms=100.)).run()
        self.assertEqual(result['requests'][0]['start'], 7.)
        self.assertEqual(result['metrics']['queue_p95_ms'], 7.)


if __name__ == '__main__':
    unittest.main()
