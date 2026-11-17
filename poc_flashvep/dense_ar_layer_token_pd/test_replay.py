"""Small exact signals for replay transitions and bounded oracle search."""
import itertools
import unittest
from .replay import Group, Model, State, evaluate, groups, oracle, run


def fixture():
    tables = {name: {} for name in ('P_source_ms', 'D_prepare_ms', 'negotiation_ms', 'decode_step_ms')}
    for batch in (1, 2, 3, 4):
        tables['decode_step_ms'][str(batch)] = dict(p50=2.)
        for layers in (0, 2, 4, 8, 12):
            key = f'{batch}:{layers}'
            tables['P_source_ms'][key] = dict(p50=batch * (36 - layers))
            tables['D_prepare_ms'][key] = dict(p50=1 + batch * layers)
            tables['negotiation_ms'][key] = dict(p50=0.)
    slo = dict(output_caps={str(cap): {'Decode_priority': dict(TTFT_ms=10000, ITL_ms=10, TPOT_ms=10)} for cap in (16, 64, 200)})
    legacy = dict(handoff_ms={str(b): {str(m): 1. for m in (0, 2, 4, 8, 12)} for b in (1, 2, 3, 4)})
    return Model(dict(tables=tables), legacy, slo)


class ReplayTests(unittest.TestCase):
    def test_same_step_repeats_exactly(self):
        model = fixture()
        g = Group((0, 1, 2, 3), (0.,) * 4, (16,) * 4)
        a, b = model.advance(State(), g, 8), model.advance(State(), g, 8)
        self.assertEqual(a, b)
        self.assertEqual(State().rows, [])

    def test_source_and_destination_dependencies(self):
        model = fixture()
        work = groups([0.] * 8, [200] * 8, 4)
        state = run(model, work, lambda _i, _g: 12)
        self.assertGreaterEqual(state.decisions[1]['P_start'], state.decisions[0]['P_ready'])
        self.assertGreaterEqual(state.decisions[1]['D_start'], state.decisions[0]['D_done'])
        self.assertGreaterEqual(state.decisions[1]['D_start'], state.decisions[1]['P_ready'])

    def test_no_new_work_preempts_existing_decode(self):
        model = fixture()
        first = model.advance(State(), groups([0.] * 4, [200] * 4, 4)[0], 0)
        second = model.advance(first, Group((4, 5, 6, 7), (0.,) * 4, (16,) * 4), 12)
        self.assertGreaterEqual(second.decisions[-1]['D_start'], first.D)

    def test_small_oracle_matches_exhaustive_search(self):
        model = fixture()
        work = groups([0.] * 12, [16] * 4 + [64] * 4 + [16] * 4, 4)
        expected = max(evaluate(run(model, work, lambda i, _g, p=plan: p[i]))['SLO_output_tokens_s']
                       for plan in itertools.product((0, 2, 4, 8, 12), repeat=3))
        actual = evaluate(oracle(model, work, width=1024))['SLO_output_tokens_s']
        self.assertAlmostEqual(actual, expected)

    def test_credit_fallback_is_state_updating(self):
        model = fixture()
        model.hidden_slots = 0
        state = model.advance(State(), groups([0.] * 4, [16] * 4, 4)[0], 12)
        self.assertEqual(state.decisions[0]['layers'], 0)
        self.assertEqual(state.decisions[0]['reason'], 'CONSERVATIVE_CREDIT_NATIVE_FALLBACK')


if __name__ == '__main__':
    unittest.main()
