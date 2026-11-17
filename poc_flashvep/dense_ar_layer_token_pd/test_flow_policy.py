"""Check the online surrogate against the state-updating replay."""
import copy
import unittest
from .flow_policy import plan_score
from .replay import evaluate, groups, run
from .test_replay import fixture


class FlowTests(unittest.TestCase):
    def test_chain_agrees_with_replay_when_credits_do_not_bind(self):
        model = fixture()
        arrivals = [0.] * 12
        caps = [16] * 4 + [64] * 4 + [200] * 4
        work = groups(arrivals, caps, 4)
        visible = [dict(caps=g.caps, arrivals=g.arrivals) for g in work]
        for plan in ((0, 0, 0), (12, 0, 4), (2, 8, 12)):
            expected = evaluate(run(model, work, lambda i, _g: plan[i]))
            score, _ = plan_score(visible, plan, 0., 0., model, 'Decode_priority')
            self.assertAlmostEqual(score[0], expected['SLO_output_tokens_s'])
            self.assertAlmostEqual(score[1], expected['raw_output_tokens_s'])

    def test_decode_service_waits_for_the_existing_cohort(self):
        model = fixture()
        score, detail = plan_score([dict(caps=(16,)*4, arrivals=(0.,)*4)],
            (12,), 0., 10., model, 'Decode_priority')
        self.assertGreater(detail['D_finish'], 10.)

    def test_inputs_are_not_mutated_and_cache_is_repeatable(self):
        model = fixture()
        visible = [dict(caps=(16,64,200,16), arrivals=(0.,)*4)]
        before = copy.deepcopy(visible)
        first = plan_score(visible, (8,), 0., 0., model, 'Decode_priority')
        second = plan_score(visible, (8,), 0., 0., model, 'Decode_priority')
        self.assertEqual(visible, before)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
