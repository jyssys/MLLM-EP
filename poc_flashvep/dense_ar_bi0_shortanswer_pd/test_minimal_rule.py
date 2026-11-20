import unittest
import json
import numpy as np
from .minimal_rule import choose_action,expected_remaining,simulate,same_selection
from .offline_headroom import action_costs


class MinimalRuleTests(unittest.TestCase):
    def test_current_only_and_queue_pressure_can_differ(self):
        costs={'Native0':(.5,.02,.1),'Layer12':(.3,.25,.1)}
        self.assertEqual(choose_action(costs,0.,0,(.5,.02,.1),4)[0],'Native0')
        self.assertEqual(choose_action(costs,0.,4,(.5,.02,.1),4)[0],'Layer12')

    def test_D_pressure_can_reject_a_larger_suffix(self):
        costs={'Native0':(.5,.02,.1),'Layer12':(.3,.25,.1)}
        self.assertEqual(choose_action(costs,2.,4,(.5,.02,.1),4)[0],'Native0')

    def test_output_prior_uses_emitted_count_not_future_EOS(self):
        self.assertEqual(expected_remaining([3,6,12],6),6.)
        self.assertGreater(expected_remaining([3,6,12],20),0.)

    def test_source_submission_cost_is_not_dropped(self):
        model=dict(actions=dict(Native0=dict(curves={
            'P_prepare_to_ready_ms':dict(coefficients_ms=[1000,0,0]),
            'P_serial_submit_to_ready_ms':dict(coefficients_ms=[2000,0,0]),
            'ready_to_first_worker_output_ms':dict(coefficients_ms=[3000,0,0])})))
        self.assertEqual(action_costs(model,dict(input_tokens=8192),3,.01)['Native0'],(2.,3.,.02))

    def test_numpy_arrival_backlog_is_JSON_serializable(self):
        curves={'P_prepare_to_ready_ms':dict(coefficients_ms=[10,0,0]),
            'ready_to_first_worker_output_ms':dict(coefficients_ms=[10,0,0])}
        model=dict(actions=dict(Native0=dict(curves=curves)))
        rows=[dict(input_tokens=8192,task='QA')]*2
        states,actions,observations=simulate(np.array([0.,.005]),rows,
            [dict(Native0=(.01,.01,.01))]*2,model,[1,0,0],{'QA':[2,3]},(.01,.01,.01),1)
        json.dumps(dict(states=states,actions=actions,observations=observations),allow_nan=False)
        self.assertIsInstance(observations[0]['P_visible_backlog'],int)

    def test_frozen_JSON_keys_preserve_the_same_selection(self):
        prior={'QA':[2,3]};scores={0:1.5,4:2.5}
        frozen=json.loads(json.dumps(dict(horizon=4,discovery_scores=scores,task_output_prior=prior)))
        self.assertTrue(same_selection(frozen,4,scores,prior))
        self.assertFalse(same_selection(frozen,0,scores,prior))


if __name__=='__main__':unittest.main()
