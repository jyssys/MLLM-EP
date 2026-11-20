import unittest
from .oracle_audit import rectify,PROXY


class OracleAuditTests(unittest.TestCase):
    def test_feasible_subset_is_preserved_without_mutating_original(self):
        def item(value):return dict(metrics={PROXY:value},states=[dict(P_start=0.)])
        original=dict(oracles=dict(Layer=item(10.),Token=item(12.),All=item(11.)),
            best_fixed_per_cell_hindsight=item(12.),discovery_frozen_fixed=item(6.))
        value,changed=rectify(original)
        self.assertTrue(changed)
        self.assertEqual(value['axis_additional_proxy_headroom'],0.)
        self.assertEqual(value['over_hindsight_fixed'],1.)
        self.assertEqual(value['over_discovery_frozen_fixed'],2.)
        self.assertEqual(original['oracles']['All']['metrics'][PROXY],11.)
