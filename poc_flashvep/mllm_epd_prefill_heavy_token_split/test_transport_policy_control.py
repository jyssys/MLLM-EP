"""Native EC policy controls must not mutate a live image epoch."""
import unittest
from unittest.mock import patch

from . import transport_policy_control as control


class PolicyContractTest(unittest.TestCase):
    def test_actual_policy_call_stops_before_rpc_when_ownership_is_not_empty(self):
        with patch.object(control,'ownership_empty',side_effect=AssertionError('Live image remains')),\
                patch.object(control,'rpc') as rpc:
            with self.assertRaises(AssertionError):control.policy('unit_policy',False)
            rpc.assert_not_called()

    def test_each_start_has_two_periods_per_policy_in_symmetric_order(self):
        for index in range(3):
            order=control.policy_order(index)
            self.assertEqual(order,list(reversed(order)))
            self.assertEqual(order.count(True),2);self.assertEqual(order.count(False),2)
        self.assertNotEqual(control.policy_order(0),control.policy_order(1))


if __name__=='__main__':unittest.main()
