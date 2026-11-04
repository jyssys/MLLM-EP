import unittest
from .lifetime_gate import evaluate


class LifetimeTests(unittest.TestCase):
    def setUp(self):
        self.closed = dict(metrics=[dict(request_id='r0_q0', transfer_id='x', cancelled=False)])
        self.events = [dict(tid='x', event='COORDINATOR_released', role=role) for role in ('prefill', 'decode')]
        self.owners = [dict(tid='x', role='prefill', event='NATIVE_ACK_RECEIVED', acknowledged=True, time=1.),
                       dict(tid='x', role='prefill', event='NATIVE_CONNECTOR_FORGET', allocation_after=False, time=2.)]
        self.contracts = [dict(role=role, used_KV_blocks=0, prefix_caching_enabled=False, reused_prefix_tokens=0)
                          for role in ('prefill', 'decode')]

    def result(self):
        return evaluate(self.closed, self.events, self.owners, self.contracts, 1)

    def test_valid_retirement(self):
        self.assertTrue(self.result()['passed'])

    def test_early_retirement_and_leaks_fail(self):
        self.owners[1]['time'] = .5
        self.assertFalse(self.result()['passed'])
        self.owners[1]['time'] = 2.
        self.contracts[1]['used_KV_blocks'] = 1
        self.assertFalse(self.result()['passed'])

    def test_missing_credit_release_fails(self):
        self.events.pop()
        self.assertFalse(self.result()['passed'])


if __name__ == '__main__':
    unittest.main()
