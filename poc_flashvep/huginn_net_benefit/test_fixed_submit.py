"""The fixed-depth fast path must not accept adaptive or prompt execution."""
import unittest
from types import SimpleNamespace as S
from .fixed_submit import fixed_contract


class FixedSubmissionTests(unittest.TestCase):
    def request(self, **updates):
        values = dict(min_loops=32, max_loops=32, exit_threshold=1.)
        values.update(updates)
        return S(sampling_params=S(**values))

    def test_fixed_depth_is_eligible(self):
        self.assertTrue(fixed_contract([self.request()], 32))

    def test_threshold_cannot_enable_early_exit(self):
        self.assertFalse(fixed_contract([self.request(exit_threshold=.99)], 32))

    def test_every_request_must_match(self):
        self.assertFalse(fixed_contract([self.request(), self.request(min_loops=1)], 32))

    def test_empty_batch_is_not_eligible(self):
        self.assertFalse(fixed_contract([], 32))


if __name__ == '__main__':
    unittest.main()
