"""Keep complete-response SLOs and independent timing lanes distinct."""
import unittest
from ..analyze_serving import completed_response,comparison_stratum

class MetricTests(unittest.TestCase):
    def test_length_cap_is_not_complete_response(self):
        self.assertFalse(completed_response(dict(status='COMPLETE',finish_reason='length')))
        self.assertTrue(completed_response(dict(status='COMPLETE',finish_reason='stop')))
        self.assertFalse(completed_response(dict(status='TIMEOUT',finish_reason='timeout')))

    def test_rates_and_lanes_cannot_pool(self):
        a=dict(split='FINAL',lane='CHARACTERIZATION',workload='MIXED',scenario=dict(rate=.25),configured_N=8)
        b=dict(a,scenario=dict(rate=.9))
        c=dict(a,lane='QUIET')
        self.assertNotEqual(comparison_stratum(a),comparison_stratum(b))
        self.assertNotEqual(comparison_stratum(a),comparison_stratum(c))

if __name__=='__main__':unittest.main()
