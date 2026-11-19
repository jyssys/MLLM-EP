"""Require an exact disjoint host-time identity."""
import unittest
from .historical_audit import classify


class GuardAuditTest(unittest.TestCase):
    def test_overlap_is_not_counted_twice(self):
        r=classify(0.,10.,4.,[dict(first_ready=0.,all_ready=1.,start=2.,finish=7.)])
        self.assertAlmostEqual(sum(r.values()),10000.)
        self.assertAlmostEqual(r['own_partner_only'],2000.)
        self.assertAlmostEqual(r['prior_cohort_guard_and_own_partner'],2000.)
        self.assertAlmostEqual(r['prior_cohort_guard_only'],3000.)
        self.assertAlmostEqual(r['eligibility_residual_NOT_INSTRUMENTED'],3000.)

    def test_no_wait_is_zero_not_missing(self):
        self.assertEqual(classify(1.,1.,1.,[]),{})


if __name__=='__main__':
    unittest.main()
