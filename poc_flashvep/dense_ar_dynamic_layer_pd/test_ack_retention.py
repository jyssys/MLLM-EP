import unittest
from .ack_retention import LeaseGuard


class LifetimeTests(unittest.TestCase):
    def test_renew_before_native_expiry(self):
        g=LeaseGuard(60);p={'q':30.};g.retain(p,0);g.retain(p,29)
        self.assertEqual(p['q'],59.);self.assertEqual(len(g.renewals),1)
    def test_actual_ack_removes_tracking(self):
        g=LeaseGuard(60);p={'q':30.};g.retain(p,0);p.clear();g.retain(p,1)
        self.assertEqual(g.seen,{})
    def test_no_silent_free_after_missing_ack(self):
        g=LeaseGuard(30);p={'q':30.};g.retain(p,0)
        with self.assertRaises(AssertionError):g.retain(p,31)
        self.assertIn('q',p)


if __name__=='__main__':unittest.main()
