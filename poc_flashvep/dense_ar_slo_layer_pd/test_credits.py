"""Keep prospective macro credits and exact ownership independent."""
import unittest
from .credits import Credits


class CreditTests(unittest.TestCase):
    def test_real_macro_credit(self):
        c=Credits();ids=[f'q{i}' for i in range(16)]
        self.assertTrue(c.reserve('g1',28,ids,501,10905,decode_reserve=208,max_pending=32))
        c.commit('g1')
        for i,rid in enumerate(ids):
            c.map_request(f'internal{i}',rid);c.validate('g1',rid,28);c.allocated(f'internal{i}')
        self.assertEqual(c.outstanding(),0)
        for rid in ids:c.finish(rid)
        self.assertEqual(c.reservations,{})
    def test_insufficient_KV_is_not_a_partial_reservation(self):
        c=Credits();self.assertFalse(c.reserve('g',28,['q0','q1'],501,1001))
        self.assertEqual(c.reservations,{})
    def test_no_postcommit_cancellation(self):
        c=Credits();c.reserve('g',28,['q0'],501,2000);c.commit('g')
        with self.assertRaises(AssertionError):c.cancel('g')
    def test_precommit_cancellation_returns_credits(self):
        c=Credits();c.reserve('g',28,['q0'],501,2000);c.cancel('g')
        self.assertEqual(c.outstanding(),0)
    def test_wrong_cut_cannot_import(self):
        c=Credits();c.reserve('g',28,['q0'],501,2000);c.commit('g')
        with self.assertRaises(AssertionError):c.validate('g','q0',24)


if __name__=='__main__':unittest.main()
