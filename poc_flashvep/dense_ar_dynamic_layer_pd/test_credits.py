"""Exercise stale-cut, cancellation, and reservation accounting failures."""
import unittest
from .credits import Credits


class CreditTests(unittest.TestCase):
    def test_capacity_rejection_does_not_create_a_reservation(self):
        c=Credits();self.assertFalse(c.reserve('a',28,['q0'],501,500));self.assertEqual(c.outstanding(),0)
    def test_future_decode_capacity_stays_reserved(self):
        c=Credits();self.assertFalse(c.reserve('a',28,['q0'],501,600,decode_reserve=100))
    def test_two_reservations_cannot_spend_the_same_blocks(self):
        c=Credits();self.assertTrue(c.reserve('a',28,['q0'],501,1000));self.assertFalse(c.reserve('b',32,['q1'],501,1000))
    def test_cancel_is_legal_only_before_commit(self):
        c=Credits();c.reserve('a',28,['q0'],501,1000);c.cancel('a');self.assertEqual(c.outstanding(),0)
        c.reserve('b',28,['q1'],501,1000);c.commit('b')
        with self.assertRaises(AssertionError):c.cancel('b')
    def test_stale_cut_cannot_import(self):
        c=Credits();c.reserve('a',28,['q0'],501,1000);c.commit('a')
        with self.assertRaises(AssertionError):c.validate('a','q0',32)
    def test_physical_allocation_releases_only_its_credit(self):
        c=Credits();c.reserve('a',28,['q0','q1'],501,2000);c.commit('a');c.map_request('internal0','q0')
        c.allocated('internal0');self.assertEqual(c.outstanding(),501);c.finish('q0');self.assertIn('a',c.reservations)
    def test_all_completions_retire_the_generation(self):
        c=Credits();c.reserve('a',24,['q0'],501,1000);c.commit('a');c.map_request('i','q0');c.allocated('i');c.finish('q0')
        self.assertFalse(c.reservations)


if __name__=='__main__':unittest.main()
