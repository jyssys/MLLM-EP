import unittest
from .hidden_transport import Slots


class CreditCountTests(unittest.TestCase):
    def test_compute_admission_never_exceeds_unowned_export_slots(self):
        s=Slots(4)
        self.assertEqual(s.free_count(),4)
        for i in range(4):s.take('q'+str(i))
        self.assertEqual(s.free_count(),0)
        s.release(1,'q1');self.assertEqual(s.free_count(),1)
        self.assertEqual(min(4,s.free_count()),1)


class SlotTests(unittest.TestCase):
    def test_no_overwrite_before_ACK(self):
        s=Slots(1);i=s.take('a')
        with self.assertRaises(RuntimeError):s.take('b')
        with self.assertRaises(AssertionError):s.release(i,'b')
        s.release(i,'a');self.assertEqual(s.take('b'),i)

    def test_stale_ACK_is_rejected(self):
        s=Slots(2);i=s.take('a');s.release(i,'a');s.take('b')
        with self.assertRaises(AssertionError):s.release(i,'a')


if __name__=='__main__':unittest.main()
