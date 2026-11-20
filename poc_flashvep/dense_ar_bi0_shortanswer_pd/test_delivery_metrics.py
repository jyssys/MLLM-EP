import unittest
from .metrics import delivery_intervals


class DeliveryTests(unittest.TestCase):
    def test_first_coalesced_event(self):
        values=delivery_intervals([dict(count=2,time=0.),dict(count=4,time=.01)],4)
        self.assertEqual(values,[0.,10.,0.])

    def test_one_token(self):
        self.assertEqual(delivery_intervals([dict(count=1,time=3.)],1),[])

    def test_missing_emission_is_rejected(self):
        with self.assertRaises(AssertionError):
            delivery_intervals([dict(count=1,time=0.),dict(count=2,time=1.)],3)


if __name__=='__main__':unittest.main()
