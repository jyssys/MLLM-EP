import unittest
from .probe_quantum import OutputCredit


class CreditTests(unittest.TestCase):
    def test_no_accumulated_future_budget(self):
        value = OutputCredit()
        value.refresh()
        value.refresh()
        value.consume(True)
        self.assertFalse(value.allows(True))
        with self.assertRaises(RuntimeError):
            value.consume(True)
        value.refresh()
        self.assertTrue(value.allows(True))

    def test_no_background_can_drain(self):
        value = OutputCredit()
        for _ in range(32):
            value.consume(False)
        self.assertEqual(value.submissions, 32)
        self.assertTrue(value.allows(False))


if __name__ == '__main__':
    unittest.main()
