import unittest
from .drained_recovery import Retry


class DrainedRetryTest(unittest.TestCase):
    def test_cooldown_and_ownership(self):
        state = Retry()
        state.miss(4, 10.)
        self.assertEqual(state.ceiling, 2)
        reason = 'COMPLETION_RESERVATION_MISSED'
        self.assertFalse(state.ready(10.5, 700., reason, [], set(), {}, False))
        self.assertTrue(state.ready(10.8, 700., reason, [], set(), {}, False))
        self.assertFalse(state.ready(10.8, 700., reason, [1], set(), {}, False))
        self.assertFalse(state.ready(10.8, 700., reason, [], {1}, {}, False))
        self.assertFalse(state.ready(10.8, 700., reason, [], set(), {1: 2}, False))
        self.assertFalse(state.ready(10.8, 700., reason, [], set(), {}, True))

    def test_one_retry_and_no_unknown_recovery(self):
        state = Retry()
        state.miss(256, 10.)
        self.assertEqual(state.ceiling, 128)
        self.assertFalse(state.ready(11., 700., 'DECODE_TAIL', [], set(), {}, False))
        state.attempts = 1
        self.assertFalse(state.ready(11., 700., 'COMPLETION_RESERVATION_MISSED', [], set(), {}, False))
        state.miss(1, 11.)
        self.assertEqual(state.ceiling, 0)


if __name__ == '__main__':
    unittest.main()
