import unittest
from types import SimpleNamespace
from .decode_priority import choose_decode


class PriorityTests(unittest.TestCase):
    def test_ready_but_empty_does_not_prevent_progress(self):
        calls = []
        owner = SimpleNamespace(has_ready=lambda phase: True,
            _take=lambda phase: calls.append(phase) or ('core' if phase == 'core' else None))
        self.assertEqual(choose_decode(owner, ('prelude', 'coda', 'core')), 'core')
        self.assertEqual(calls, ['prelude', 'coda', 'core'])

    def test_waiting_work_is_not_a_ready_decode(self):
        owner = SimpleNamespace(has_ready=lambda phase: False,
            _take=lambda phase: self.fail('An unready batch was taken'))
        self.assertIsNone(choose_decode(owner, ('prelude', 'coda', 'core')))
