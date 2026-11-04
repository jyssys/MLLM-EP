import unittest
from .probe_reference import completion_check


class CompletionContract(unittest.TestCase):
    def test_admission_limit_is_not_offered_count(self):
        rows = [dict(request_id=str(i)) for i in range(6)]
        receipt = dict(enqueued=6, completed=6, all_completed=True)
        self.assertTrue(completion_check(rows, rows, receipt, 12))

    def test_missing_duplicate_and_zero_work_fail(self):
        rows = [dict(request_id=str(i)) for i in range(6)]
        receipt = dict(enqueued=6, completed=6, all_completed=True)
        self.assertFalse(completion_check(rows[:-1], rows, receipt, 12))
        self.assertFalse(completion_check(rows + rows[:1], rows, receipt, 12))
        self.assertFalse(completion_check([], [], dict(enqueued=0, completed=0, all_completed=True), 12))
        self.assertFalse(completion_check(rows, rows, dict(receipt, all_completed=False), 12))


if __name__ == '__main__':
    unittest.main()
