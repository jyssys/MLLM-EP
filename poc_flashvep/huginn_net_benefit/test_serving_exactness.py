import unittest
from .serving_exactness import differences


class ServingExactnessTests(unittest.TestCase):
    def row(self, seed=1, output=None):
        return dict(seed=seed, input_ids=[7, 8], output_ids=[3, 4] if output is None else output)

    def test_all_tokens_are_compared(self):
        self.assertEqual(differences([self.row()], [self.row(output=[3, 5])]), [1])

    def test_record_order_does_not_change_identity(self):
        rows = [self.row(1), self.row(2)]
        self.assertEqual(differences(rows, rows[::-1]), [])

    def test_missing_request_is_a_failure(self):
        self.assertEqual(differences([self.row()], []), [1])

    def test_duplicate_seed_is_not_silently_overwritten(self):
        with self.assertRaises(ValueError):
            differences([self.row(), self.row()], [self.row()])


if __name__ == '__main__':
    unittest.main()
