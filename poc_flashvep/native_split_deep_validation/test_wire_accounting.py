import unittest
from .wire_accounting import classify


class WireAccountingTests(unittest.TestCase):
    def test_KV_hidden_and_descriptor_counts_are_exact(self):
        value = classify([(110, 20, 0), (300, 5, 0)],
                         [(100, 200, 'KV_bytes'), (300, 400, 'hidden_bytes')], 25)
        self.assertEqual(value, dict(KV_bytes=20, hidden_bytes=5, descriptors=2))

    def test_outside_registered_storage_is_not_silently_classified(self):
        with self.assertRaises(RuntimeError):
            classify([(99, 5, 0)], [(100, 200, 'KV_bytes')], 5)

    def test_ticket_size_must_match_the_actual_regions(self):
        with self.assertRaises(RuntimeError):
            classify([(100, 5, 0)], [(100, 200, 'KV_bytes')], 4)


if __name__ == '__main__':
    unittest.main()
