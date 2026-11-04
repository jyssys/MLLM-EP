import unittest
from .source_slot_audit import classify


class SlotSampleTests(unittest.TestCase):
    def test_transfer_phase_does_not_hide_an_unretired_submission(self):
        row = dict(requests=[dict(phase='RequestPhase.TRANSFERRING')],
                   all_submitted_unretired_ids=['request'])
        self.assertEqual(classify(row), (False, True, False))
        row['all_submitted_unretired_ids'] = []
        self.assertEqual(classify(row), (False, False, True))

    def test_compute_and_transfer_requests_are_not_transfer_only(self):
        row = dict(requests=[dict(phase='RequestPhase.TRANSFERRING'),
                             dict(phase='RequestPhase.PREFILLING_RECURRENT')],
                   all_submitted_unretired_ids=[])
        self.assertEqual(classify(row), (True, False, False))


if __name__ == '__main__':
    unittest.main()
