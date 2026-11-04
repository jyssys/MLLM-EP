"""Prevent a bounded-window success from unlocking method experiments."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from .conditional_split_qualification import recovery_ready


class RecoveryReadiness(unittest.TestCase):
    def test_host_audit_requires_replacement(self):
        state = dict(status='RECOVERY_READY', same_run_gates=True,
                     whole_run_decode_SLO=True, independent_repeats=True,
                     exact_state_and_output=True, results=[dict(profile='audited')])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'state').mkdir()
            (root / 'state/host_audit_overlap_test.json').write_text(json.dumps(
                dict(run='audited', exclude_from_clean_confirmation=True)))
            with patch('poc_flashvep.fluidtoken_loop_split.conditional_split_qualification.ROOT', root):
                self.assertFalse(recovery_ready(state))
                self.assertTrue(recovery_ready(dict(state, results=[dict(profile='replacement')])))

    def test_bounded_window_is_insufficient(self):
        self.assertFalse(recovery_ready(dict(status='THREE_START_RECOVERY_PASSED')))

    def test_requires_every_scope(self):
        state = dict(status='RECOVERY_READY', same_run_gates=True,
                     whole_run_decode_SLO=True, independent_repeats=True,
                     exact_state_and_output=True)
        self.assertTrue(recovery_ready(state))
        for key in ('same_run_gates', 'whole_run_decode_SLO', 'independent_repeats', 'exact_state_and_output'):
            with self.subTest(key=key):
                self.assertFalse(recovery_ready(dict(state, **{key: False})))


if __name__ == '__main__':
    unittest.main()
