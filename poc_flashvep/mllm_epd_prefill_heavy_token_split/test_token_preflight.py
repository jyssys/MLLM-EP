"""Source receipts must not become GPU leases in the token preflight."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import token_correctness_control as control


class TokenPreflightTest(unittest.TestCase):
    def records(self, root):
        state = root / 'state'
        state.mkdir()
        (state / 'main100_baseline_confirmation_C14_complete.json').write_text(
            json.dumps(dict(status='COMPLETE')))
        names = []
        roles = dict(agg=['A4', 'A5', 'A6'], e_pd=['E4', 'P5', 'P6'],
                     native=['E4', 'P5', 'D6'])
        for family, members in roles.items():
            for start in [1, 2, 3]:
                for member in members:
                    name = f'server_base_C14_{family}_S{start}_{member}'
                    path = state / (name + '.json')
                    path.write_text(json.dumps(dict(finished=True, guardian=2**30)))
                    (state / (name + '_boot.source_snapshot.json')).write_text(
                        json.dumps(dict(archive_SHA256='diagnostic-receipt')))
                    names.append(path)
        return names

    def test_completed_leases_ignore_source_receipts(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.records(root)
            with patch.object(control, 'ROOT', root):
                control.wait_baselines()

    def test_live_guardian_blocks_the_next_gpu_job(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.records(root)
            paths[0].write_text(json.dumps(dict(finished=True, guardian=os.getpid())))
            with patch.object(control, 'ROOT', root), self.assertRaises(AssertionError):
                control.wait_baselines()

    def test_missing_lease_is_not_a_success(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.records(root)
            paths[0].unlink()
            with patch.object(control, 'ROOT', root), self.assertRaises(AssertionError):
                control.wait_baselines()


if __name__ == '__main__':
    unittest.main()
