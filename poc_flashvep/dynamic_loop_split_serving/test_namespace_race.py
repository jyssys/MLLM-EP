import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import run


class NamespaceRaceTests(unittest.TestCase):
    def scenario(self, receipt=None, message=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / 'state').mkdir()
        (root / 'raw').mkdir()
        name = 'shared_test'

        def raced_launch(actual):
            self.assertEqual(actual, name)
            folder = root / 'raw' / name
            folder.mkdir()
            if receipt is not None:
                (folder / 'dispatcher_result.json').write_text(json.dumps(receipt))
            raise RuntimeError(message or 'Unfinished namespace requires explicit evidence review: ' + name)

        with patch.object(run, 'ROOT', root), patch.object(run, 'initialize'), \
                patch.object(run, '_execute_locked', side_effect=raced_launch):
            return run.execute(name)

    def test_legacy_controller_finishes_between_check_and_dispatch(self):
        self.assertTrue(self.scenario(dict(complete=True)))

    def test_failed_legacy_receipt_does_not_become_success(self):
        self.assertFalse(self.scenario(dict(complete=False)))

    def test_missing_receipt_still_requires_evidence_review(self):
        with self.assertRaisesRegex(RuntimeError, 'evidence review'):
            self.scenario()

    def test_unrelated_runtime_error_remains_an_error(self):
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            self.scenario(dict(complete=True), 'model failed')


if __name__ == '__main__':
    unittest.main()
