"""A failed independent study can release resources, but cannot supply correctness."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import interference_probe_control as controller


class PredecessorScopeTest(unittest.TestCase):
    def test_terminal_failed_predecessor_does_not_supply_quality_proof(self):
        with TemporaryDirectory() as directory,patch.object(controller,'ROOT',Path(directory)):
            root=Path(directory);(root/'state').mkdir();tag='whole_DEFLECTION_C43R_TEST'
            (root/'state'/f'{tag}_controller.json').write_text(json.dumps(dict(pid=999999999)))
            controller.wait_for_whole_resource(tag)
            record=json.loads((root/'state'/f'{controller.TAG}_resource_predecessor_closeout.json').read_text())
            self.assertFalse(record['predecessor_completed'])
            self.assertTrue(record['independent_correctness_preflight_required'])
            self.assertIn('No correctness',record['dependency'])

    def test_a_live_unrelated_process_is_not_treated_as_the_expected_controller(self):
        with TemporaryDirectory() as directory,patch.object(controller,'ROOT',Path(directory)):
            root=Path(directory);(root/'state').mkdir();tag='whole_DEFLECTION_C43R_TEST'
            (root/'state'/f'{tag}_controller.json').write_text(json.dumps(dict(pid=os.getpid())))
            with self.assertRaises(AssertionError):controller.wait_for_whole_resource(tag)


if __name__=='__main__':unittest.main()
