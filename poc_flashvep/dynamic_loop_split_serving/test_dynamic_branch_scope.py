import unittest
from .dynamic_recovery import validate_branch_scope


class BranchScopeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(phase='EXACT_DYNAMIC_BRANCH_DIAGNOSTIC', state_probe=True,
            verify_transfer=True, count=8, calibration_serial=True)

    def test_serial8_and_concurrent32_are_explicit_correctness_cases(self):
        validate_branch_scope(self.cfg)
        validate_branch_scope(dict(self.cfg, count=32, calibration_serial=False))
        with self.assertRaises(ValueError):
            validate_branch_scope(dict(self.cfg, calibration_serial=False))

    def test_no_serving_or_untraced_case_can_force_acceptance(self):
        with self.assertRaises(ValueError):
            validate_branch_scope(dict(self.cfg, phase='CLEAN_SERVING'))
        with self.assertRaises(ValueError):
            validate_branch_scope(dict(self.cfg, state_probe=False))


if __name__ == '__main__':
    unittest.main()
