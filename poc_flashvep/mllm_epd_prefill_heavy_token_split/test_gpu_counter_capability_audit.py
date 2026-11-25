"""Observed restrictions are facts, not a fabricated hardware measurement."""
import unittest

from .gpu_counter_capability_audit import privilege_facts


class CounterCapabilityTest(unittest.TestCase):
    def test_restricted_nonprivileged_state_is_recorded(self):
        row=privilege_facts('RmProfilingAdminOnly: 1\n','CapEff:\t0000000000000000\n')
        self.assertEqual(row['RmProfilingAdminOnly'],1)
        self.assertFalse(row['effective_CAP_SYS_ADMIN'])

    def test_absent_fields_are_unknown_not_unrestricted(self):
        row=privilege_facts('other: 1\n','other: 2\n')
        self.assertIsNone(row['RmProfilingAdminOnly']);self.assertIsNone(row['effective_CAP_SYS_ADMIN'])


if __name__=='__main__':unittest.main()
