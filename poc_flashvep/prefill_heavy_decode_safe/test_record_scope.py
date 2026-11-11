"""Causal scope tests use synthetic clocks, not experiment data."""
import unittest
from .record_scope import classify


class RecordScopeTests(unittest.TestCase):
    def row(self):return dict(kind='graph',start_ns='0',end_ns='0',correlation='7')
    def test_unknown_in_window_is_not_discarded(self):
        self.assertFalse(classify([self.row()],[dict(correlation='7',start_ns='15',end_ns='16')],10,20,guard_ns=0)['passed'])
    def test_late_cpu_launch_cannot_execute_in_an_earlier_window(self):
        self.assertTrue(classify([self.row()],[dict(correlation='7',start_ns='21',end_ns='22')],10,20,guard_ns=0)['passed'])
    def test_early_cpu_launch_alone_does_not_prove_gpu_completion(self):
        self.assertFalse(classify([self.row()],[dict(correlation='7',start_ns='1',end_ns='2')],10,20,guard_ns=0)['passed'])
    def test_later_device_fence_proves_completion(self):
        fence=dict(begin=dict(cupti_ns=3),end=dict(cupti_ns=4))
        self.assertTrue(classify([self.row()],[dict(correlation='7',start_ns='1',end_ns='2')],10,20,fence,0)['passed'])
    def test_launch_racing_with_a_fence_is_unresolved(self):
        fence=dict(begin=dict(cupti_ns=3),end=dict(cupti_ns=5))
        self.assertFalse(classify([self.row()],[dict(correlation='7',start_ns='2',end_ns='4')],10,20,fence,0)['passed'])


if __name__=='__main__':unittest.main()
