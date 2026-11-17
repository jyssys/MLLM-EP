"""Reject group drift in post-hoc budget controls."""
import unittest
from .geometry_control import group_size


class GeometryTests(unittest.TestCase):
    def test_default_limit(self):
        self.assertEqual(group_size(7,dict(P_slots=4,P_group_size=4)),4)

    def test_exact_recorded_membership(self):
        cfg=dict(P_slots=4,P_group_size=4,planned_source_group_sizes=[1,3,4,2])
        self.assertEqual([group_size(i,cfg) for i in (0,1,4,8)],[1,3,4,2])

    def test_no_mid_group_reentry(self):
        cfg=dict(P_slots=4,P_group_size=4,planned_source_group_sizes=[1,3,4])
        for index in (2,3,5,8):
            with self.assertRaises(AssertionError):
                group_size(index,cfg)


if __name__=='__main__':
    unittest.main()
