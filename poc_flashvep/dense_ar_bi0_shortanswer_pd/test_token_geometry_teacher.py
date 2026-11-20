import unittest
from .token_geometry_teacher import clipped_budget


class TokenGeometryTeacherTests(unittest.TestCase):
    def test_prefix_boundary_is_not_crossed(self):
        self.assertEqual(clipped_budget(8192,8192,12000),3808)
        self.assertEqual(clipped_budget(8192,0,12000),8192)
        self.assertEqual(clipped_budget(8192,12000,12000),8192)
