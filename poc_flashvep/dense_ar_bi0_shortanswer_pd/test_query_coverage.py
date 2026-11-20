import unittest
from .axis_timeline import contiguous_queries


class QueryCoverageTests(unittest.TestCase):
    def test_partial_prefix_and_suffix(self):
        self.assertTrue(contiguous_queries([dict(start=0,count=8192),dict(start=8192,count=808)],0,9000))
        self.assertTrue(contiguous_queries([dict(start=9000,count=1000),dict(start=10000,count=1)],9000,10000))

    def test_equal_sum_with_overlap_is_rejected(self):
        self.assertFalse(contiguous_queries([dict(start=0,count=5),dict(start=4,count=5)],0,10))

    def test_native_recomputation(self):
        self.assertTrue(contiguous_queries([dict(start=7999,count=1),dict(start=8000,count=1)],7999,8000))


if __name__=='__main__':unittest.main()
