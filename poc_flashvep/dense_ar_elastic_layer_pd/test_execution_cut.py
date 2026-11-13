import unittest
from .execution_cut import destination_suffix_cut


class ExecutionCutTest(unittest.TestCase):
    def test_native_join_with_resident_migrated_decode(self):
        self.assertIsNone(destination_suffix_cut([28,36],[False,True],[8010,8000],8000))
    def test_real_suffix(self):self.assertEqual(destination_suffix_cut([28,28],[True,True],[0,0],8000),28)
    def test_mixed_suffix_and_decode_rejected(self):
        with self.assertRaises(AssertionError):destination_suffix_cut([28,36],[True,False],[0,8010],8000)
    def test_mixed_suffix_frontiers_rejected(self):
        with self.assertRaises(AssertionError):destination_suffix_cut([28,32],[True,True],[0,0],8000)


if __name__=='__main__':unittest.main()
