"""Reject controller syntax errors before a resource lease is acquired."""
import ast
from pathlib import Path
import unittest
from .partial_axis import token_prefix


class SourceContracts(unittest.TestCase):
    def test_every_campaign_module_parses(self):
        for path in Path(__file__).parent.glob('*.py'):
            with self.subTest(path=path.name):ast.parse(path.read_text(),filename=str(path))
    def test_token_plan_records_the_actual_aligned_prefix(self):
        for length in (1910,4608,8064,38912):
            for cut in (1,2,4,6,8,12):
                prefix=token_prefix(length,cut)
                self.assertEqual(prefix%16,0)
                self.assertTrue(0<prefix<length)
                if length%576==0:self.assertEqual((length-prefix)*36,length*cut)


if __name__=='__main__':unittest.main()
