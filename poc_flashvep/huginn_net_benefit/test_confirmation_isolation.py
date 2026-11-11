import unittest
from .confirmation_isolation import resolve


class IsolationTests(unittest.TestCase):
    def test_no_implicit_replacement(self):
        self.assertEqual(resolve('a', {}), 'a')

    def test_only_declared_cell_changes(self):
        mapping = {'a':'repeat'}
        self.assertEqual(resolve('a', mapping), 'repeat')
        self.assertEqual(resolve('b', mapping), 'b')
