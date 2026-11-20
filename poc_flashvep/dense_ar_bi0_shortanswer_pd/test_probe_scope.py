import unittest
from .numerical_probe import computed_layers


class ProbeScopeTests(unittest.TestCase):
    def test_partial_source_reads_only_computed_layers(self):
        self.assertEqual(computed_layers('prefill',{},dict(axis_kind='Layer',axis_cut=4)),32)
        self.assertEqual(computed_layers('decode',{},dict(axis_kind='Layer',axis_cut=4)),36)

    def test_native_fallback_has_all_layers(self):
        self.assertEqual(computed_layers('prefill',dict(axis_action='Native',axis_cut=0),dict(axis_kind='Layer',axis_cut=4)),36)
        self.assertEqual(computed_layers('prefill',dict(axis_action='Token',axis_cut=4),{}),36)
