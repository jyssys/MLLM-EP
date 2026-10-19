import copy
import unittest

from poc_ouro_parallelism_deep_discovery.cuda_graph_typed_event_nodes import (
    require_default_cuts, validate_typed_insertion)


def fixture():
    return dict(nodes=[dict(handle=hex(node), kind=0) for node in range(1, 5)],
        edges=[dict(source=hex(a), target=hex(b), raw_hex=raw, is_default=raw == '0000000000000000')
               for a, b, raw in [(1, 2, '0100010000000000'),
                                 (2, 3, '0000000000000000'), (3, 4, '0000000000000000')]],
        reserved_nonzero_count=0)


class TypedEventTests(unittest.TestCase):
    def test_default_cuts_pass(self):
        self.assertEqual(require_default_cuts(fixture(), [0, 2, 4]), [1, 2, 3, 4])

    def test_programmatic_cut_refused(self):
        with self.assertRaisesRegex(ValueError, 'strengthen'):
            require_default_cuts(fixture(), [0, 1, 4])

    def test_reserved_metadata_refused(self):
        before = fixture()
        before['reserved_nonzero_count'] = 1
        with self.assertRaisesRegex(ValueError, 'Reserved'):
            require_default_cuts(before, [0, 4])

    def test_unknown_semantics_refused(self):
        before = fixture()
        before['edges'][0]['raw_hex'] = '0200010000000000'
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            require_default_cuts(before, [0, 4])

    def insertion(self):
        before = fixture()
        after = copy.deepcopy(before)
        after['nodes'].append(dict(handle='0x5', kind=7))
        after['edges'].extend(dict(source=a, target=b, raw_hex='0000000000000000', is_default=True)
                              for a, b in [('0x2', '0x5'), ('0x5', '0x3')])
        return before, after

    def test_preserved_programmatic_edge_passes(self):
        before, after = self.insertion()
        validate_typed_insertion(before, after, [5])

    def test_changed_original_metadata_refused(self):
        before, after = self.insertion()
        after['edges'][0]['raw_hex'] = '0000000000000000'
        with self.assertRaisesRegex(ValueError, 'original dependency'):
            validate_typed_insertion(before, after, [5])

    def test_nondefault_added_edge_refused(self):
        before, after = self.insertion()
        after['edges'][-1]['raw_hex'] = '0100010000000000'
        with self.assertRaisesRegex(ValueError, 'unsupported dependency'):
            validate_typed_insertion(before, after, [5])

    def test_changed_node_refused(self):
        before, after = self.insertion()
        after['nodes'][0]['kind'] = 1
        with self.assertRaisesRegex(ValueError, 'original node'):
            validate_typed_insertion(before, after, [5])


if __name__ == '__main__':
    unittest.main()
