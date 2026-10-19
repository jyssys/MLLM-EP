"""CPU-only parser checks. These fixtures are not model performance evidence."""
import unittest
from poc_ouro_parallelism_deep_discovery.scripts.parse_native_graph_dot import parse_graph, classify


def node(index):
    return ('"graph_1_node_%d"[style="bold" shape="record" label="{KERNEL\n'
            '| {ID | %d (topoId: %d) | test_kernel} \n'
            '| {{node handle | func handle} | {0x0010 | 0x0020}}\n}"];\n') % (index, index, 10-index)


class DotTests(unittest.TestCase):
    def test_serial(self):
        value = node(0) + node(1) + '"graph_1_node_0" -> "graph_1_node_1" [headlabel=0];'
        nodes, edges, order, _, _, widths = parse_graph(value)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        self.assertEqual(order, ['graph_1_node_0', 'graph_1_node_1'])
        self.assertEqual(max(widths), 1)

    def test_unknown_edge(self):
        with self.assertRaisesRegex(ValueError, 'unknown node'):
            parse_graph(node(0) + '"graph_1_node_0" -> "graph_1_node_1";')

    def test_cycle(self):
        with self.assertRaisesRegex(ValueError, 'cycle'):
            parse_graph(node(0) + '"graph_1_node_0" -> "graph_1_node_0";')

    def test_classification(self):
        self.assertEqual(classify('FlashAttnFwdCombine'), 'ATTENTION_SPLIT_MERGE')
        self.assertEqual(classify('FlashAttnFwdSm90'), 'ATTENTION_CORE')
        self.assertEqual(classify('unrecognized'), 'UNRESOLVED')

    def test_memcpy(self):
        text = ('"graph_1_node_0"[shape="record" label="{\nMEMCPY\n'
                '| {{ID | node handle} | {0 (topoId: 0) | 0x0010}}\n}"];')
        nodes, *_ = parse_graph(text)
        self.assertEqual(nodes['graph_1_node_0']['kind'], 'MEMCPY')


if __name__ == '__main__':
    unittest.main()
