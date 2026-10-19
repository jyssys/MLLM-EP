import unittest
import xml.etree.ElementTree as ET

from poc_ouro_parallelism_deep_discovery.scripts.plot_context_controls import render, validate_cells


def fixture():
    return [dict(size=size, TP=tp, context=context, median_wall_ms=context/100+1000/tp,
                 timing_trials=6, frozen_prompt_patterns=2)
            for size in ['1.4B', '2.6B'] for tp in [1, 2, 4]
            for context in [1024, 8192, 16384, 32768]]


class Tests(unittest.TestCase):
    def test_select_one_model(self):
        self.assertEqual(len(validate_cells(fixture(), '1.4B')), 12)

    def test_render_has_exact_cell_tooltips(self):
        root = ET.fromstring(render(fixture(), '2.6B'))
        namespace = {'s': 'http://www.w3.org/2000/svg'}
        self.assertEqual(len(root.findall('.//s:circle', namespace)), 12)
        self.assertEqual(len(root.findall('.//s:polyline', namespace)), 3)
        text = ''.join(root.itertext())
        self.assertIn('No population CI', text)
        self.assertIn('not natural reasoning latency', text)
        self.assertIn('TP4, N32768: 577.680000 ms', text)

    def test_reject_missing_or_duplicate(self):
        cells = fixture()
        for rows in [cells[1:], cells+[cells[0]]]:
            with self.assertRaises(ValueError):
                validate_cells(rows, '1.4B')

    def test_reject_changed_denominator(self):
        cells = fixture()
        cells[0]['frozen_prompt_patterns'] = 400
        with self.assertRaises(ValueError):
            render(cells, '1.4B')


if __name__ == '__main__':
    unittest.main()
