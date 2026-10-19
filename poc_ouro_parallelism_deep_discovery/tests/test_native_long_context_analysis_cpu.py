import unittest
from pathlib import Path
import tempfile

from poc_ouro_parallelism_deep_discovery.scripts.analyze_native_long_context import request_metrics, plot_series
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg


class MetricsTests(unittest.TestCase):
    def test_decode_denominator_uses127_intervals_not128_tokens(self):
        row = dict(wall_ms=2000, generated_tokens=128, metrics=dict(num_generation_tokens=128,
            is_corrupted=False, first_token_ts=200, last_token_ts=201.27,
            first_token_latency=.1, arrival_time=99999999999))
        result = request_metrics(row)
        self.assertAlmostEqual(result['native_decode_TPOT_ms'], 10)
        self.assertEqual(result['native_first_token_latency_ms'], 100)
        self.assertEqual(result['output_TPS'], 64)

    def test_corrupted_metrics_do_not_enter_analysis(self):
        with self.assertRaises(AssertionError):
            request_metrics(dict(wall_ms=2000, generated_tokens=128,
                metrics=dict(num_generation_tokens=128, is_corrupted=True)))

    def test_plot_series_matches_existing_svg_schema(self):
        data = [dict(size='1.4B', TP=1, context=1024, latency=10)]
        series = plot_series(data, '1.4B', 'latency')
        self.assertEqual(series, [dict(name='TP1', points=[(1024, 10)])])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'figure.svg'
            svg(path, series, 'CPU fixture', 'Tokens', 'ms')
            self.assertIn('<svg', path.read_text())


if __name__ == '__main__':
    unittest.main()
