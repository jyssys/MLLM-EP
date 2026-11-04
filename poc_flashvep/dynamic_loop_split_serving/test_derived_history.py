import json
import tempfile
import unittest
from pathlib import Path
from .common import write_derived


class HistoryTests(unittest.TestCase):
    def test_changed_derived_result_retains_original_bytes(self):
        with tempfile.TemporaryDirectory(prefix='derived_history_test_') as temporary:
            path = Path(temporary) / 'serving_metrics.json'
            write_derived(path, dict(value=1))
            original = path.read_bytes()
            write_derived(path, dict(value=2))
            histories = list(path.parent.glob('serving_metrics_history_*.json'))
            self.assertEqual(len(histories), 1)
            self.assertEqual(histories[0].read_bytes(), original)
            self.assertEqual(json.loads(path.read_text()), dict(value=2))
            write_derived(path, dict(value=2))
            self.assertEqual(len(list(path.parent.glob('serving_metrics_history_*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
