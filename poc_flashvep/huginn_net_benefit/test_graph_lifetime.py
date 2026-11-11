import unittest
from types import SimpleNamespace
from .graph_lifetime import reset_entries
from .exit_integrity import validate_exit


class GraphTests(unittest.TestCase):
    def test_reset_once_before_owner_release(self):
        calls = []
        graph = SimpleNamespace(reset=lambda: calls.append('reset'))
        cache = SimpleNamespace(entries={0: SimpleNamespace(graph=graph),
            1: SimpleNamespace(graph=graph), 2: SimpleNamespace(graph=None)})
        self.assertEqual(reset_entries(cache, set()), 1)
        self.assertEqual(calls, ['reset'])
        self.assertTrue(all(entry.graph is None for entry in cache.entries.values()))

    def test_empty_cache(self):
        self.assertEqual(reset_entries(None, set()), 0)

    def test_stopped_is_not_successful_exit(self):
        validate_exit(0)
        for code in (None, -11, 1):
            with self.assertRaises(RuntimeError):
                validate_exit(code)
