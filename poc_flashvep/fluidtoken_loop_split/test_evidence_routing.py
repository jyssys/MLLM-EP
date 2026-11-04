import json
import tempfile
import types
import unittest
from pathlib import Path
from ..looped_pd_abc.telemetry import EvidenceWriter
from .evidence_routing import enqueue, route_aliases


class EvidenceRouting(unittest.TestCase):
    def test_captured_aliases_and_unrelated_function(self):
        def old(*args):
            pass
        def other(*args):
            pass
        def replacement(*args):
            pass
        modules = [('poc_flashvep.a', types.SimpleNamespace(append=old)),
                   ('poc_flashvep.b', types.SimpleNamespace(append=other)),
                   ('unrelated', types.SimpleNamespace(append=old))]
        self.assertEqual(route_aliases(modules, (old,), replacement), ['poc_flashvep.a.append'])
        self.assertIs(modules[0][1].append, replacement)
        self.assertIs(modules[1][1].append, other)
        self.assertIs(modules[2][1].append, old)

    def test_exact_records_snapshot_order_and_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            writer = EvidenceWriter()
            try:
                for index in range(200):
                    value = dict(index=index, nested=[index, None, True])
                    enqueue(writer, path, value)
                    value['nested'][0] = -1
                writer.flush()
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows, [dict(index=i, nested=[i, None, True]) for i in range(200)])
                self.assertEqual(writer.records, writer.background_records)
                with self.assertRaises(ValueError):
                    enqueue(writer, path, dict(value=float('nan')))
                self.assertEqual(writer.records, 200)
            finally:
                writer.close()


if __name__ == '__main__':
    unittest.main()
