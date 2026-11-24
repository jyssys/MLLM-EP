"""A fixed source stratum must not replace or leak the frozen heldout dataset."""
import json
import unittest


class DocumentStratumTest(unittest.TestCase):
    def test_take_all_parent_discovery_documents_and_no_heldout(self):
        from .document_stratum_control import source_rows
        from .common import ROOT,digest
        path,rows=source_rows();parent=json.loads(path.read_text())
        expected={r['request_id'] for r in parent['requests'] if r['request_id'] in parent['discovery_ids'] and
            'MMLongBench' in r['source']}
        self.assertEqual({r['request_id'] for r in rows},expected)
        self.assertEqual(len(rows),37)
        self.assertTrue(expected.isdisjoint(parent['heldout_ids']))
        self.assertEqual(digest(ROOT/'data/main100_manifest_template_v2.json'),
            '910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1')


if __name__=='__main__':unittest.main()
