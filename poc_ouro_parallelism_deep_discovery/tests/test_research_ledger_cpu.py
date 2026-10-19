import unittest

from poc_ouro_parallelism_deep_discovery.scripts.build_research_ledger import jsonl_kind


class LedgerTests(unittest.TestCase):
    def test_query_only_ledgers_do_not_inflate_source_count(self):
        self.assertEqual(jsonl_kind('c14_cyclic_prefill_searches_v1.jsonl', {'query': 'test'}), 'search')
        self.assertEqual(jsonl_kind('state_transform_frontier_searches_v1.jsonl', {'query': 'test'}), 'search')

    def test_source_ledger_retains_default_type(self):
        self.assertEqual(jsonl_kind('literature_sources.jsonl', {'url': 'https://example.com'}), 'source')

    def test_explicit_type_precedes_filename(self):
        self.assertEqual(jsonl_kind('mixed_searches.jsonl', {'record_type': 'source'}), 'source')
        self.assertEqual(jsonl_kind('mixed_sources.jsonl', {'type': 'search_batch'}), 'search_batch')


if __name__ == '__main__':
    unittest.main()
