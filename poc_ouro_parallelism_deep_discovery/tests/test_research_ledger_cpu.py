import unittest

from poc_ouro_parallelism_deep_discovery.scripts.build_research_ledger import canonical, indexed_sources, jsonl_kind


class LedgerTests(unittest.TestCase):
    def test_query_only_ledgers_do_not_inflate_source_count(self):
        self.assertEqual(jsonl_kind('c14_cyclic_prefill_searches_v1.jsonl', {'query': 'test'}), 'search')
        self.assertEqual(jsonl_kind('state_transform_frontier_searches_v1.jsonl', {'query': 'test'}), 'search')

    def test_source_ledger_retains_default_type(self):
        self.assertEqual(jsonl_kind('literature_sources.jsonl', {'url': 'https://example.com'}), 'source')

    def test_explicit_type_precedes_filename(self):
        self.assertEqual(jsonl_kind('mixed_searches.jsonl', {'record_type': 'source'}), 'source')
        self.assertEqual(jsonl_kind('mixed_sources.jsonl', {'type': 'search_batch'}), 'search_batch')

    def test_primary_observation_keeps_paper_and_code_evidence(self):
        row = {'paper': 'https://arxiv.org/html/2407.01425v1',
               'code': 'https://github.com/example/repository/blob/pin/file.py'}
        self.assertEqual(list(indexed_sources([row], '/primary_observations')),
                         [('/primary_observations/0', row)])
        self.assertEqual(canonical(row), 'arxiv:2407.01425')

    def test_code_only_primary_observation_is_not_dropped(self):
        row = {'code': 'https://github.com/example/repository/blob/pin/file.py'}
        self.assertEqual(list(indexed_sources(row, '/primary_observations/0')),
                         [('/primary_observations/0', row)])
        self.assertEqual(canonical(row), 'url:' + row['code'])

    def test_invalid_arxiv_month_is_not_a_work_key(self):
        self.assertEqual(canonical({'url': 'https://doi.org/10.1145/3689031.3696072'}),
                         'url:https://doi.org/10.1145/3689031.3696072')


if __name__ == '__main__':
    unittest.main()
