"""Retain synthetic characterization of the original analyzer boundary. CPU only.

These tests do not test a later integrity supplement or inspect actual outcomes.
"""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from poc_ouro_parallelism_deep_discovery.contract import PROJECT, ROOT, read, sha
from poc_ouro_parallelism_deep_discovery.scripts import analyze_baseline_extension as analyzer
from poc_ouro_parallelism_deep_discovery.tests.test_baseline_extension_analysis_cpu import fixture, evaluator


class OriginalAnalyzerBoundaryTests(unittest.TestCase):
    def test_same_length_prompt_mutation_is_accepted_by_original_boundary(self):
        row, value = fixture()
        row['prompt'] = 'FROZEN_SYNTHETIC_PROMPT'
        value['prompt_ids'] = [2] * len(value['prompt_ids'])
        analyzer.request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH', evaluator)

    def test_unbound_text_changes_synthetic_grade_with_identical_output_ids(self):
        row, value = fixture()
        tokens = list(value['output_ids'])
        value['text'] = '1'
        left = analyzer.request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH')
        value['text'] = '2'
        right = analyzer.request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH')
        self.assertEqual(value['output_ids'], tokens)
        self.assertTrue(left['correct_completed'])
        self.assertFalse(right['correct_completed'])

    def test_unconfigured_stop_is_accepted_by_original_boundary(self):
        row, value = fixture()
        row['stop_strings'] = []
        value['output_ids'][-1] = 11
        value['finish_reason'] = 'stop'
        result = analyzer.request_metrics(row, value, '1.4B', 'adaptive', 5, 'HASH', evaluator)
        self.assertTrue(result['correct_completed'])
        self.assertFalse(result['truncated'])

    def test_synthetic_arm_admission_and_ignored_launch_anchor(self):
        archive = ROOT / 'artifacts/analyze_baseline_extension_pre_integrity_review_v1.py'
        self.assertEqual(sha(archive), 'f3c5d5bb0ddd10dd16a09cca0f4a43b132ab4d361f65a9633e963f6bb98cc731')
        specification = importlib.util.spec_from_file_location('_baseline_pre_integrity_review', archive)
        analyzer = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(analyzer)
        real_metrics = analyzer.request_metrics
        def synthetic_metrics(*args, **kwargs):
            return real_metrics(*args, **kwargs, evaluate=evaluator)
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory(prefix='ouro_analyzer_review_') as directory:
            root = Path(directory) / 'campaign'
            rows = [dict(id=f'synthetic/{index}', dataset='gsm8k' if index < 32 else 'math500',
                         answer='1', prompt='Synthetic only', stop_strings=[]) for index in range(64)]
            protocol = dict(N=64, rows=rows)
            protocol_path = root / 'artifacts/baseline_extension_protocol_v1.json'
            put(protocol_path, protocol)
            put(root / 'artifacts/probe_protocol.json', dict(rows=[]))
            contract = copy.deepcopy(read(ROOT / 'artifacts/baseline_extension_analysis_contract_v1.json'))
            # Use actual checked source files without any actual output directory.
            contract['source_sha256'] = {str(PROJECT / key): value
                                        for key, value in contract['source_sha256'].items()}
            put(root / 'artifacts/baseline_extension_analysis_contract_v1.json', contract)
            put(root / 'artifacts/baseline_extension_launch_contract_v1.json', dict(
                source_sha256={'artifacts/baseline_extension_protocol_v1.json': 'DELIBERATE_MISMATCH'}))
            protocol_hash = sha(protocol_path)
            job = root / 'outputs/baseline_extension_base14_fixed_gpu4_v1'
            sub = job / 'baseline_extension'
            status = dict(success=True, N_completed=64, N_expected=64, source_protocol_sha256=protocol_hash)
            put(sub / 'status.json', status)
            put(job / 'guard_status.json', dict(success=True, exit_code=0))
            put(sub / 'frozen_protocol.json', protocol)
            for frozen in rows:
                _, value = fixture()
                value.update(id=frozen['id'], dataset=frozen['dataset'], mode='fixed', physical_gpu=4,
                             source_protocol_sha256=protocol_hash)
                for depth in value['depth_events']:
                    depth['depth'] = 4
                value['stats'].update(recurrent_steps=12, recurrent_batches=12,
                                     recurrent_graph_hits=11, exit_depth_histogram={'3': 3})
                with gzip.open(sub / (frozen['id'].replace('/', '_') + '.json.gz'), 'wt') as stream:
                    json.dump(value, stream)
            with patch.object(analyzer, 'request_metrics', synthetic_metrics):
                result = analyzer.analyze(root)
                self.assertEqual(len(result['records']), 64)
                self.assertEqual(sum(job['status'] == 'AVAILABLE' for job in result['jobs']), 1)
                self.assertEqual(sum(job['status'] == 'NOT_AVAILABLE' for job in result['jobs']), 3)
                put(job / 'guard_status.json', dict(success=False, exit_code=1))
                self.assertEqual(analyzer.analyze(root)['records'], [])
                put(job / 'guard_status.json', dict(success=True, exit_code=0))
                put(sub / 'status.json', dict(status, N_completed=63))
                with self.assertRaises(ValueError):
                    analyzer.analyze(root)


if __name__ == '__main__':
    unittest.main()
