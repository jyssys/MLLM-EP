"""Metadata-only freeze checks. No runtime, model, or GPU import."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from poc_ouro_parallelism_deep_discovery.scripts.freeze_baseline_extension import freeze


class BaselineExtensionTests(unittest.TestCase):
    def setup_pool(self, root):
        project, campaign = root / 'project', root / 'campaign'
        paths = {
            'pool': project / 'poc_decode_first_depth_aware_dp/artifacts/quality_protocol_v3_pilot32.json',
            'prior': project / 'poc_depth_cdb_factorial/artifacts/workload_frozen.json',
            'old': campaign / 'artifacts/probe_protocol.json'}
        rows = [dict(id=f'{domain}/{index}', dataset=domain, split='pilot' if index < 4 else 'heldout',
                     prompt=f'fixture{index}', answer=str(index)) for domain in ('gsm8k', 'math500') for index in range(100)]
        values = dict(pool=dict(rows=rows), prior=dict(quality=[dict(id='gsm8k/10')], speed=[dict(id='math500/10')]),
                      old=dict(rows=[dict(id=f'{domain}/{index}') for domain in ('gsm8k', 'math500') for index in range(4, 10)]))
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(values[key]))
        return project, campaign, paths

    def test_disjoint_balanced_deterministic_freeze(self):
        with tempfile.TemporaryDirectory(prefix='ouro_baseline_freeze_') as directory:
            project, campaign, paths = self.setup_pool(Path(directory))
            with patch('poc_ouro_parallelism_deep_discovery.scripts.freeze_baseline_extension.PROJECT', project), \
                 patch('poc_ouro_parallelism_deep_discovery.scripts.freeze_baseline_extension.ROOT', campaign):
                a, b = freeze(campaign / 'a.json'), freeze(campaign / 'b.json')
                self.assertEqual([row['id'] for row in a['rows']], [row['id'] for row in b['rows']])
                self.assertEqual(len(a['rows']), 64)
                for domain in ('gsm8k', 'math500'):
                    ids = {row['id'] for row in a['rows'] if row['dataset'] == domain}
                    self.assertEqual(len(ids), 32)
                    self.assertFalse(ids & {f'{domain}/{index}' for index in range(11)})
                self.assertFalse(a['protected_results_inspected'])
                self.assertFalse(a['selection_uses_outcomes'])
                self.assertEqual(a['quality_cap'], 8192)
                with self.assertRaises(FileExistsError):
                    freeze(campaign / 'a.json')

    def test_answers_do_not_change_selected_IDs(self):
        with tempfile.TemporaryDirectory(prefix='ouro_baseline_freeze_') as directory:
            project, campaign, paths = self.setup_pool(Path(directory))
            with patch('poc_ouro_parallelism_deep_discovery.scripts.freeze_baseline_extension.PROJECT', project), \
                 patch('poc_ouro_parallelism_deep_discovery.scripts.freeze_baseline_extension.ROOT', campaign):
                before = freeze(campaign / 'a.json')
                pool = json.loads(paths['pool'].read_text())
                for row in pool['rows']:
                    row['answer'] = 'CHANGED_WITHOUT_SELECTION_EFFECT'
                paths['pool'].write_text(json.dumps(pool))
                after = freeze(campaign / 'b.json')
                self.assertEqual([row['id'] for row in before['rows']], [row['id'] for row in after['rows']])

    def test_entrypoints_remain_torch_free_on_import(self):
        for name in ('freeze_baseline_extension', 'baseline_extension'):
            command = [sys.executable, '-c', f'import sys; import poc_ouro_parallelism_deep_discovery.scripts.{name}; assert "torch" not in sys.modules']
            subprocess.run(command, check=True, capture_output=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
