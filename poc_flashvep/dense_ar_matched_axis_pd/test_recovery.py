"""Check that a declared interrupted slot uses only its replacement name."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from . import campaign,recovery


class RecoveryTests(unittest.TestCase):
    def test_dispatch_uses_replacement_without_changing_configuration(self):
        cfg=dict(name='fixture_interrupted_layer',matrix_stage='discovery',evaluation_arm='Layer',
            static_layers=12,cap=128,arrival_rate=4.,count=144)
        replacement=dict(cfg,name='fixture_interrupted_layer_resume1')
        value=dict(output_gate=dict(passed=True))
        with patch.object(campaign,'resolve_configuration',return_value=replacement,create=True) as resolve, \
             patch.object(campaign,'write'),patch.object(campaign,'save',side_effect=lambda c:c['name']), \
             patch.object(campaign,'execute',return_value=True) as execute, \
             patch.object(campaign,'metrics',return_value=value):
            self.assertEqual(campaign.run_job(cfg),value)
        resolve.assert_called_once_with(cfg)
        execute.assert_called_once_with(replacement['name'])
        self.assertEqual({k:v for k,v in replacement.items() if k!='name'},
            {k:v for k,v in cfg.items() if k!='name'})

    def test_only_the_recorded_interruption_digest_is_acknowledged(self):
        with tempfile.TemporaryDirectory(prefix='matched_axis_recovery_test_') as directory:
            root=Path(directory);path=root/'raw/interrupted/dispatcher_result.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(complete=False,returncode=-15)))
            value=dict(validated_before_resume=True,replacements={'interrupted':'replacement'},
                interrupted_runs={'interrupted':dict(replacement_name='replacement',dispatcher_SHA256=recovery.digest(path))})
            with patch.object(recovery,'ROOT',root),patch.object(recovery,'record',return_value=value):
                self.assertTrue(recovery.acknowledged_failure('interrupted'))
                self.assertFalse(recovery.acknowledged_failure('replacement'))
                path.write_text(json.dumps(dict(complete=False,returncode=1)))
                self.assertFalse(recovery.acknowledged_failure('interrupted'))


if __name__=='__main__':
    unittest.main()
