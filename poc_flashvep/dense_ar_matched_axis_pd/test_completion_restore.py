"""A completed monitor must not leave all approved devices idle."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from . import batch_matched_monitor as monitor


class CompletionRestoreTest(unittest.TestCase):
    def test_completed_monitor_restores_owned_burners(self):
        owner = Mock()
        with tempfile.TemporaryDirectory(prefix='matched_completion_test_') as folder:
            with patch.object(monitor, 'snapshot', return_value={
                    'critical': [], 'all_scopes_complete': True, 'controllers': []}), \
                    patch.object(monitor, 'write'), \
                    patch.object(monitor, 'preserve', create=True), \
                    patch.object(monitor, 'resources', return_value=owner, create=True), \
                    patch.object(monitor, 'QUIET_LOCK', Path(folder)/'quiet.lock', create=True):
                monitor.main()
        owner.restore.assert_called_once_with(list(range(8)))


if __name__ == '__main__':
    unittest.main()
