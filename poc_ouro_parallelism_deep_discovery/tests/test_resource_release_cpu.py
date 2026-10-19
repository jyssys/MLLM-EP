import json
from pathlib import Path
import tempfile
import unittest

from poc_ouro_parallelism_deep_discovery.resource_release import release_complete


class ReleaseTests(unittest.TestCase):
    def test_status_is_not_final_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'guard_status.json').write_text(json.dumps(dict(success=True)))
            self.assertFalse(release_complete(root))

    def test_final_cleanup_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'guard_status.json').write_text(json.dumps(dict(success=True)))
            cleanup = root / 'guard_cleanup_validation.json'
            cleanup.write_text(json.dumps(dict(owned_GPU_contexts_remaining=[123])))
            self.assertFalse(release_complete(root))
            cleanup.write_text(json.dumps(dict(owned_GPU_contexts_remaining=[])))
            self.assertTrue(release_complete(root))

    def test_failed_run_is_not_successful_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'guard_status.json').write_text(json.dumps(dict(success=False)))
            (root / 'guard_cleanup_validation.json').write_text(json.dumps(dict(owned_GPU_contexts_remaining=[])))
            self.assertFalse(release_complete(root))


if __name__ == '__main__':
    unittest.main()
