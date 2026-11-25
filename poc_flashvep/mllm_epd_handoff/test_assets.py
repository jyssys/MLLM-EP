import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .assets import MANIFESTS, ORIGINAL_REPO, RELATIVE_ROOT, inventory, prepare, relative_asset, verify


class AssetTests(unittest.TestCase):
    def fixture(self, root):
        paths = [RELATIVE_ROOT / 'media/x/image.png', RELATIVE_ROOT / 'data/request_tensors/test.npz']
        for path in paths:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(b'test asset')
        sha = hashlib.sha256(b'test asset').hexdigest()
        value = dict(requests=[dict(request_id='test', question='Original question', prompt_SHA256='unchanged',
                    images=[dict(path=str(ORIGINAL_REPO / paths[0]), SHA256=sha)],
                    token_metadata_file=str(ORIGINAL_REPO / paths[1]), token_metadata_SHA256=sha)])
        for name in MANIFESTS:
            (root / RELATIVE_ROOT / 'data' / name).write_text(json.dumps(value))
        return value, paths

    def test_deduplicate_and_verify(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            catalog = inventory(root)
            self.assertEqual(len(catalog['files']), 2)
            self.assertEqual(catalog['unique_requests'], 1)
            self.assertEqual(verify(root, catalog)['status'], 'ASSET_BYTES_VERIFIED')

    def test_relocation_changes_only_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, paths = self.fixture(root)
            catalog = inventory(root)
            prepare(root, catalog, root / 'fresh_session')
            actual = json.loads((root / 'fresh_session' / MANIFESTS[0]).read_text())
            expected = copy.deepcopy(original)
            expected['requests'][0]['images'][0]['path'] = str(root / paths[0])
            expected['requests'][0]['token_metadata_file'] = str(root / paths[1])
            self.assertEqual(actual, expected)
            self.assertEqual(json.loads((root / RELATIVE_ROOT / 'data' / MANIFESTS[0]).read_text()), original)

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, paths = self.fixture(root)
            catalog = inventory(root)
            (root / paths[0]).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify(root, catalog)

    def test_traversal_and_unrelated_assets_rejected(self):
        for path in ('/tmp/private', str(ORIGINAL_REPO / RELATIVE_ROOT / '../other'),
                     str(ORIGINAL_REPO / RELATIVE_ROOT / 'state/old_pid.json')):
            with self.assertRaises(ValueError):
                relative_asset(path)

    def test_frozen_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with self.assertRaises(ValueError):
                prepare(root, inventory(root), root / RELATIVE_ROOT / 'new')

    def test_existing_session_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            catalog = inventory(root)
            prepare(root, catalog, root / 'fresh_session')
            with self.assertRaises(FileExistsError):
                prepare(root, catalog, root / 'fresh_session')
