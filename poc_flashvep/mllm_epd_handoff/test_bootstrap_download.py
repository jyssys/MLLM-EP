import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .assets import RELATIVE_ROOT, ORIGINAL_REPO, MANIFESTS
from .bootstrap_download import fetch_one, install_media, parquet_name, safe_name, selected


class DownloadTests(unittest.TestCase):
    def test_original_locator_does_not_require_old_server(self):
        row = dict(source_revision='frozen', parquet='/different/home/hub/snapshots/frozen/data/val.parquet')
        self.assertEqual(parquet_name(row), 'data/val.parquet')

    def test_invalid_locator_rejected(self):
        with self.assertRaises(ValueError):
            parquet_name(dict(source_revision='frozen', parquet='/different/revision/data/val.parquet'))
        for name in ('../outside', '/etc/passwd', 'data/../../outside'):
            with self.assertRaises(ValueError):
                safe_name(name)

    def test_download_uses_revision_and_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'upstream'
            path.write_bytes(b'original')
            item = dict(repo_id='official/source', repo_type='dataset', revision='fixed', filename='data/a.parquet',
                        bytes=8, SHA256=hashlib.sha256(b'original').hexdigest())
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                return path
            self.assertEqual(fetch_one(item, Path(folder), offline=True, downloader=download), path)
            self.assertEqual(calls[0]['revision'], 'fixed')
            self.assertTrue(calls[0]['local_files_only'])
            path.write_bytes(b'changed!')
            with self.assertRaises(ValueError):
                fetch_one(item, Path(folder), downloader=download)

    def test_scope_keeps_frozen_primary_and_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            data = repo / RELATIVE_ROOT / 'data'
            data.mkdir(parents=True)
            rows = [dict(request_id=str(i), prompt_SHA256=str(i)) for i in range(3)]
            for name, selection in zip(MANIFESTS, (rows[:2], rows[:1], rows[2:])):
                (data / name).write_text(json.dumps(dict(requests=selection)))
            self.assertEqual(set(selected(repo, 'main100')[1]), {'0', '2'})
            self.assertEqual(set(selected(repo, 'main400')[1]), {'0', '1', '2'})

    def test_image_copy_is_local_and_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            image = base / 'generated.png'
            image.write_bytes(b'original image bytes')
            expected = dict(path=str(ORIGINAL_REPO / RELATIVE_ROOT / 'media/ab/image.png'),
                            SHA256=hashlib.sha256(image.read_bytes()).hexdigest(),
                            pixel_SHA256='pixel hash', original_resolution=[1, 2])
            actual = dict(expected, path=str(image))
            row = dict(request_id='request', images=[expected], question='Original?', reference_answers=['yes'])
            material = dict(images=[actual], question='Original?', reference_answers=['yes'])
            repo = base / 'clone'
            install_media(repo, row, material)
            target = repo / RELATIVE_ROOT / 'media/ab/image.png'
            self.assertEqual(target.read_bytes(), image.read_bytes())
            target.write_bytes(b'existing user file')
            with self.assertRaises(ValueError):
                install_media(repo, row, material)
            self.assertEqual(target.read_bytes(), b'existing user file')

    def test_changed_image_or_question_is_not_a_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            row = dict(request_id='request', images=[dict()], question='Original?', reference_answers=['yes'])
            material = dict(images=[dict()], question='Changed?', reference_answers=['yes'])
            with self.assertRaises(ValueError):
                install_media(repo, row, material)
