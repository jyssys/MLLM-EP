"""Boot-source extraction must preserve bytes and reject escaping paths."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile


class FrozenSourceTest(unittest.TestCase):
    def archive(self, root, member='poc_flashvep/frozen_test.py'):
        from .common import digest
        target = root / 'source.zip'
        content = b'VALUE = 17\n'
        rows = [dict(member='poc_flashvep/__init__.py', SHA256=hashlib.sha256(b'').hexdigest()),
                dict(member=member, SHA256=hashlib.sha256(content).hexdigest())]
        with ZipFile(target, 'x') as value:
            value.writestr('poc_flashvep/__init__.py', b'')
            value.writestr(member, content)
            value.writestr('source_manifest.json', json.dumps(rows))
        return dict(archive=str(target), archive_SHA256=digest(target))

    def test_new_python_process_imports_the_verified_snapshot(self):
        from . import source_snapshot as module
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.archive(root)
            with patch.object(module, 'ROOT', root):
                folder = module.materialize(record)
            result = subprocess.run([sys.executable, '-c',
                'import poc_flashvep.frozen_test as m; print(m.__file__); assert m.VALUE == 17'],
                cwd=folder, capture_output=True, text=True, check=True)
            self.assertEqual(Path(result.stdout.strip()), folder / 'poc_flashvep/frozen_test.py')
            self.assertEqual((folder / 'poc_flashvep/frozen_test.py').stat().st_mode & 0o777, 0o444)

    def test_traversal_is_rejected_before_the_member_is_written(self):
        from . import source_snapshot as module
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.archive(root, '../escaped.py')
            with patch.object(module, 'ROOT', root), self.assertRaises(AssertionError):
                module.materialize(record)
            self.assertFalse((root / 'escaped.py').exists())

    def test_corrupt_archive_is_rejected(self):
        from . import source_snapshot as module
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.archive(root)
            record['archive_SHA256'] = '0' * 64
            with patch.object(module, 'ROOT', root), self.assertRaises(AssertionError):
                module.materialize(record)

    def test_concurrent_identical_bytes_have_distinct_boot_directories(self):
        import shutil
        from . import source_snapshot as module
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.archive(root)
            other = root / 'another_boot.zip'
            shutil.copyfile(first['archive'], other)
            second = dict(first, archive=str(other))
            with patch.object(module, 'ROOT', root):
                a = module.materialize(first)
                b = module.materialize(second)
            self.assertNotEqual(a, b)


if __name__ == '__main__':
    unittest.main()
