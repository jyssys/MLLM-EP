"""Keep complete startup hash validation outside the environment size limit."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from .common import source_hash_manifest


class SourceManifestTests(unittest.TestCase):
    def test_file_has_priority_without_losing_entries(self):
        values={f'/frozen/path/{i}':'a'*64 for i in range(5000)}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'hashes.json';path.write_text(json.dumps(values))
            with patch.dict(os.environ,{'DENSE_SOURCE_HASHES_FILE':str(path),'DENSE_SOURCE_HASHES':'{}'}):
                self.assertEqual(source_hash_manifest(),values)
    def test_legacy_startup_format_is_retained(self):
        with patch.dict(os.environ,{'DENSE_SOURCE_HASHES':'{"old":"hash"}'},clear=True):
            self.assertEqual(source_hash_manifest(),{'old':'hash'})


if __name__=='__main__':unittest.main()
