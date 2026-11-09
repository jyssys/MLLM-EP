"""Check lossless deduplication and checksum rejection on small fixtures."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import evidence_bundle as bundle


class EvidenceTests(unittest.TestCase):
    def test_roundtrip_and_checksum_failure(self):
        with tempfile.TemporaryDirectory(prefix='idle-loop-bundle-test-') as temp:
            root=Path(temp)/'campaign';(root/'raw/run').mkdir(parents=True)
            (root/'raw/run/a.bin').write_bytes(b'full exact evidence\0'*100)
            (root/'raw/run/b.bin').write_bytes((root/'raw/run/a.bin').read_bytes())
            (root/'config.json').write_text('{"depth":32}\n')
            with patch.object(bundle,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):bundle.build()
            target=root/'evidence_bundle';manifest=bundle.verify(target)
            self.assertEqual(len(manifest['files']),3);self.assertEqual(manifest['unique_blobs'],2)
            recovered=Path(temp)/'recovered';bundle.extract(target,recovered)
            self.assertEqual((recovered/'raw/run/a.bin').read_bytes(),(root/'raw/run/a.bin').read_bytes())
            shard=target/manifest['shards'][0]['path']
            with shard.open('ab') as output:output.write(b'corruption')
            with self.assertRaises(AssertionError):bundle.verify(target)


if __name__=='__main__':unittest.main()
