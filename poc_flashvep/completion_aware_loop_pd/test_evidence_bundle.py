"""Verify lossless chunked publication, including empty and duplicate files."""
import tempfile
import unittest
from pathlib import Path
from . import evidence_bundle as b


class BundleTests(unittest.TestCase):
    def test_round_trip_and_deduplication(self):
        with tempfile.TemporaryDirectory(prefix='loop_pd_bundle_test_') as temporary:
            p=Path(temporary);src=p/'src';src.mkdir()
            (src/'empty').touch();data=b'abcdefgh'*30
            (src/'one').write_bytes(data);(src/'two').write_bytes(data)
            old=b.CHUNK;b.CHUNK=64
            try:
                m=b.pack(src,p/'bundle');b.verify(p/'bundle',p/'restored')
                self.assertEqual((p/'restored/one').read_bytes(),data)
                self.assertEqual((p/'restored/two').read_bytes(),data)
                self.assertEqual((p/'restored/empty').read_bytes(),b'')
                self.assertEqual(m['unique_chunks'],2)
            finally:b.CHUNK=old


if __name__=='__main__':unittest.main()
