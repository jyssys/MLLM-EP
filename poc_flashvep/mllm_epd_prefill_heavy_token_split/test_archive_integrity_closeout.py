"""Lossless evidence and source checks must reject corruption and path escape."""
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from .archive_integrity_closeout import confined_path,verify_raw_archive,verify_source_archive,tensor_receipts,verify_executed_source
from .common import digest


class ArchiveIntegrityTest(unittest.TestCase):
    def test_append_only_history_requires_explicit_allowlist_and_preserves_exact_prefix(self):
        with TemporaryDirectory() as folder:
            repo=Path(folder);root=repo/'evidence';raw=root/'raw';raw.mkdir(parents=True)
            source=raw/'processor_handoff_decode_gpu6.jsonl';target=root/'raw.gz'
            source.write_bytes(b'{"sample":1}\n');target.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
            row=dict(source=str(source.relative_to(repo)),compressed_path='evidence/raw.gz',source_bytes=source.stat().st_size,
                compressed_bytes=target.stat().st_size,source_SHA256=digest(source),compressed_SHA256=digest(target))
            with source.open('ab') as stream:stream.write(b'{"sample":2}\n')
            with self.assertRaises(AssertionError):verify_raw_archive(row,repo,root)
            value=verify_raw_archive(row,repo,root,allow_append=True)
            self.assertEqual(value['status'],'HISTORICAL_APPEND_PREFIX_VERIFIED')
            self.assertEqual(value['appended_JSONL_records'],1)
            source.write_bytes(b'{"sample":9}\n{"sample":2}\n')
            with self.assertRaises(AssertionError):verify_raw_archive(row,repo,root,allow_append=True)

    def test_executed_materialization_is_independent_proof_not_a_metadata_rewrite(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);member='poc_flashvep/mllm_epd_prefill_heavy_token_split/run.py';data=b'X=1\n'
            sha=hashlib.sha256(data).hexdigest();archive_sha='a'*64
            cwd=root/'runtime_sources'/('a'*16+'_original.source');(cwd/Path(member).parent).mkdir(parents=True)
            (cwd/member).write_bytes(data)
            manifest=[dict(member=member,bytes=len(data),SHA256=sha)]
            (cwd/'verified_source_manifest.json').write_text(json.dumps(dict(archive_SHA256=archive_sha,files=manifest)))
            lease=dict(executed_source_cwd=str(cwd),source_archive_SHA256=archive_sha)
            result=verify_executed_source(lease,manifest,root)
            self.assertTrue(result['all_materialized_members_match'])
            (cwd/member).write_bytes(b'X=2\n')
            with self.assertRaises(AssertionError):verify_executed_source(lease,manifest,root)

    def test_real_gzip_roundtrip_and_changed_source(self):
        with TemporaryDirectory() as folder:
            repo=Path(folder);root=repo/'evidence';root.mkdir();source=root/'raw.jsonl';target=root/'raw.gz'
            source.write_bytes(b'{"sample":1}\n');target.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
            row=dict(source='evidence/raw.jsonl',compressed_path='evidence/raw.gz',source_bytes=source.stat().st_size,
                compressed_bytes=target.stat().st_size,source_SHA256=digest(source),compressed_SHA256=digest(target))
            self.assertEqual(verify_raw_archive(row,repo,root)['status'],'LOSSLESS_VERIFIED')
            source.write_bytes(b'{"sample":2}\n')
            with self.assertRaises(AssertionError):verify_raw_archive(row,repo,root)

    def test_path_escape_and_absolute_paths_are_rejected(self):
        with TemporaryDirectory() as folder:
            repo=Path(folder);root=repo/'evidence';root.mkdir();outside=repo/'outside';outside.write_bytes(b'x')
            for path in ['outside','evidence/../outside',str(outside)]:
                with self.assertRaises(AssertionError):confined_path(repo,root,path)

    def test_source_manifest_members_match_original_bytes(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'source.zip';member='poc_flashvep/mllm_epd_prefill_heavy_token_split/test.py';data=b'VALUE=4\n'
            rows=[dict(member=member,bytes=len(data),SHA256=hashlib.sha256(data).hexdigest())]
            with ZipFile(path,'x') as archive:
                archive.writestr(member,data);archive.writestr('source_manifest.json',json.dumps(rows))
            result=verify_source_archive(path,digest(path))
            self.assertEqual(result['campaign_member_SHA256']['test.py'],rows[0]['SHA256'])
            with self.assertRaises(AssertionError):verify_source_archive(path,'0'*64)

    def test_tensor_receipts_follow_all_three_recorded_schemas(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir();tensor=raw/'sample.safetensors';tensor.write_bytes(b'original')
            sha=digest(tensor)
            rows=[dict(tensor_file=str(tensor),tensor_SHA256=sha),dict(file=str(tensor),SHA256=sha),
                dict(path=str(tensor),SHA256=sha),dict(file='/dev/shm/ephemeral.safetensors',SHA256=sha)]
            (raw/'manifest.json').write_text(json.dumps(dict(rows=rows)))
            result=tensor_receipts(root)
            self.assertEqual(list(result),[tensor.resolve()]);self.assertEqual(len(result[tensor.resolve()]),3)


if __name__=='__main__':unittest.main()
