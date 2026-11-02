"""Require exact Chrome preservation and reject unnecessary event-tree work."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from .profile_trace_export import exporter,ArtifactTar


class Export(unittest.TestCase):
    def test_lossless_archive_preserves_member_bytes(self):
        import io,tarfile
        data=b'actual kernel trace bytes\n'*100
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'archive.tar.gz';wrapper=ArtifactTar(tarfile)
            with wrapper.open(path,'w:gz') as archive:
                member=tarfile.TarInfo('trace.json');member.size=len(data)
                archive.addfile(member,io.BytesIO(data))
            with wrapper.open(path,'r:gz') as archive:
                self.assertEqual(archive.extractfile('trace.json').read(),data)
    def test_cuda_only_preserves_exact_trace_without_native_tree(self):
        payload='{"traceEvents":[{"cat":"kernel","ts":3,"dur":2}]}'
        calls=[]
        profiler=types.SimpleNamespace(export_chrome_trace=lambda path:Path(path).write_text(payload),
            wait_for_exports=lambda:calls.append('wait'),
            events=lambda:(_ for _ in ()).throw(AssertionError('Native CPU tree must not run')))
        config=types.SimpleNamespace(activities=('cuda',),record_shapes=False,with_stack=False,profile_memory=False,with_flops=False)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            exporter(lambda *a:self.fail('Wrong fallback'))(profiler,path,config,'cuda:0')
            self.assertEqual((path/'trace.json').read_text(),payload)
            self.assertEqual(json.loads((path/'native_event_table_status.json').read_text())['status'],'NOT_COLLECTED')
            self.assertFalse((path/'events.jsonl').exists())
        self.assertEqual(calls,['wait'])

    def test_CPU_or_extended_collection_keeps_native_export(self):
        called=[]
        config=types.SimpleNamespace(activities=('cpu','cuda'),record_shapes=False,with_stack=False,profile_memory=False,with_flops=False)
        exporter(lambda *args:called.append(args))('p','s',config,'d')
        self.assertEqual(len(called),1)


if __name__=='__main__':unittest.main()
