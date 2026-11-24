"""Preserve the actual compiled artifacts used by the numerical controls."""
import gzip
import hashlib
import tarfile
import time

from .common import ROOT,digest,preserve,write


def main():
    preserve()
    variants=[
        ('prefill','e981dc0d0d'),('decode','6a1816e5cb'),
        ('prefill','torch_aot_compile/4b2ad6ec0c483166538a036c0666e49b4d2fedf4ed4406e914709498f914743b'),
        ('decode','torch_aot_compile/8e4c8532496d76692e3406e97bbb0c693b184be17ca784a1743c5f1550710fc3')]
    paths=[]
    for role,variant in variants:
        folder=ROOT/'runtime_cache'/role/'torch_compile_cache'/variant
        assert folder.is_dir()
        paths.extend(p for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.lock'))
    paths=sorted(set(paths))
    records=[dict(path=str(p.relative_to(ROOT)),SHA256=digest(p),bytes=p.stat().st_size) for p in paths]
    destination=ROOT/'archives/token_C17_compiled_artifacts_v0.tar.gz'
    with destination.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as archive:
                for p in paths:archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    expected={r['path']:r for r in records};seen=set()
    with tarfile.open(destination,'r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected
            record=expected[member.name];source=archive.extractfile(member);assert source is not None
            hasher=hashlib.sha256()
            while block:=source.read(8*1024**2):hasher.update(block)
            assert hasher.hexdigest()==record['SHA256']==digest(ROOT/member.name)
            assert member.size==record['bytes']
            seen.add(member.name)
    assert seen==set(expected)
    write(ROOT/'analysis/token_C17_compiler_evidence_v0.json',dict(time=time.time(),passed=True,
        files=records,archive=str(destination),archive_SHA256=digest(destination),
        archive_bytes=destination.stat().st_size,source_bytes=sum(r['bytes'] for r in records),
        scope='Actual P/D backbone and AOT artifacts logged by C17R1. No model weights or media. '
            'The original FX graphs are identical by SHA256. Equal FX graphs do not imply equal lowered kernels. '
            'All artifact bytes remain local and have lossless archive verification.'))
    preserve();print('COMPILER_ARTIFACTS_PRESERVED',len(records),destination.stat().st_size,flush=True)


if __name__=='__main__':main()
