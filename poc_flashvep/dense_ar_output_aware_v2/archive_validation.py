"""Check decompressed archive bytes without extracting or changing raw evidence."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time
from .common import ROOT, write, digest, preserve
from .evidence_pack import ZSTD


def check_archive(path, expected):
    seen=set();size=0
    child=subprocess.Popen([ZSTD,'-q','-d','-c',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=child.stdout,mode='r|') as archive:
            for member in archive:
                assert member.isfile() and member.name in expected and member.name not in seen,member.name
                seen.add(member.name);record=expected[member.name]
                assert member.size==record['bytes'],member.name
                checksum=hashlib.sha256();stream=archive.extractfile(member)
                while data:=stream.read(8*1024**2):checksum.update(data)
                assert checksum.hexdigest()==record['SHA256'],member.name
                size+=member.size
        stderr=child.stderr.read().decode(errors='replace')
        assert child.wait()==0,stderr
        assert seen==set(expected),(path,set(expected)-seen)
        return dict(archive=str(path),passed=True,files=len(seen),uncompressed_bytes=size)
    finally:
        child.stdout.close();child.stderr.close()
        if child.poll() is None:child.terminate();child.wait()


def original_archive(parts):
    assert parts
    for part in parts:
        path=ROOT/part['path'];assert path.stat().st_size==part['bytes'] and digest(path)==part['SHA256']
    path=ROOT/re.sub(r'\.part[0-9]{3}$','',parts[0]['path'])
    assert path.exists(),path
    return path


def run():
    preserve();tasks=[];source_archives={}
    for folder,key in (('publication/evidence/manifests','included'),('publication/minimal_evidence/manifests','files')):
        for manifest in sorted((ROOT/folder).glob('*.json')):
            record=json.loads(manifest.read_text())
            tasks.append((original_archive(record['compressed_parts']),{r['path']:r for r in record[key]}))
            for source in record.get('exact_worker_source_archives',[]):
                source_archives[source['published']]=source['SHA256']
    for manifest in sorted((ROOT/'local_tensor_archives').glob('*.json')):
        record=json.loads(manifest.read_text())
        tasks.append((original_archive([record['archive']]),{r['path']:r for r in record['tensor_files']}))
    record=json.loads((ROOT/'local_dataset_archives/dataset_archive_manifest_v2.json').read_text())
    tasks.append((original_archive(record['parts']),{r['path']:r for r in record['files']}))
    for manifest in sorted((ROOT/'publication/derived_tables').glob('*.manifest.json')):
        record=json.loads(manifest.read_text())
        tasks.append((original_archive(record['compressed_parts']),{record['source']:dict(bytes=record['source_bytes'],SHA256=record['source_SHA256'])}))
    results=[];started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda task:check_archive(*task),tasks):results.append(result)
    assert all(digest(ROOT/path)==sha for path,sha in source_archives.items())
    write(ROOT/'analysis/lossless_archive_integrity_v2.json',dict(time=time.time(),passed=True,
        archives=len(results),files=sum(r['files'] for r in results),uncompressed_bytes=sum(r['uncompressed_bytes'] for r in results),
        exact_source_archives=len(source_archives),elapsed_seconds=time.time()-started,results=results,
        scope='Every decompressed tar member matches its original byte count and SHA256 manifest. '
            'No archive is extracted into the workspace. The original raw files remain unchanged. '
            'The full raw logs,tensors and bulk dataset archives remain local. Public structured evidence and derived tables remain separate.'))
    preserve();print('LOSSLESS_ARCHIVE_SHA256_VALIDATION',len(results),'PASS',flush=True)


if __name__=='__main__':run()
