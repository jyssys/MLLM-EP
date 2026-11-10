"""Package large kernel traces losslessly without changing the raw evidence tree."""
import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from .common import ROOT,write,digest

CHUNK=8*1024**2
SHARD=48*1024**2


def pack(source,target):
    assert not target.exists(),'Use a new immutable bundle namespace'
    target.mkdir(parents=True)
    files=[];blobs={};excluded=[]
    with tempfile.TemporaryDirectory(prefix='loop_pd_evidence_chunks_') as temp:
        staging=Path(temp)
        for path in sorted(source.rglob('*')):
            if not path.is_file() or target in path.parents:continue
            relative=path.relative_to(source)
            if '__pycache__' in relative.parts or path.suffix in ('.pyc','.lock') or path.name in (
                    'burner.log','final_remote_verification.json','bundle_live_tree_check.json'):
                excluded.append(str(relative));continue
            h=hashlib.sha256();chunks=[];size=0
            with path.open('rb') as f:
                for data in iter(lambda:f.read(CHUNK),b''):
                    h.update(data);size+=len(data);sha=hashlib.sha256(data).hexdigest();chunks.append(sha)
                    if sha not in blobs:
                        p=staging/sha;p.write_bytes(data);blobs[sha]=dict(path=p,bytes=len(data))
            files.append(dict(path=str(relative),sha256=h.hexdigest(),bytes=size,chunks=chunks))
        groups=[];batch=[];amount=0
        for sha,row in blobs.items():
            if batch and amount+row['bytes']>SHARD:groups.append(batch);batch=[];amount=0
            batch.append(sha);amount+=row['bytes']
        if batch:groups.append(batch)
        shards=[]
        for index,batch in enumerate(groups):
            path=target/f'raw_{index:04d}.tar.gz'
            with path.open('wb') as out,gzip.GzipFile(fileobj=out,mode='wb',compresslevel=6,mtime=0) as zipped:
                with tarfile.open(fileobj=zipped,mode='w|') as archive:
                    for sha in batch:
                        info=tarfile.TarInfo('chunks/'+sha);info.size=blobs[sha]['bytes'];info.mode=0o644;info.mtime=0
                        with blobs[sha]['path'].open('rb') as f:archive.addfile(info,f)
            assert path.stat().st_size<95*1024**2
            shards.append(dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size))
    manifest=dict(format='LOSSLESS_SHA256_CHUNKED_EVIDENCE_V2',chunk_bytes=CHUNK,files=files,shards=shards,
        unique_chunks=len(blobs),original_bytes=sum(r['bytes'] for r in files),
        bundled_bytes=sum(r['bytes'] for r in shards),excluded=excluded,
        scope='Every listed file is recoverable from its ordered SHA256 chunks. Original raw files remain unchanged. Python bytecode, lock files, and the live Burner log are excluded. A frozen Burner log is a separate listed file.')
    write(target/'manifest.json',manifest);return manifest


def verify(target,destination=None):
    manifest=json.loads((target/'manifest.json').read_text());seen={}
    if destination is not None:
        assert not destination.exists(),'Extraction requires a new directory'
        destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='loop_pd_verify_chunks_') as temp:
        staging=Path(temp)
        for shard in manifest['shards']:
            path=target/shard['path'];assert digest(path)==shard['sha256']
            with tarfile.open(path,'r:gz') as archive:
                for member in archive:
                    assert member.isfile() and member.name.startswith('chunks/')
                    sha=member.name.split('/')[1];assert len(sha)==64 and sha not in seen
                    p=staging/sha;h=hashlib.sha256();size=0
                    with archive.extractfile(member) as src,p.open('wb') as out:
                        for data in iter(lambda:src.read(1024**2),b''):h.update(data);size+=len(data);out.write(data)
                    assert h.hexdigest()==sha;seen[sha]=size
        for row in manifest['files']:
            relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
            h=hashlib.sha256();size=0;output=None
            if destination is not None:
                p=destination/relative;p.parent.mkdir(parents=True,exist_ok=True);output=p.open('wb')
            try:
                for sha in row['chunks']:
                    assert sha in seen
                    with (staging/sha).open('rb') as f:
                        for data in iter(lambda:f.read(1024**2),b''):
                            h.update(data);size+=len(data)
                            if output is not None:output.write(data)
            finally:
                if output is not None:output.close()
            assert h.hexdigest()==row['sha256'] and size==row['bytes']
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--extract',type=Path);a=p.parse_args()
    target=ROOT/'evidence_bundle'
    if a.verify or a.extract:v=verify(target,a.extract)
    else:
        live=ROOT/'burner.log';frozen=ROOT/'state/burner_log_at_closure.log'
        if live.exists() and not frozen.exists():shutil.copyfile(live,frozen)
        v=pack(ROOT,target);verify(target)
    print(json.dumps({k:v[k] for k in ('format','original_bytes','bundled_bytes','unique_chunks')}))


if __name__=='__main__':main()
