"""Retain immutable evidence in lossless, size-limited Git artifacts."""
import gzip,hashlib,json,os,tarfile,tempfile,time
from .common import *

CHUNK=8*1024**2
SHARD=48*1024**2


def main():
    initialize();assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    target=ROOT/'evidence_bundle';assert not target.exists(),'Do not overwrite a bundle'
    live=ROOT/'burner.log';frozen=ROOT/'state/burner_log_at_closure.log'
    if live.exists() and not frozen.exists():
        size=live.stat().st_size
        with live.open('rb') as src,frozen.open('wb') as out:
            remaining=size
            while remaining:
                data=src.read(min(1024**2,remaining));assert data
                out.write(data);remaining-=len(data)
    target.mkdir();files=[];blobs={};excluded=[]
    with tempfile.TemporaryDirectory(prefix='pd_discovery_chunks_') as temp:
        staging=Path(temp)
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or target in path.parents:continue
            relative=path.relative_to(ROOT)
            if {'worker_source','__pycache__'}&set(relative.parts) or path.suffix in ('.pyc','.lock') or path.name in (
                'burner.log','final_remote_verification.json','publication_receipt.json','bundle_live_tree_check.json'):
                excluded.append(str(relative));continue
            h=hashlib.sha256();chunks=[];size=0
            with path.open('rb') as src:
                for data in iter(lambda:src.read(CHUNK),b''):
                    h.update(data);size+=len(data);sha=hashlib.sha256(data).hexdigest();chunks.append(sha)
                    if sha not in blobs:
                        item=staging/sha;item.write_bytes(data);blobs[sha]=(item,len(data))
            files.append(dict(path=str(relative),sha256=h.hexdigest(),bytes=size,chunks=chunks))
        groups=[];batch=[];amount=0
        for sha,(path,size) in blobs.items():
            if batch and amount+size>SHARD:groups.append(batch);batch=[];amount=0
            batch.append(sha);amount+=size
        if batch:groups.append(batch)
        shards=[]
        for index,batch in enumerate(groups):
            path=target/f'raw_{index:04d}.tar.gz'
            with path.open('wb') as out,gzip.GzipFile(fileobj=out,mode='wb',compresslevel=6,mtime=0) as z:
                with tarfile.open(fileobj=z,mode='w|') as archive:
                    for sha in batch:
                        item,size=blobs[sha];info=tarfile.TarInfo('chunks/'+sha)
                        info.size=size;info.mode=0o644;info.mtime=0
                        with item.open('rb') as src:archive.addfile(info,src)
            assert path.stat().st_size<95*1024**2
            shards.append(dict(path=path.name,bytes=path.stat().st_size,sha256=digest(path)))
    manifest=dict(format='LOSSLESS_SHA256_CHUNKED_EVIDENCE_V2',chunk_bytes=CHUNK,
        files=files,shards=shards,unique_chunks=len(blobs),original_bytes=sum(r['bytes'] for r in files),
        bundled_bytes=sum(r['bytes'] for r in shards),excluded=excluded,
        scope='Listed files reconstruct exactly from ordered SHA256 chunks. Original raw files remain unchanged. Extracted worker_source trees are redundant with retained per-run source ZIP files. Bytecode, lock files, and the live Burner log are excluded. The frozen log contains the exact byte prefix at closure.')
    write(target/'manifest.json',manifest)
    from ..completion_aware_loop_pd.evidence_bundle import verify
    verify(target)
    write(ROOT/'state/bundle_verification.json',dict(time=time.time(),verified=True,
        manifest_SHA256=digest(target/'manifest.json'),files=len(files),
        original_bytes=manifest['original_bytes'],bundled_bytes=manifest['bundled_bytes']))
    print(json.dumps({k:manifest[k] for k in ('original_bytes','bundled_bytes','unique_chunks')}))


if __name__=='__main__':main()
