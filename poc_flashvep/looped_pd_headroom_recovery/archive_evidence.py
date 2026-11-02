"""Publish oversized evidence losslessly. Retain every local original."""
import argparse
import gzip
import hashlib
from .common import *

CHUNK=64*1024**2


def build():
    target=ROOT/'archives';target.mkdir(exist_ok=True)
    previous=target/'manifest.json'
    known=json.loads(previous.read_text()) if previous.exists() else {'originals':[]}
    records={v['original']:v for v in known['originals']}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or target in path.parents:continue
        if path.stat().st_size<48*1024**2 or path.name.endswith(('.tmp','.lock')):continue
        if path.name=='burner.log':continue  # A live resource log is not frozen evidence.
        relative=str(path.relative_to(ROOT));sha=digest(path)
        old=records.get(relative)
        if old and old['SHA256']==sha and all((target/p['file']).exists() for p in old['parts']):continue
        parts=[]
        with path.open('rb') as stream:
            index=0
            while True:
                block=stream.read(CHUNK)
                if not block:break
                filename=f'{sha}.part{index:04d}.gz'
                result=target/filename
                if not result.exists():
                    with result.open('wb') as raw:
                        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as compressed:
                            compressed.write(block)
                with gzip.open(result,'rb') as verify:
                    assert verify.read()==block,'Archive part does not match its original'
                parts.append(dict(file=filename,original_chunk_bytes=len(block),compressed_bytes=result.stat().st_size,
                    compressed_SHA256=digest(result),original_chunk_SHA256=hashlib.sha256(block).hexdigest()))
                index+=1
        records[relative]=dict(original=relative,bytes=path.stat().st_size,SHA256=sha,parts=parts,
            reconstruction='Decompress each numbered gzip member. Concatenate decompressed bytes in part order. Verify the original SHA256.')
    manifest=dict(created=time.time(),originals=sorted(records.values(),key=lambda v:v['original']),
        scope='Lossless chunked copies only. No original file is deleted or changed. Oversized originals are replaced by these copies only in Git publication.')
    write(previous,manifest)
    print(json.dumps(dict(originals=len(records),parts=sum(len(v['parts']) for v in records.values()))))
    return manifest


def restore(destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'archives/manifest.json').read_text())
    for value in manifest['originals']:
        target=(destination/value['original']).resolve()
        assert destination in target.parents,'Archive target escapes the destination'
        if target.exists():
            assert digest(target)==value['SHA256'],'Refuse to overwrite a different existing file'
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.restore.tmp')
        with temporary.open('xb') as output:
            for part in value['parts']:
                source=ROOT/'archives'/part['file']
                assert digest(source)==part['compressed_SHA256']
                with gzip.open(source,'rb') as stream:
                    while True:
                        block=stream.read(1024**2)
                        if not block:break
                        output.write(block)
        assert digest(temporary)==value['SHA256']
        temporary.replace(target)
    print(json.dumps(dict(restored=len(manifest['originals']),destination=str(destination))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','restore'));p.add_argument('--destination')
    a=p.parse_args()
    if a.action=='build':build()
    else:
        assert a.destination,'Restoration requires an explicit destination'
        restore(a.destination)
