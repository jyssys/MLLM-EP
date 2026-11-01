"""Publish large immutable evidence as verified lossless, bounded-size parts."""
import argparse
import gzip
import shutil
import tempfile
from .common import *

MANIFEST=ROOT/'archives/archive_manifest.json'
PART_BYTES=64*1024*1024


def manifest():return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def verified_originals():
    values=manifest();verified=set()
    for relative,entry in values.items():
        source=ROOT/relative
        assert source.exists() and source.stat().st_size==entry['original_bytes'] and digest(source)==entry['original_SHA256']
        for part in entry['parts']:
            path=ROOT/part['path']
            assert path.exists() and path.stat().st_size==part['bytes'] and digest(path)==part['SHA256']
        verified.add(source)
    return verified


def create(minimum_bytes=96_000_000):
    values=manifest();directory=ROOT/'archives';directory.mkdir(exist_ok=True)
    originals=[p for p in ROOT.rglob('*') if p.is_file() and 'archives' not in p.relative_to(ROOT).parts and
        p.stat().st_size>=minimum_bytes and str(p.relative_to(ROOT)) not in values]
    for source in originals:
        relative=str(source.relative_to(ROOT));original_hash=digest(source)
        stem=source.parent.name+'_'+source.name+'_'+original_hash[:12]+'.gz'
        with tempfile.TemporaryDirectory(prefix='atomic_evidence_compress_') as temporary:
            compressed=Path(temporary)/'evidence.gz'
            with source.open('rb') as incoming,compressed.open('wb') as outgoing:
                with gzip.GzipFile(fileobj=outgoing,mode='wb',filename='',mtime=0,compresslevel=6) as encoded:
                    shutil.copyfileobj(incoming,encoded,length=8*1024*1024)
            parts=[]
            with compressed.open('rb') as incoming:
                index=0
                while block:=incoming.read(PART_BYTES):
                    path=directory/(stem+f'.part{index:04d}');assert not path.exists(),'Never replace an evidence part'
                    with path.open('xb') as outgoing:outgoing.write(block)
                    parts.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,SHA256=digest(path)));index+=1
        assert digest(source)==original_hash,'Source evidence changed during archival'
        values[relative]=dict(original_bytes=source.stat().st_size,original_SHA256=original_hash,parts=parts,
            compression='gzip stream split into ordered 64-MiB parts',original_preserved_locally=True)
        write(MANIFEST,values);print('LOSSLESS_ARCHIVE',relative,len(parts),flush=True)
    verified_originals();return values


def restore(relative,destination):
    entry=manifest()[relative];target=Path(destination);assert not target.exists(),'Restore requires a new destination'
    with tempfile.TemporaryDirectory(prefix='atomic_evidence_restore_') as temporary:
        compressed=Path(temporary)/'evidence.gz'
        with compressed.open('xb') as outgoing:
            for part in entry['parts']:
                path=ROOT/part['path'];assert digest(path)==part['SHA256']
                with path.open('rb') as incoming:shutil.copyfileobj(incoming,outgoing)
        with gzip.open(compressed,'rb') as incoming,target.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing)
    assert target.stat().st_size==entry['original_bytes'] and digest(target)==entry['original_SHA256']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore');parser.add_argument('--destination');args=parser.parse_args()
    if args.restore:
        assert args.destination;restore(args.restore,args.destination)
    else:create()


if __name__=='__main__':main()
