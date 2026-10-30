"""Lossless gzip publication. Never delete or overwrite original evidence."""
import argparse
import gzip
import shutil
from pathlib import Path
from .common import ROOT,write,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore',action='store_true');args=parser.parse_args()
    manifest=[]
    paths=list((ROOT/'raw').glob('*/result.json'))+list((ROOT/'data').glob('*.json'))
    paths+=list((ROOT/'raw').glob('*/timeline.nsys-rep'))+list((ROOT/'raw').glob('*/timeline.sqlite'))
    for path in sorted(paths):
        packed=path.with_suffix(path.suffix+'.gz')
        if not packed.exists():
            with path.open('rb') as source,packed.open('wb') as target:
                with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as stream:
                    shutil.copyfileobj(source,stream)
        import hashlib
        h=hashlib.sha256()
        with gzip.open(packed,'rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=digest(path):raise RuntimeError('Compressed evidence does not match its original')
        manifest.append(dict(original=str(path.relative_to(ROOT)),compressed=str(packed.relative_to(ROOT)),
            original_bytes=path.stat().st_size,compressed_bytes=packed.stat().st_size,original_sha256=digest(path),
            compressed_sha256=digest(packed),exact_decompression=True))
    if args.restore:
        import json
        for row in json.loads((ROOT/'configs/artifact_storage.json').read_text()):
            path=ROOT/row['original'];packed=ROOT/row['compressed']
            if path.exists():
                if digest(path)!=row['original_sha256']:raise RuntimeError('Refuse to replace different local evidence')
                continue
            with gzip.open(packed,'rb') as source,path.open('xb') as target:shutil.copyfileobj(source,target)
            if digest(path)!=row['original_sha256']:raise RuntimeError('Restored evidence hash mismatch')
    elif manifest:
        write(ROOT/'configs/artifact_storage.json',manifest)
    print('VERIFIED',len(manifest),'lossless archives')


if __name__=='__main__':main()
