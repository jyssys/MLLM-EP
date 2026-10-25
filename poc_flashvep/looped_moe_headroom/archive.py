"""Preserve compact lossless evidence. Keep the original local files unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from .common import ROOT,digest,write

def pack(source,target):
    files=sorted(p for p in source.rglob('*') if p.is_file())
    records=[dict(path=str(p.relative_to(source)),bytes=p.stat().st_size,sha256=digest(p)) for p in files]
    temporary=target.with_suffix(target.suffix+'.tmp')
    with tarfile.open(temporary,'w:xz',preset=3) as archive:
        for p in files:
            archive.add(p,arcname=p.relative_to(source).as_posix(),recursive=False)
    temporary.replace(target)
    assert target.stat().st_size<95*1024*1024,('Archive exceeds per-file limit',target)
    with tarfile.open(target,'r:xz') as archive:
        for record in records:
            f=archive.extractfile(record['path'])
            h=hashlib.sha256()
            for data in iter(lambda:f.read(8<<20),b''):
                h.update(data)
            assert h.hexdigest()==record['sha256'],record
    return dict(source=str(source),archive=target.name,bytes=target.stat().st_size,sha256=digest(target),files=records)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--extract',action='store_true')
    a=p.parse_args()
    directory=ROOT/'archives'
    directory.mkdir(exist_ok=True)
    if a.extract:
        manifest=json.loads((ROOT/'archive_manifest.json').read_text())
        for record in manifest['archives']:
            target=ROOT/'logs' if record['archive']=='logs.tar.xz' else ROOT/'raw'/record['archive'].removesuffix('.tar.xz')
            target.mkdir(parents=True,exist_ok=True)
            assert digest(directory/record['archive'])==record['sha256']
            with tarfile.open(directory/record['archive'],'r:xz') as archive:
                for member in archive.getmembers():
                    existing=target/member.name
                    if existing.exists():
                        expected=next(r['sha256'] for r in record['files'] if r['path']==member.name)
                        assert digest(existing)==expected,('Refuse overwrite',existing)
                archive.extractall(target,filter='data')
        return
    records=[]
    for source in sorted((ROOT/'raw').iterdir()):
        if source.is_dir():
            records.append(pack(source,directory/f'{source.name}.tar.xz'))
    records.append(pack(ROOT/'logs',directory/'logs.tar.xz'))
    write(ROOT/'archive_manifest.json',dict(archives=records,originals_retained=True,lossless_sha256_verified=True))
    full=json.loads((ROOT/'workload_manifest.json').read_text())
    public=dict(full)
    public['rows']=[{k:v for k,v in row.items() if k not in ('text','question')} for row in full['rows']]
    public['local_full_manifest_sha256']=digest(ROOT/'workload_manifest.json')
    public['redistribution']='Raw language passages stay local. IDs, source revisions, and text hashes enable reconstruction.'
    write(ROOT/'workload_manifest_public.json',public)
    print('ARCHIVED',len(records),sum(r['bytes'] for r in records))

if __name__=='__main__':
    main()
