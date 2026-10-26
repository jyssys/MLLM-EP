"""Lossless study archives. Never remove or replace different original evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import tarfile
from .common import ROOT, digest, write
from ..looped_moe_headroom.archive import pack

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extract',action='store_true')
    args=parser.parse_args()
    directory=ROOT/'archives'
    directory.mkdir(exist_ok=True)
    manifest=ROOT/'archive_manifest.json'
    if args.extract:
        value=json.loads(manifest.read_text())
        for record in value['archives']:
            archive=directory/record['archive']
            assert digest(archive)==record['sha256'],archive
            destination=ROOT/record['relative_source']
            destination.mkdir(parents=True,exist_ok=True)
            expected={v['path']:v['sha256'] for v in record['files']}
            with tarfile.open(archive,'r:xz') as stream:
                for member in stream.getmembers():
                    assert member.isfile() and member.name in expected,member.name
                    target=destination/member.name
                    assert target.resolve().is_relative_to(destination.resolve()),target
                    if target.exists():assert digest(target)==expected[member.name],target
                stream.extractall(destination,filter='data')
            for name,sha in expected.items():assert digest(destination/name)==sha,name
        print('RESTORED',len(value['archives']))
        return
    sources=sorted(p for p in (ROOT/'raw').iterdir() if p.is_dir())+[ROOT/'logs']
    assert all((p/'complete.json').exists() for p in sources if p.name!='logs'),'Archive only completed campaigns'
    previous={}
    for index in [ROOT/'state/preliminary_archives.json',manifest]:
        if index.exists():
            value=json.loads(index.read_text())
            for record in value if isinstance(value,list) else value['archives']:
                previous[record['archive']]=record
    def one(source):
        name=('logs' if source.name=='logs' else source.name)+'.tar.xz'
        old=previous.get(name)
        files=sorted(p for p in source.rglob('*') if p.is_file())
        if old and (directory/name).exists() and digest(directory/name)==old['sha256']:
            expected={r['path']:r['sha256'] for r in old['files']}
            if set(expected)=={str(p.relative_to(source)) for p in files} and all(digest(p)==expected[str(p.relative_to(source))] for p in files):
                record=dict(old,relative_source=str(source.relative_to(ROOT)))
                return record
        record=pack(source,directory/name)
        record['relative_source']=str(source.relative_to(ROOT))
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(one,sources))
    write(manifest,dict(archives=records,originals_retained=True,lossless_sha256_verified=True))
    print('ARCHIVED',len(records),sum(r['bytes'] for r in records))

if __name__=='__main__':main()
