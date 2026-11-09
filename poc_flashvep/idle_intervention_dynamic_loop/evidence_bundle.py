"""Package every evidence byte with deduplication. Keep the local raw tree intact."""
import argparse
import gzip
import hashlib
import io
import tarfile
from .common import *


def build():
    target=ROOT/'evidence_bundle';target.mkdir(exist_ok=True)
    assert not (target/'manifest.json').exists(),'Keep a published bundle immutable'
    # The approved burners remain active after closure. Freeze their operational
    # log once. Do not hash a log while a surviving engine can append to it.
    live_log=ROOT/'burner.log'
    frozen_log=ROOT/'state/burner_log_at_closure.log'
    if live_log.exists() and not frozen_log.exists():
        import shutil
        shutil.copyfile(live_log,frozen_log)
    records=[];unique={};excluded=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or target in path.parents:continue
        relative=path.relative_to(ROOT)
        if '__pycache__' in path.parts or path.suffix in ('.pyc','.lock'):
            excluded.append(str(relative));continue
        source=frozen_log if path==live_log else path
        sha=digest(source);size=source.stat().st_size
        records.append(dict(path=str(relative),sha256=sha,bytes=size));unique.setdefault(sha,source)
    batches=[];batch=[];amount=0
    for sha,path in unique.items():
        size=path.stat().st_size
        assert size<90*2**20,'A raw artifact exceeds the publication shard limit'
        if batch and amount+size>48*2**20:batches.append(batch);batch=[];amount=0
        batch.append((sha,path));amount+=size
    if batch:batches.append(batch)
    shards=[]
    for index,batch in enumerate(batches):
        path=target/f'raw_{index:03d}.tar.gz'
        assert not path.exists(),'An unfinished bundle requires a new explicit namespace'
        with path.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',mtime=0,compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|') as archive:
                for sha,source in batch:
                    size=source.stat().st_size;metadata=tarfile.TarInfo('blobs/'+sha)
                    metadata.size=size;metadata.mode=0o644;metadata.mtime=0
                    with source.open('rb') as data:archive.addfile(metadata,data)
        assert path.stat().st_size<95*2**20
        shards.append(dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size))
    manifest=dict(format='SHA256 content-addressed lossless raw evidence v1',files=records,
        shards=shards,unique_blobs=len(unique),original_bytes=sum(r['bytes'] for r in records),
        bundled_bytes=sum(r['bytes'] for r in shards),excluded_non_evidence=excluded,
        scope='Every raw/configuration/source-snapshot byte remains recoverable. Duplicate files share one blob. Only Python bytecode and advisory lock files are excluded. The active burner log uses its closure snapshot. Later burner messages are outside the scientific evidence interval. The original local raw tree is unchanged.')
    write(target/'manifest.json',manifest)
    verify(target)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','excluded_non_evidence')},indent=2))


def verify(target):
    manifest=json.loads((target/'manifest.json').read_text());seen={}
    for item in manifest['shards']:
        path=target/item['path'];assert digest(path)==item['sha256']
        with tarfile.open(path,'r:gz') as archive:
            for member in archive:
                assert member.isfile() and member.name.startswith('blobs/')
                sha=member.name.split('/')[1];stream=archive.extractfile(member)
                h=hashlib.sha256();size=0
                for block in iter(lambda:stream.read(2**20),b''):h.update(block);size+=len(block)
                assert h.hexdigest()==sha and sha not in seen
                seen[sha]=size
    assert all(seen.get(r['sha256'])==r['bytes'] for r in manifest['files'])
    return manifest


def extract(target,destination):
    manifest=verify(target);destination=destination.resolve()
    assert not destination.exists(),'Extraction requires a new directory'
    destination.mkdir(parents=True);lookup={}
    for row in manifest['files']:
        relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
        lookup.setdefault(row['sha256'],[]).append(destination/relative)
    for item in manifest['shards']:
        with tarfile.open(target/item['path'],'r:gz') as archive:
            for member in archive:
                data=archive.extractfile(member).read()
                for output in lookup[member.name.split('/')[1]]:
                    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(data)
    assert all(digest(destination/r['path'])==r['sha256'] for r in manifest['files'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--extract',type=Path);a=p.parse_args()
    folder=ROOT/'evidence_bundle'
    if a.extract:extract(folder,a.extract)
    elif a.verify:print(json.dumps(dict(verified_files=len(verify(folder)['files']))))
    else:build()
