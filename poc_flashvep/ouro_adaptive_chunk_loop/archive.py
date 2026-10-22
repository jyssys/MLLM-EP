"""Publish portable raw metrics. Keep large tensors and full traces local."""
import concurrent.futures
import json
from pathlib import Path
import tarfile
from collections import defaultdict
from .common import ROOT,write,digest

def category(folder):
    name=folder.name
    if name.startswith('p2_P'):
        parts=name.split('_')
        return '_'.join((parts[0],parts[1],parts[-2]))
    if name.startswith('p0_'):
        return 'p0_contention'
    if name.startswith('p1_quality'):
        return 'quality_natural'
    if name.startswith('p2_observer'):
        return 'measurement_controls'
    return 'numerical_diagnostics'

def main():
    groups=defaultdict(list);local=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob('*')):
            if not path.is_file():
                continue
            if path.suffix=='.pt' or path.name.endswith('_heavy_trace.json'):
                local.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path),
                    reason='Large tensor/profiler evidence remains local. Summary and hash are portable.'))
            else:
                groups[category(folder)].append(path)
    groups['worker_logs']=sorted((ROOT/'logs').glob('*.log'))
    # Two representative complete profiler traces remain portable in compressed form.
    for gpu in (0,1):
        path=ROOT/'raw'/f'p2_observer_control_gpu{gpu}_v2'/'grouped_C512_heavy_trace.json'
        if path.exists():
            groups[f'grouped_profiler_gpu{gpu}']=[path]
    destination=ROOT/'raw_archives';destination.mkdir(exist_ok=True)
    def pack(item):
        name,files=item
        output=destination/f'{name}.tar.gz'
        inventory=[]
        for path in files:
            inventory.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path)))
        with tarfile.open(output,'w:gz',compresslevel=5) as archive:
            for path in files:
                archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
        # Verify decompressed member bytes, not only the container checksum.
        with tarfile.open(output,'r:gz') as archive:
            entries={r['path']:r for r in inventory}
            for member in archive:
                stream=archive.extractfile(member)
                actual=__import__('hashlib').file_digest(stream,'sha256').hexdigest()
                assert actual==entries[member.name]['sha256'],member.name
        assert output.stat().st_size<95*2**20, ('ARCHIVE_TOO_LARGE_FOR_NORMAL_GIT',output)
        return dict(path=str(output.relative_to(ROOT)),bytes=output.stat().st_size,sha256=digest(output),members=inventory)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest=list(pool.map(pack,groups.items()))
    write(ROOT/'raw_archive_manifest.json',dict(archives=manifest,verification='Every decompressed member SHA256 checked',
        excludes='Large PT captures and unselected heavy traces remain local. No model weights or cache credentials included.'))
    write(ROOT/'local_large_artifacts.json',local)
    print(json.dumps(dict(archives=len(manifest),archive_bytes=sum(r['bytes'] for r in manifest),local_large_files=len(local)),indent=2))

if __name__=='__main__':
    main()
